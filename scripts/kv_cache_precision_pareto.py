#!/usr/bin/env python3
"""Hardware-real static KV-cache precision screen using direct FlashInfer attention."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/kv_cache_precision_pareto"
CONFIG = OUT / "experiment_config.json"
PROTOCOL = OUT / "protocol.md"
HISTORY = ROOT / "results/phase1a_run.json"
SOURCES = [
    "scripts/kv_cache_precision_pareto.py",
    "scripts/prepare_kv_cache_inputs.py",
    "scripts/run_kv_cache_precision_pareto.sh",
    "results/kv_cache_precision_pareto/experiment_config.json",
    "results/kv_cache_precision_pareto/protocol.md",
]
MODES = ("BF16", "FP8_E4M3", "FP8_E5M2")
RAW_FIELDS = [
    "mode", "phase", "prompt_length", "concurrency", "repeat", "request_id", "token_index",
    "status", "prompt_tokens", "output_tokens", "nll", "baseline_nll", "delta_nll", "kl",
    "em", "ttft_ms", "tpot_ms", "completion_ms", "wall_ms", "error",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_array(value: np.ndarray) -> str:
    return hash_bytes(np.ascontiguousarray(value).tobytes())


def save_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def metadata_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().nbytes())


def mode_dtype(mode: str) -> torch.dtype:
    return {
        "BF16": torch.bfloat16,
        "FP8_E4M3": torch.float8_e4m3fn,
        "FP8_E5M2": torch.float8_e5m2,
    }[mode]


def quantize_cache(value: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "BF16":
        return value.to(torch.bfloat16)
    # A fixed unit calibration scale is frozen in the protocol. The FlashInfer
    # kernel receives the same scale and reads this one-byte tensor directly.
    return value.float().to(mode_dtype(mode))


def load_inputs(out: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((out / "prompt_manifest.json").read_text())
    arrays: dict[str, np.ndarray] = {}
    with np.load(out / "prompts.npz", allow_pickle=False) as data:
        for name in data.files:
            arrays[name] = data[name]
    require(manifest["status"] == "prepared_no_model_forward", "Inputs were not prepared by the fixed CPU preparer")
    for name, expected in manifest["array_sha256"].items():
        require(name in arrays and hash_array(arrays[name]) == expected, f"Prompt array hash mismatch: {name}")
    return manifest, arrays


def state_inventory(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return dict(list(model.named_parameters()) + list(model.named_buffers()))


class StaticPagedKV:
    """Fixed-capacity paged cache; attention receives its raw storage tensors."""

    def __init__(self, batch: int, max_length: int, mode: str, workspace: torch.Tensor):
        self.batch = batch
        self.max_length = max_length
        self.mode = mode
        self.page_size = 16
        self.num_pages_per_request = math.ceil(max_length / self.page_size)
        self.num_pages = batch * self.num_pages_per_request
        dtype = mode_dtype(mode)
        self.num_layers = 36
        shape = (self.num_layers, self.num_pages, self.page_size, 8, 128)
        # A real decoder cache needs an independent K/V payload for every layer;
        # sharing one payload across layers would make later decode steps read
        # another layer's history and would under-report memory.
        self.k = torch.empty(shape, device="cuda", dtype=dtype)
        self.v = torch.empty(shape, device="cuda", dtype=dtype)
        self.scale = 1.0 if mode != "BF16" else None
        self.workspace = workspace
        self.prefill = None
        self.decode = None
        self._make_wrappers()

    @property
    def payload_bytes(self) -> int:
        return tensor_bytes(self.k) + tensor_bytes(self.v)

    @property
    def scales_bytes(self) -> int:
        return 0

    def _make_wrappers(self) -> None:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
        self.prefill = BatchPrefillWithPagedKVCacheWrapper(self.workspace, kv_layout="NHD", backend="auto")
        self.decode = BatchDecodeWithPagedKVCacheWrapper(self.workspace, kv_layout="NHD", backend="auto")

    def _last_page_lengths(self, lengths: int) -> torch.Tensor:
        value = lengths % self.page_size or self.page_size
        return torch.full((self.batch,), value, device="cuda", dtype=torch.int32)

    def write(self, key: torch.Tensor, value: torch.Tensor, start: int, layer_idx: int = 0) -> None:
        # key/value: [batch, sequence, kv_heads, head_dim]
        require(0 <= layer_idx < self.num_layers, "Unexpected layer index")
        bsz, sequence, heads, dim = key.shape
        require((bsz, heads, dim) == (self.batch, 8, 128), "Unexpected KV projection shape")
        require(start + sequence <= self.max_length, "Static cache capacity exceeded")
        kq = quantize_cache(key, self.mode)
        vq = quantize_cache(value, self.mode)
        for offset in range(sequence):
            position = start + offset
            page = position // self.page_size
            slot = position % self.page_size
            page_ids = torch.arange(self.batch, device="cuda") * self.num_pages_per_request + page
            self.k[layer_idx, page_ids, slot] = kq[:, offset]
            self.v[layer_idx, page_ids, slot] = vq[:, offset]

    def _page_metadata(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        used = math.ceil(length / self.page_size)
        indptr = torch.arange(self.batch + 1, device="cuda", dtype=torch.int32) * used
        indices = torch.cat([
            torch.arange(b * self.num_pages_per_request, b * self.num_pages_per_request + used,
                         device="cuda", dtype=torch.int32)
            for b in range(self.batch)
        ])
        return indptr, indices

    def plan_prefill(self, prompt_length: int) -> None:
        qo_indptr = torch.arange(self.batch + 1, device="cuda", dtype=torch.int32) * prompt_length
        paged_indptr, paged_indices = self._page_metadata(prompt_length)
        last = self._last_page_lengths(prompt_length)
        seq = torch.full((self.batch,), prompt_length, device="cuda", dtype=torch.int32)
        self.prefill.plan(
            qo_indptr, paged_indptr, paged_indices, last,
            num_qo_heads=32, num_kv_heads=8, head_dim_qk=128, page_size=self.page_size,
            head_dim_vo=128, causal=True, q_data_type=torch.bfloat16,
            kv_data_type=mode_dtype(self.mode), seq_lens=seq, seq_lens_q=seq,
            # FlashInfer 0.5.2 requires host metadata for its non-CUDA-graph plan;
            # leaving these optional maxima unset avoids its unbound-local bug.
            disable_split_kv=True,
        )

    def attend_prefill(self, q: torch.Tensor, prompt_length: int, layer_idx: int = 0) -> torch.Tensor:
        # q: [batch, sequence, q_heads, head_dim]
        require(0 <= layer_idx < self.num_layers, "Unexpected layer index")
        self.plan_prefill(prompt_length)
        q_flat = q.reshape(self.batch * prompt_length, 32, 128)
        kwargs = {} if self.mode == "BF16" else {"k_scale": 1.0, "v_scale": 1.0}
        result = self.prefill.run(q_flat, (self.k[layer_idx], self.v[layer_idx]), **kwargs)
        return result.reshape(self.batch, prompt_length, 32, 128)

    def attend_decode(self, q: torch.Tensor, current_length: int, layer_idx: int = 0) -> torch.Tensor:
        # q: [batch, q_heads, head_dim], cache includes the newly written token.
        require(0 <= layer_idx < self.num_layers, "Unexpected layer index")
        paged_indptr, paged_indices = self._page_metadata(current_length)
        last = self._last_page_lengths(current_length)
        seq = torch.full((self.batch,), current_length, device="cuda", dtype=torch.int32)
        self.decode.plan(
            paged_indptr, paged_indices, last, num_qo_heads=32, num_kv_heads=8,
            head_dim=128, page_size=self.page_size, q_data_type=torch.bfloat16,
            kv_data_type=mode_dtype(self.mode), seq_lens=seq, disable_split_kv=True,
        )
        kwargs = {} if self.mode == "BF16" else {"k_scale": 1.0, "v_scale": 1.0}
        return self.decode.run(q, (self.k[layer_idx], self.v[layer_idx]), **kwargs)

    def profile(self) -> dict:
        return {
            "mode": self.mode, "batch": self.batch, "max_length": self.max_length,
            "num_layers": self.num_layers, "page_size": self.page_size,
            "pages_per_request": self.num_pages_per_request,
            "k_shape": list(self.k.shape), "v_shape": list(self.v.shape),
            "cache_dtype": str(self.k.dtype), "k_bytes": tensor_bytes(self.k),
            "v_bytes": tensor_bytes(self.v), "scale_bytes": self.scales_bytes,
            "cache_payload_bytes": self.payload_bytes, "scale": self.scale,
            "configured_payload_bytes": self.payload_bytes + self.scales_bytes,
            "attention_api": "flashinfer.BatchPrefillWithPagedKVCacheWrapper + BatchDecodeWithPagedKVCacheWrapper",
            "direct_kernel_read": True,
            "whole_cache_bf16_restore": False,
            "cache_tensor_passed_to_kernel_dtype": str(mode_dtype(self.mode)),
        }


def prepare_model() -> torch.nn.Module:
    history = json.loads(HISTORY.read_text())
    model_path = Path(history["model_source"])
    for name, expected in history["checkpoint_sha256"].items():
        require(hash_file(model_path / name) == expected, f"Pinned checkpoint changed: {name}")
    require(hash_file(model_path / "config.json") == history["checkpoint_config_sha256"], "Pinned model config changed")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True,
        trust_remote_code=False, attn_implementation="sdpa",
    )
    model.eval().requires_grad_(False)
    require(all(p.dtype == torch.bfloat16 for p in model.parameters()), "Weights are not all BF16")
    require(len(model.model.layers) == 36 and model.config.num_attention_heads == 32 and
            model.config.num_key_value_heads == 8, "Unexpected Qwen3 architecture")
    require(model.lm_head.weight is model.model.embed_tokens.weight, "Unexpected untied LM head")
    return model


def qwen_attention(model: torch.nn.Module, hidden: torch.Tensor, position_ids: torch.Tensor,
                   cache: StaticPagedKV, start: int) -> torch.Tensor:
    """One model forward over all layers, with direct paged-cache attention."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    input_shape = hidden.shape[:-1]
    bsz, seq_len, _ = hidden.shape
    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden
        hidden_norm = layer.input_layernorm(hidden)
        attn = layer.self_attn
        q = attn.q_norm(attn.q_proj(hidden_norm).view(bsz, seq_len, 32, 128)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden_norm).view(bsz, seq_len, 8, 128)).transpose(1, 2)
        v = attn.v_proj(hidden_norm).view(bsz, seq_len, 8, 128).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        k_nhd = k.transpose(1, 2).contiguous()
        v_nhd = v.transpose(1, 2).contiguous()
        cache.write(k_nhd, v_nhd, start, layer_idx=layer_idx)
        if seq_len > 1:
            attn_out = cache.attend_prefill(q.transpose(1, 2).contiguous(), seq_len, layer_idx=layer_idx)
        else:
            attn_out = cache.attend_decode(q[:, :, 0, :], start + 1, layer_idx=layer_idx).unsqueeze(1)
        hidden = residual + attn.o_proj(attn_out.reshape(*input_shape, -1))
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
    return model.model.norm(hidden)


def forward_logits(model: torch.nn.Module, input_ids: torch.Tensor, cache: StaticPagedKV,
                   start: int) -> torch.Tensor:
    positions = torch.arange(start, start + input_ids.shape[1], device="cuda", dtype=torch.long)
    positions = positions.unsqueeze(0).expand(input_ids.shape[0], -1)
    hidden = qwen_attention(model, model.model.embed_tokens(input_ids), positions, cache, start)
    return model.lm_head(hidden[:, -1:, :]).squeeze(1)


def run_teacher_forced(model: torch.nn.Module, prompt: np.ndarray, target: np.ndarray, mode: str,
                       workspace: torch.Tensor, ref_logp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    prompt_t = torch.as_tensor(prompt, device="cuda", dtype=torch.long).unsqueeze(0)
    target_t = torch.as_tensor(target, device="cuda", dtype=torch.long).unsqueeze(0)
    cache = StaticPagedKV(1, prompt.shape[0] + target.shape[0], mode, workspace)
    logits = forward_logits(model, prompt_t, cache, 0)
    logps: list[np.ndarray] = []
    nlls: list[float] = []
    kls: list[float] = []
    for t in range(target.shape[0]):
        logp = F.log_softmax(logits.float(), dim=-1)
        if ref_logp is not None:
            ref = torch.from_numpy(ref_logp[t]).to(device="cuda")
            kl = (ref.exp() * (ref - logp)).sum().item()
        else:
            kl = 0.0
        nll = -logp[0, int(target[t])].item()
        nlls.append(float(nll)); kls.append(float(kl))
        if ref_logp is None:
            logps.append(logp[0].detach().cpu().numpy().astype(np.float32))
        if t + 1 < target.shape[0]:
            logits = forward_logits(model, target_t[:, t:t + 1], cache, prompt.shape[0] + t)
    reference = np.asarray(logps, dtype=np.float32) if ref_logp is None else np.empty((0,))
    profile = cache.profile()
    del cache
    torch.cuda.empty_cache()
    return np.asarray(nlls, dtype=np.float64), np.asarray(kls, dtype=np.float64), {"profile": profile, "reference_logp": reference}


def generate_batch(model: torch.nn.Module, prompts: np.ndarray, mode: str, workspace: torch.Tensor,
                   output_len: int) -> dict:
    bsz, prompt_len = prompts.shape
    prompt_t = torch.as_tensor(prompts, device="cuda", dtype=torch.long)
    cache = StaticPagedKV(bsz, prompt_len + output_len, mode, workspace)
    start_time = time.perf_counter()
    logits = forward_logits(model, prompt_t, cache, 0)
    torch.cuda.synchronize()
    first_time = time.perf_counter()
    tokens = [torch.argmax(logits, dim=-1)]
    token_times = [first_time]
    current = tokens[0]
    for step in range(1, output_len):
        logits = forward_logits(model, current[:, None], cache, prompt_len + step - 1)
        torch.cuda.synchronize()
        token_times.append(time.perf_counter())
        current = torch.argmax(logits, dim=-1)
        tokens.append(current)
    end_time = time.perf_counter()
    output = torch.stack(tokens, dim=1).cpu().numpy().astype(np.int64)
    first_ms = (first_time - start_time) * 1000.0
    timestamps = np.asarray([(t - start_time) * 1000.0 for t in token_times], dtype=np.float64)
    intervals = np.diff(timestamps)
    tpot_ms = float(intervals.mean()) if len(intervals) else 0.0
    profile = cache.profile()
    profile.update({
        "status": "complete", "start_utc": now(), "wall_ms": (end_time - start_time) * 1000.0,
        "ttft_ms": first_ms, "tpot_ms": tpot_ms,
        "gpu_allocated_after_cache_bytes": int(torch.cuda.memory_allocated()),
        "gpu_reserved_after_cache_bytes": int(torch.cuda.memory_reserved()),
    })
    del cache
    torch.cuda.empty_cache()
    return {"output": output, "timestamps_ms": timestamps.tolist(), "profile": profile,
            "ttft_ms": first_ms, "tpot_ms": tpot_ms, "wall_ms": (end_time - start_time) * 1000.0}


def write_raw_row(out: Path, row: dict) -> None:
    path = out / "raw_metrics.csv"
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in RAW_FIELDS})
        handle.flush()


def bootstrap_ci(values: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_manifest(out: Path, manifest: dict) -> None:
    manifest["updated_utc"] = now()
    save_json(out / "run_manifest.json", manifest)


def capability_probe(manifest: dict) -> dict:
    result: dict = {"attempted_utc": now(), "vllm": {}, "flashinfer": {}, "four_bit": {}}
    try:
        import vllm
        result["vllm"] = {"imported": True, "version": getattr(vllm, "__version__", metadata_version("vllm"))}
    except Exception:
        result["vllm"] = {"imported": False, "error": traceback.format_exc()}
    try:
        import flashinfer
        result["flashinfer"] = {"imported": True, "version": getattr(flashinfer, "__version__", metadata_version("flashinfer")),
                                  "module": str(Path(flashinfer.__file__).resolve())}
    except Exception:
        result["flashinfer"] = {"imported": False, "error": traceback.format_exc()}
        return result
    try:
        import torch
        import flashinfer.decode as flashinfer_decode
        import flashinfer.prefill as flashinfer_prefill
        fp4_names = [name for name in dir(torch) if "float4" in name.lower() or "fp4" in name.lower()]
        flashinfer_fp4_names = [name for name in dir(flashinfer) if "float4" in name.lower() or "fp4" in name.lower()]
        paged_fp4_names = [name for name in set(dir(flashinfer_prefill)) | set(dir(flashinfer_decode))
                           if "paged" in name.lower() and "fp4" in name.lower()]
        result["four_bit"] = {"supported": False, "attempted": True,
                              "reason": "FlashInfer exposes FP4 utility/wrapper symbols but no direct paged FP4 prefill/decode attention entry point; torch has no native FP4 dtype. uint8 without a kernel is not counted as packed 4-bit.",
                              "torch_fp4_symbols": fp4_names,
                              "flashinfer_fp4_symbols": flashinfer_fp4_names,
                              "direct_paged_fp4_symbols": sorted(paged_fp4_names),
                              "checked_modules": ["flashinfer", "flashinfer.prefill", "flashinfer.decode"]}
    except Exception:
        result["four_bit"] = {"supported": False, "attempted": True, "error": traceback.format_exc()}
    manifest["capability_probe"] = result
    return result


def load_or_init_manifest(out: Path, cfg: dict, args: argparse.Namespace) -> dict:
    path = out / "run_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        require(manifest["config_sha256"] == hash_file(CONFIG), "Config changed after run initialization")
        require(manifest["protocol_sha256"] == hash_file(PROTOCOL), "Protocol changed after run initialization")
        return manifest
    source_hashes = {}
    for name in SOURCES:
        source = ROOT / name
        require(source.is_file(), f"Missing frozen source: {name}")
        source_hashes[name] = hash_file(source)
    history = json.loads(HISTORY.read_text())
    old_paths = [p for p in ROOT.joinpath("results").rglob("*") if p.is_file() and not p.is_relative_to(out)]
    history_hashes = {p.relative_to(ROOT).as_posix(): hash_file(p) for p in sorted(old_paths)}
    manifest = {
        "status": "initialized", "created_utc": now(), "config_sha256": hash_file(CONFIG),
        "protocol_sha256": hash_file(PROTOCOL), "source_sha256": source_hashes,
        "history_sha256": history_hashes, "history_commit": "1eef360",
        "history_checkpoint_sha256": history["checkpoint_sha256"],
        "history_model_revision": history["config"]["model_revision"],
        "input_sha256": {name: hash_file(out / name) for name in ("prompts.npz", "prompt_manifest.json")},
        "config": cfg, "command": shlex.join([sys.executable, *sys.argv]), "cwd": str(Path.cwd()),
        "host": platform.node(), "platform": platform.platform(), "python": sys.version,
        "versions": {name: metadata_version(name) for name in ("torch", "transformers", "numpy", "safetensors", "huggingface-hub", "flashinfer", "vllm")},
        "environment": {key: os.environ.get(key) for key in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "HF_HOME", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "gpu_inventory_before": subprocess.check_output(["nvidia-smi"], text=True),
        "phases_requested": args.phase, "completed": [], "errors": [], "ooms": [],
        "raw_metrics": "raw_metrics.csv", "raw_runs": "raw_runs.jsonl", "raw_cache_profiles": "raw_cache_profiles.jsonl",
    }
    save_json(path, manifest)
    return manifest


def run_probe(out: Path, manifest: dict, arrays: dict[str, np.ndarray]) -> None:
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "Exactly one usable BF16 CUDA GPU is required")
    free, total = torch.cuda.mem_get_info()
    require(free >= 12 * 1024**3, f"Need at least 12 GiB free before loading model; free={free}")
    model = prepare_model()
    manifest["model_runtime"] = {"config": model.config.to_dict(), "device": str(torch.cuda.get_device_properties(0)),
                                  "state_tensors": len(state_inventory(model)), "weights_dtype": "bfloat16"}
    workspace = torch.empty(384 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    result = capability_probe(manifest)
    require(result["flashinfer"].get("imported") is True, "FlashInfer is unavailable; no direct compressed KV kernel")
    requested = ["BF16", "FP8_E4M3", "FP8_E5M2"]
    for mode in requested:
        started = now()
        try:
            run = generate_batch(model, arrays["performance_prompt_2048"][:1], mode, workspace, 256)
            profile = run["profile"]
            require(profile["direct_kernel_read"] and profile["whole_cache_bf16_restore"] is False, f"Direct kernel provenance failed for {mode}")
            require(profile["cache_payload_bytes"] > 0, f"Empty cache payload for {mode}")
            append_jsonl(out / "raw_cache_profiles.jsonl", {"phase": "probe", "started_utc": started, **profile})
            append_jsonl(out / "raw_runs.jsonl", {"phase": "probe", "mode": mode, "prompt_length": 2048,
                                                  "concurrency": 1, "repeat": 0, "status": "complete",
                                                  "ttft_ms": run["ttft_ms"], "tpot_ms": run["tpot_ms"],
                                                  "wall_ms": run["wall_ms"], "timestamps_ms": run["timestamps_ms"]})
            manifest.setdefault("probe", []).append({"mode": mode, "status": "complete", "profile": profile,
                                                       "ttft_ms": run["ttft_ms"], "tpot_ms": run["tpot_ms"]})
        except BaseException:
            error = traceback.format_exc()
            manifest.setdefault("probe", []).append({"mode": mode, "status": "failed", "error": error})
            manifest["errors"].append({"phase": "probe", "mode": mode, "error": error})
            # E4M3 and E5M2 are independent local 8-bit capability probes.
            # A failure is retained; the run can continue only if at least one
            # compressed mode and BF16 both pass their smoke checks.
        write_manifest(out, manifest)
    passed_modes = {r["mode"] for r in manifest.get("probe", []) if r.get("status") == "complete"}
    require("BF16" in passed_modes and passed_modes.intersection({"FP8_E4M3", "FP8_E5M2"}),
            "No supported direct compressed KV kernel passed smoke")
    del workspace, model
    torch.cuda.empty_cache()
    manifest["probe_status"] = "passed"
    write_manifest(out, manifest)


def verify_bf16_alignment(out: Path, model: torch.nn.Module, arrays: dict[str, np.ndarray], workspace: torch.Tensor, manifest: dict) -> None:
    prompt = arrays["quality_prompt_2048"][0]
    target = arrays["quality_target_2048"][0]
    nll, _, extra = run_teacher_forced(model, prompt, target, "BF16", workspace)
    ref_logp = extra["reference_logp"]
    full = torch.as_tensor(np.concatenate([prompt, target[:-1]]), device="cuda", dtype=torch.long).unsqueeze(0)
    positions = torch.arange(prompt.shape[0] - 1, prompt.shape[0] - 1 + target.shape[0], device="cuda")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        no_cache = model(input_ids=full, use_cache=False, logits_to_keep=positions, return_dict=True).logits.squeeze(0)
    no_cache_logp = F.log_softmax(no_cache.float(), dim=-1).cpu().numpy()
    cached_nll = -ref_logp[np.arange(len(target)), target]
    reference_nll = -no_cache_logp[np.arange(len(target)), target]
    max_logit_error = float(np.max(np.abs(ref_logp - no_cache_logp)))
    mean_nll_abs = float(np.mean(np.abs(cached_nll - reference_nll)))
    result = {"status": "passed", "prompt_length": 2048, "prompt_id": 0,
              "max_abs_log_probability_error": max_logit_error, "mean_abs_nll_error": mean_nll_abs,
              "cached_nll_sum": float(nll.sum()), "no_cache_nll_sum": float(reference_nll.sum()),
              "criterion": "implementation sanity bound: mean absolute NLL <= 0.05 and max absolute log-probability <= 1.0; BF16 backend reduction differences are not a compressed-quality gate"}
    require(mean_nll_abs <= 0.05 and max_logit_error <= 1.0,
            f"BF16 cached decode does not align with no-cache baseline: {result}")
    save_json(out / "bf16_cached_decode_alignment.json", result)
    manifest["bf16_alignment"] = result
    write_manifest(out, manifest)


def run_quality(out: Path, manifest: dict, arrays: dict[str, np.ndarray]) -> None:
    model = prepare_model()
    workspace = torch.empty(384 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    verify_bf16_alignment(out, model, arrays, workspace, manifest)
    supported = [r["mode"] for r in manifest.get("probe", []) if r.get("status") == "complete"]
    modes = [m for m in supported if m in MODES]
    require("BF16" in modes and modes != ["BF16"], "Smoke did not establish BF16 and a direct compressed mode")
    for length in manifest["config"]["prompt_lengths"]:
        for prompt_id in range(manifest["config"]["quality_prompts_per_length"]):
            prompt = arrays[f"quality_prompt_{length}"][prompt_id]
            target = arrays[f"quality_target_{length}"][prompt_id]
            nll_ref, _, extra = run_teacher_forced(model, prompt, target, "BF16", workspace)
            ref_logp = extra["reference_logp"]
            for token_index, value in enumerate(nll_ref):
                write_raw_row(out, {"mode": "BF16", "phase": "quality", "prompt_length": length, "concurrency": 1,
                                    "repeat": 0, "request_id": prompt_id, "token_index": token_index, "status": "complete",
                                    "prompt_tokens": length, "output_tokens": 256, "nll": value, "baseline_nll": value,
                                    "delta_nll": 0.0, "kl": 0.0})
            for mode in modes:
                if mode == "BF16":
                    profile = extra["profile"]
                    append_jsonl(out / "raw_cache_profiles.jsonl", {"phase": "quality", "mode": mode, "status": "complete", "prompt_length": length,
                                                                      "prompt_id": prompt_id, **profile})
                    continue
                nll, kl, extra_mode = run_teacher_forced(model, prompt, target, mode, workspace, ref_logp)
                profile = extra_mode["profile"]
                append_jsonl(out / "raw_cache_profiles.jsonl", {"phase": "quality", "mode": mode, "status": "complete", "prompt_length": length,
                                                                  "prompt_id": prompt_id, **profile})
                for token_index, (value, ref_value, kl_value) in enumerate(zip(nll, nll_ref, kl)):
                    write_raw_row(out, {"mode": mode, "phase": "quality", "prompt_length": length, "concurrency": 1,
                                        "repeat": 0, "request_id": prompt_id, "token_index": token_index, "status": "complete",
                                        "prompt_tokens": length, "output_tokens": 256, "nll": value,
                                        "baseline_nll": ref_value, "delta_nll": value - ref_value, "kl": kl_value})
        manifest.setdefault("completed", []).append({"phase": "quality", "prompt_lengths": [length], "modes": modes})
        write_manifest(out, manifest)

    # Retrieval EM uses the exact same direct cache runner and one fixed greedy output per item.
    retrieval_rows = []
    for mode in modes:
        for retrieval_id, prompt in enumerate(arrays["retrieval_prompt_8192"]):
            run = generate_batch(model, prompt[None, :], mode, workspace, 256)
            output = run["output"][0].tolist()
            tokenizer_dir = out / "tokenizer"
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
            decoded = tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            answer = manifest["retrieval_answers"][retrieval_id] if "retrieval_answers" in manifest else json.loads((out / "prompt_manifest.json").read_text())["retrieval_answers"][retrieval_id]
            first_line = decoded.splitlines()[0].strip() if decoded.splitlines() else decoded.strip()
            em = int(first_line == answer)
            retrieval_rows.append({"mode": mode, "retrieval_id": retrieval_id, "status": "complete", "em": em,
                                   "answer": answer, "decoded": decoded, "output_token_ids": output,
                                   "ttft_ms": run["ttft_ms"], "tpot_ms": run["tpot_ms"], "cache_profile": run["profile"]})
            write_raw_row(out, {"mode": mode, "phase": "retrieval", "prompt_length": 8192, "concurrency": 1,
                                "repeat": 0, "request_id": retrieval_id, "token_index": "", "status": "complete",
                                "prompt_tokens": 8192, "output_tokens": 256, "em": em,
                                "ttft_ms": run["ttft_ms"], "tpot_ms": run["tpot_ms"], "wall_ms": run["wall_ms"]})
            append_jsonl(out / "raw_cache_profiles.jsonl", {"phase": "retrieval", "mode": mode, "retrieval_id": retrieval_id, **run["profile"]})
    save_json(out / "retrieval_raw.json", retrieval_rows)
    manifest["retrieval_status"] = "complete"
    write_manifest(out, manifest)
    del workspace, model
    torch.cuda.empty_cache()


def performance_one(out: Path, model: torch.nn.Module, workspace: torch.Tensor, arrays: dict[str, np.ndarray],
                    mode: str, length: int, concurrency: int, repeat: int, output_len: int) -> None:
    prompts = arrays[f"performance_prompt_{length}"][:concurrency]
    started = now()
    try:
        allocated_before = int(torch.cuda.memory_allocated())
        run = generate_batch(model, prompts, mode, workspace, output_len)
        profile = run["profile"]
        profile.update({"phase": "performance", "mode": mode, "prompt_length": length,
                        "concurrency": concurrency, "repeat": repeat, "started_utc": started,
                        "allocated_before_cache_bytes": allocated_before})
        append_jsonl(out / "raw_cache_profiles.jsonl", profile)
        append_jsonl(out / "raw_runs.jsonl", {"phase": "performance", "mode": mode, "prompt_length": length,
                                              "concurrency": concurrency, "repeat": repeat, "status": "complete",
                                              "ttft_ms": run["ttft_ms"], "tpot_ms": run["tpot_ms"], "wall_ms": run["wall_ms"],
                                              "timestamps_ms": run["timestamps_ms"], "output_token_ids": run["output"].tolist(),
                                              "cache_payload_bytes": profile["cache_payload_bytes"],
                                              "gpu_allocated_after_cache_bytes": profile["gpu_allocated_after_cache_bytes"],
                                              "gpu_reserved_after_cache_bytes": profile["gpu_reserved_after_cache_bytes"]})
        for request_id in range(concurrency):
            write_raw_row(out, {"mode": mode, "phase": "performance", "prompt_length": length,
                                "concurrency": concurrency, "repeat": repeat, "request_id": request_id,
                                "status": "complete", "prompt_tokens": length, "output_tokens": output_len,
                                "ttft_ms": run["ttft_ms"], "tpot_ms": run["tpot_ms"], "completion_ms": run["wall_ms"],
                                "wall_ms": run["wall_ms"]})
    except BaseException:
        error = traceback.format_exc()
        record = {"phase": "performance", "mode": mode, "prompt_length": length, "concurrency": concurrency,
                  "repeat": repeat, "status": "oom_or_error", "error": error, "started_utc": started}
        append_jsonl(out / "raw_runs.jsonl", record)
        write_raw_row(out, {"mode": mode, "phase": "performance", "prompt_length": length,
                            "concurrency": concurrency, "repeat": repeat, "status": "oom_or_error", "error": error})
        manifest = json.loads((out / "run_manifest.json").read_text())
        if "out of memory" in error.lower() or "cudaerror" in error.lower():
            manifest.setdefault("ooms", []).append(record)
        else:
            manifest.setdefault("errors", []).append(record)
        write_manifest(out, manifest)
        torch.cuda.empty_cache()


def performance_repeats(out: Path, mode: str, length: int, concurrency: int) -> set[int]:
    path = out / "raw_runs.jsonl"
    if not path.exists():
        return set()
    repeats = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("phase") == "performance" and row.get("mode") == mode and
                row.get("prompt_length") == length and row.get("concurrency") == concurrency):
            repeats.add(int(row["repeat"]))
    return repeats


def run_performance(out: Path, manifest: dict, arrays: dict[str, np.ndarray]) -> None:
    model = prepare_model()
    workspace = torch.empty(384 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    modes = [r["mode"] for r in manifest.get("probe", []) if r.get("status") == "complete"]
    cfg = manifest["config"]
    required_repeats = {-1, *range(cfg["repeats"])}
    for mode in modes:
        for length in cfg["prompt_lengths"]:
            for concurrency in cfg["performance_concurrency"]:
                # Resume safely after an external timeout: every fixed repeat is
                # recorded exactly once, while a completed/OOM cell is untouched.
                existing = performance_repeats(out, mode, length, concurrency)
                for repeat in sorted(required_repeats - existing):
                    performance_one(out, model, workspace, arrays, mode, length, concurrency,
                                    repeat, cfg["output_length"])
                if required_repeats.issubset(performance_repeats(out, mode, length, concurrency)):
                    # performance_one persists failures through a fresh manifest;
                    # refresh those fields before this process writes its older
                    # in-memory manifest, otherwise OOM/error provenance is lost.
                    persisted = json.loads((out / "run_manifest.json").read_text())
                    manifest["errors"] = persisted.get("errors", manifest.get("errors", []))
                    manifest["ooms"] = persisted.get("ooms", manifest.get("ooms", []))
                    manifest.setdefault("completed", []).append({"phase": "performance", "mode": mode,
                                                                  "prompt_length": length, "concurrency": concurrency,
                                                                  "warmups": cfg["warmups"], "repeats": cfg["repeats"]})
                    write_manifest(out, manifest)
    del workspace, model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("probe", "quality", "performance", "all"), default="all")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    cfg = json.loads((out / "experiment_config.json").read_text())
    prompt_manifest, arrays = load_inputs(out)
    require(prompt_manifest["config_sha256"] == hash_file(out / "experiment_config.json"), "Prompt/config mismatch")
    manifest = load_or_init_manifest(out, cfg, args)
    try:
        if args.phase in ("probe", "all") and manifest.get("probe_status") != "passed":
            manifest["status"] = "running_probe"; write_manifest(out, manifest)
            capability_probe(manifest); write_manifest(out, manifest)
            run_probe(out, manifest, arrays)
        if args.phase in ("quality", "all") and manifest.get("quality_status") != "complete":
            manifest["status"] = "running_quality"; write_manifest(out, manifest)
            run_quality(out, manifest, arrays); manifest["quality_status"] = "complete"; write_manifest(out, manifest)
        if args.phase in ("performance", "all") and manifest.get("performance_status") != "complete":
            manifest["status"] = "running_performance"; write_manifest(out, manifest)
            run_performance(out, manifest, arrays); manifest["performance_status"] = "complete"; write_manifest(out, manifest)
        manifest["status"] = "measured"
        manifest["finished_utc"] = now()
        manifest["gpu_inventory_after"] = subprocess.check_output(["nvidia-smi"], text=True)
        write_manifest(out, manifest)
        print(f"FINISHED phase={args.phase} status={manifest['status']}", flush=True)
    except BaseException:
        error = traceback.format_exc()
        manifest["status"] = "blocked"; manifest["last_error"] = error
        write_manifest(out, manifest)
        with (out / "blocked.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\n## {now()} blocked\n\nCommand: `{manifest['command']}`\n\n```text\n{error}\n```\n\n"
                         "保留已完成probe/quality/performance、raw metrics與OOM/error；不縮短固定prompt/output/concurrency，不用BF16解壓路徑冒充compressed結果。\n")
        raise


if __name__ == "__main__":
    main()
