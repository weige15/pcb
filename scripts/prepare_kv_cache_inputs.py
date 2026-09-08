#!/usr/bin/env python3
"""Prepare immutable token-id prompts for the static KV-cache screen (no model forward)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("results/kv_cache_precision_pareto/experiment_config.json"))
    parser.add_argument("--output", type=Path, default=Path("results/kv_cache_precision_pareto"))
    parser.add_argument("--history", type=Path, default=Path("results"))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    history = json.loads((args.history / "phase1a_run.json").read_text())

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq

    model_path = Path(snapshot_download(cfg["model_id"], revision=cfg["tokenizer_revision"], local_files_only=True,
                                        allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]))
    data_path = Path(snapshot_download(cfg["dataset_id"], repo_type="dataset", revision=cfg["dataset_revision"],
                                       local_files_only=True, allow_patterns=[cfg["dataset_config"] + "/test-00000-of-00001.parquet"]))
    parquet = data_path / cfg["dataset_config"] / "test-00000-of-00001.parquet"
    table = pq.read_table(parquet, columns=["text"])
    text = "\n\n".join(table.column("text").to_pylist())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    token_ids = np.asarray(tokenizer(text, **cfg["tokenizer_call"])["input_ids"], dtype=np.int64)
    span_hash = sha256_array(token_ids)
    expected_stream_hash = history["data"]["token_stream_sha256_le_int64"]
    if span_hash != expected_stream_hash:
        raise RuntimeError(f"Pinned token stream differs from historical stream: {span_hash} != {expected_stream_hash}")

    lengths = cfg["prompt_lengths"]
    output_len = cfg["output_length"]
    n_perf = cfg["performance_prompts_per_length"]
    n_quality = cfg["quality_prompts_per_length"]
    arrays: dict[str, np.ndarray] = {}
    selections: dict[str, list[dict[str, int]]] = {}
    for length in lengths:
        span = length + output_len
        starts = [i * span for i in range(n_perf)]
        if starts[-1] + span > len(token_ids):
            raise RuntimeError("Pinned dataset cannot provide the fixed 32 non-overlapping prompt spans")
        prompts = np.stack([token_ids[s:s + length] for s in starts])
        targets = np.stack([token_ids[s + length:s + span] for s in starts])
        arrays[f"performance_prompt_{length}"] = prompts
        arrays[f"performance_target_{length}"] = targets
        arrays[f"quality_prompt_{length}"] = prompts[:n_quality].copy()
        arrays[f"quality_target_{length}"] = targets[:n_quality].copy()
        selections[str(length)] = [{"prompt_id": i, "start_token": s, "target_start_token": s + length,
                                    "prompt_tokens": length, "target_tokens": output_len} for i, s in enumerate(starts)]

    retrieval_prompts = []
    retrieval_answers = []
    retrieval_meta = []
    retrieval_length = 8192
    # Retrieval prompts are token-id exact and use only the pinned WikiText token stream as filler.
    # The needle is inserted halfway through; the final query is always at the prompt tail.
    for i in range(cfg["retrieval_prompts"]):
        answer = f"cobalt-{17 + i}"
        needle = f"\nThe retrieval answer for item R{i:02d} is {answer}.\n"
        query = f"\nQuestion: What is the retrieval answer for item R{i:02d}? Answer:"
        needle_ids = tokenizer(needle, **cfg["tokenizer_call"])["input_ids"]
        query_ids = tokenizer(query, **cfg["tokenizer_call"])["input_ids"]
        filler_n = retrieval_length - len(needle_ids) - len(query_ids)
        if filler_n <= 0:
            raise RuntimeError("Retrieval template is longer than the fixed context")
        filler_start = n_perf * (2048 + output_len) + i * filler_n
        filler = token_ids[filler_start:filler_start + filler_n].tolist()
        if len(filler) != filler_n:
            raise RuntimeError("Pinned dataset lacks fixed retrieval filler")
        split = filler_n // 2
        prompt = filler[:split] + needle_ids + filler[split:] + query_ids
        if len(prompt) != retrieval_length:
            raise RuntimeError(f"Retrieval prompt length mismatch: {len(prompt)}")
        retrieval_prompts.append(prompt)
        retrieval_answers.append(answer)
        retrieval_meta.append({"retrieval_id": i, "needle": needle, "query": query,
                               "needle_token_count": len(needle_ids), "query_token_count": len(query_ids),
                               "needle_start": split, "answer": answer, "filler_start": filler_start,
                               "prompt_tokens": retrieval_length})
    arrays["retrieval_prompt_8192"] = np.asarray(retrieval_prompts, dtype=np.int64)

    np.savez_compressed(out / "prompts.npz", **arrays)
    tokenizer_out = out / "tokenizer"
    tokenizer.save_pretrained(tokenizer_out)
    tokenizer_hashes = {p.name: sha256_bytes(p.read_bytes()) for p in sorted(tokenizer_out.iterdir()) if p.is_file()}
    array_hashes = {name: sha256_array(value) for name, value in arrays.items()}
    manifest = {
        "status": "prepared_no_model_forward",
        "config_sha256": sha256_bytes(args.config.read_bytes()),
        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"],
        "tokenizer_revision": cfg["tokenizer_revision"],
        "dataset_id": cfg["dataset_id"], "dataset_revision": cfg["dataset_revision"],
        "dataset_parquet": str(parquet), "dataset_parquet_sha256": sha256_bytes(parquet.read_bytes()),
        "dataset_rows": table.num_rows, "joined_text_sha256": sha256_bytes(text.encode()),
        "token_stream_sha256_le_int64": span_hash, "token_count": len(token_ids),
        "tokenizer_call": cfg["tokenizer_call"], "tokenizer_assets_sha256": tokenizer_hashes,
        "input_seed": cfg["input_seed"], "prompt_lengths": lengths, "output_length": output_len,
        "performance_prompts_per_length": n_perf, "quality_prompts_per_length": n_quality,
        "selections": selections, "retrieval": retrieval_meta, "retrieval_answers": retrieval_answers,
        "array_sha256": array_hashes, "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "array_dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        "selection_rule": "performance spans start at i*(prompt_length+output_length), quality is first eight; retrieval uses deterministic filler and needle template",
    }
    (out / "prompt_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": manifest["status"], "arrays": manifest["array_shapes"],
                      "token_stream_sha256": span_hash}, indent=2))


if __name__ == "__main__":
    main()
