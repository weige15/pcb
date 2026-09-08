

## 2026-09-08T10:05:40.612207+00:00 — 本輪阻塞，未完成

Command: `/nfs/home/s314511048/.venv/bin/python scripts/phase1a.py --config scripts/phase1a_config.json --output results/smoke_20260908 --smoke-only`

Completed conditions: 0/14. See phase1a_run.json for exact completed cells.

```text
Traceback (most recent call last):
  File "/nfs/home/s314511048/pcb/scripts/phase1a.py", line 477, in main
    model_path, indices, blocks = prepare_data(cfg, out, manifest)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfs/home/s314511048/pcb/scripts/phase1a.py", line 102, in prepare_data
    model_path = Path(snapshot_download(cfg["model_id"], revision=cfg["model_revision"], local_files_only=True))
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfs/home/s314511048/.venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/nfs/home/s314511048/.venv/lib/python3.12/site-packages/huggingface_hub/_snapshot_download.py", line 321, in snapshot_download
    _raise_if_incomplete_snapshot(
  File "/nfs/home/s314511048/.venv/lib/python3.12/site-packages/huggingface_hub/_snapshot_download.py", line 576, in _raise_if_incomplete_snapshot
    raise IncompleteSnapshotError(
huggingface_hub.errors.IncompleteSnapshotError: The cached snapshot for 'Qwen/Qwen3-4B' (revision '1cfa9a7208912126459214e8b04321603b3df60c', commit 1cfa9a7208912126459214e8b04321603b3df60c) is incomplete: 1 file(s) are missing (.gitattributes). Outgoing traffic is disabled ('local_files_only=True'). Re-run the download with network access to complete the snapshot.

```

未完成項：其餘條件、未通過的控制、完整 NLL/KL/CI 與完成稽核。保留已有數據；不自動重試、換模型、減少樣本或擴大範圍。若缺 GPU/checkpoint/data 或 OOM，請使用 `/goal pause`，提供可用資源後通知繼續。
