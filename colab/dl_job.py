#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
import json
import os
import random
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

drive_root = Path(os.environ.get("FISH9B_DRIVE_ROOT", "/content/drive/MyDrive/Fish-9B"))
lccc_dir = drive_root / "data" / "lccc"
lccc_dir.mkdir(parents=True, exist_ok=True)
qwen_cache = drive_root / "model_cache" / "Qwen3.5-9B"
qwen_cache.mkdir(parents=True, exist_ok=True)

print(">>> [1/3] Downloading silver/lccc files to Google Drive /Fish-9B/data/lccc/ ...")
for fname in ["README.md", "lccc.py", "lccc_base_valid.jsonl.gz", "lccc_base_test.jsonl.gz", "lccc_base_train.jsonl.gz"]:
    dst = lccc_dir / fname
    if dst.exists() and dst.stat().st_size > 100:
        print(f"Already exists: {fname} ({dst.stat().st_size} bytes)")
        continue
    print(f"Fetching {fname}...")
    tmp_p = hf_hub_download(repo_id="silver/lccc", filename=fname, repo_type="dataset", local_dir="/content/tmp_lccc")
    shutil.copy2(tmp_p, dst)
    os.sync()
    os.remove(tmp_p)
    print(f"Saved {fname} -> {dst} ({dst.stat().st_size / (1024*1024):.1f} MB)")

print(">>> [2/3] Extracting 1000 rich multi-turn conversation seeds from LCCC -> lccc_curated_1000_seeds.jsonl ...")
curated = []
valid_gz = lccc_dir / "lccc_base_valid.jsonl.gz"
if valid_gz.exists():
    with gzip.open(valid_gz, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            dialog = json.loads(line)
            if isinstance(dialog, dict):
                dialog = dialog.get("conversation", dialog.get("dialog", []))
            if not isinstance(dialog, list):
                continue
            turns = [t.strip().replace(" ", "") for t in dialog if isinstance(t, str) and t.strip()]
            if 3 <= len(turns) <= 8 and len(turns[0]) >= 10 and sum(len(t) for t in turns) >= 45:
                curated.append({"turns": turns, "num_turns": len(turns), "first_turn": turns[0]})
            if len(curated) >= 4000:
                break

random.seed(42)
random.shuffle(curated)
selected_1000 = curated[:1000]
curated_path = lccc_dir / "lccc_curated_1000_seeds.jsonl"
with open(curated_path, "w", encoding="utf-8") as out_f:
    for idx, item in enumerate(selected_1000, 1):
        item["lccc_id"] = f"lccc_{idx:04d}"
        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
os.sync()
print(f"Saved {len(selected_1000)} curated multi-turn seeds -> {curated_path}")

print(">>> [3/3] Topping up missing Qwen/Qwen3.5-9B shards in /Fish-9B/model_cache/Qwen3.5-9B ...")
for shard in [
    "model.safetensors-00001-of-00004.safetensors",
    "model.safetensors-00002-of-00004.safetensors",
    "model.safetensors-00003-of-00004.safetensors",
    "model.safetensors-00004-of-00004.safetensors",
]:
    dst_s = qwen_cache / shard
    if dst_s.exists() and dst_s.stat().st_size > 100_000_000:
        print(f"Shard already cached: {shard} ({dst_s.stat().st_size / (1024**3):.2f} GB)")
        continue
    print(f"Downloading missing shard {shard} on CPU VM...")
    tmp_s = hf_hub_download(repo_id="Qwen/Qwen3.5-9B", filename=shard, local_dir="/content/tmp_qwen")
    shutil.copy2(tmp_s, dst_s)
    os.sync()
    os.remove(tmp_s)
    print(f"Cached shard {shard} -> {dst_s} ({dst_s.stat().st_size / (1024**3):.2f} GB)")

Path("/content/DL_ALL_DONE").touch()
print(">>> ALL DOWNLOADS AND SYNC COMPLETED!")
