#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fish-9B Google Drive 同步桥接工具，兼容 AGY gdrive CLI。
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DRIVE_IDS_PATH = Path(os.environ.get("FISH9B_DRIVE_CONFIG", BASE_DIR / "configs" / "drive_ids.json"))


def load_drive_ids() -> Dict[str, Any]:
    if not DRIVE_IDS_PATH.is_file():
        raise RuntimeError("Drive 同步未配置，请参照 configs/drive_ids.example.json 创建本地配置。")
    return json.loads(DRIVE_IDS_PATH.read_text(encoding="utf-8"))


def run_gdrive(args: List[str], account: Optional[str] = None) -> str:
    env = None
    if account:
        env = dict(os.environ)
        env["AGY_ACCOUNT"] = account
    cmd = [os.environ.get("FISH9B_GDRIVE_BIN", "gdrive"), *args]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except FileNotFoundError as exc:
        raise RuntimeError("找不到兼容的 Drive CLI，请配置 FISH9B_GDRIVE_BIN。") from exc
    if res.returncode != 0:
        raise RuntimeError(f"gdrive 命令失败 ({' '.join(cmd)}): {res.stderr or res.stdout}")
    return res.stdout.strip()


def find_file_in_folder(folder_id: str, filename: str) -> Optional[str]:
    out = run_gdrive(["readonly", "ls", folder_id, "--json"])
    if not out:
        return None
    items = json.loads(out)
    if isinstance(items, dict):
        items = items.get("files", [])
    for item in items:
        if item.get("name") == filename:
            return item.get("id")
    return None


def upsert_file(local_path: Path, folder_id: str, mime_type: Optional[str] = None) -> str:
    if not local_path.exists():
        raise FileNotFoundError(f"本地文件不存在: {local_path}")
    existing_id = find_file_in_folder(folder_id, local_path.name)
    if existing_id:
        cmd = ["mutate", "update", existing_id, str(local_path)]
        if mime_type:
            cmd.extend(["--mime-type", mime_type])
        out = run_gdrive(cmd)
        return f"Updated {local_path.name} (ID: {existing_id}): {out}"
    else:
        cmd = ["mutate", "upload", str(local_path), "--parent", folder_id]
        if mime_type:
            cmd.extend(["--mime-type", mime_type])
        out = run_gdrive(cmd)
        return f"Uploaded {local_path.name} -> folder {folder_id}: {out}"


def sync_data_to_drive() -> str:
    ids = load_drive_ids()
    data_folder_id = ids["data_folder_id"]
    logs = []

    master_file = BASE_DIR / "data" / "workbench" / "dataset_master.jsonl"
    if master_file.exists():
        logs.append(upsert_file(master_file, data_folder_id))

    exports_dir = BASE_DIR / "data" / "exports"
    if exports_dir.exists():
        for fn in ["train_sft.jsonl", "val_sft.jsonl", "dpo_pairs.jsonl", "eval_sealed.jsonl"]:
            p = exports_dir / fn
            if p.exists():
                logs.append(upsert_file(p, data_folder_id))

    prompts_file = BASE_DIR / "configs" / "prompts.yaml"
    if prompts_file.exists():
        logs.append(upsert_file(prompts_file, data_folder_id))

    return "\n".join(logs)


def status_all_accounts() -> str:
    try:
        ids = load_drive_ids()
    except RuntimeError as exc:
        return str(exc)
    accounts = [ids["owner_account"]] + ids.get("shared_accounts", [])
    lines = ["### Google Drive 连接状态"]
    for acc in accounts:
        try:
            q = run_gdrive(["readonly", "quota"], account=acc).replace("\n", " | ")
            lines.append(f"- **`{acc}`**: ✅ 已连接 ({q})")
        except Exception as e:
            lines.append(f"- **`{acc}`**: ❌ 异常 ({e})")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大肥鱼-9B Google Drive 同步桥接工具")
    parser.add_argument("action", choices=["status", "push-data"], help="执行动作")
    args = parser.parse_args()

    if args.action == "status":
        print(status_all_accounts())
    elif args.action == "push-data":
        print(sync_data_to_drive())
