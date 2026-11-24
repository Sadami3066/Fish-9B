#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大肥鱼-9B (Base) 本地省流 Colab CLI 编排器 (scripts/colab_orchestrator.py)
核心设计原则：
1. 严控本地 VPN 流量：只上传/下载几百 KB 的 JSONL 文本与压缩后的小图；
   18GB 底座模型、LoRA Checkpoints 和 GGUF 全部留在 Google Colab <-> Google Drive 机房内网。
2. 铁律防忘关机：所有开启的 Colab GPU 会话均置于 try ... finally 块中，
   一旦脚本执行完毕或异常中断，立刻调用 `colab stop -s <session>` 销毁实例。
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import List

COLAB_BIN = os.environ.get("FISH9B_COLAB_BIN", "colab")
BASE_DIR = Path(__file__).resolve().parent.parent
RUNNER_SCRIPT = BASE_DIR / "colab" / "colab_runner.py"
SESSION_NAME = "fish9b"


def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"[CMD] {' '.join(cmd)}")
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def get_usage() -> str:
    res = run_cmd([COLAB_BIN, "usage"], check=False)
    return (res.stdout or "") + (res.stderr or "")


def get_sessions() -> str:
    res = run_cmd([COLAB_BIN, "sessions"], check=False)
    return (res.stdout or "") + (res.stderr or "")


def stop_session(session_name: str = SESSION_NAME) -> str:
    res = run_cmd([COLAB_BIN, "stop", "-s", session_name], check=False)
    return (res.stdout or "") + (res.stderr or "")


def run_remote_job(
    task: str,
    gpu: str = "A100",
    lr: float = 5e-5,
    epochs: float = 1.0,
) -> str:
    """
    一键生命周期：
    1. 先将本地最新 JSONL 同步到 Google Drive (仅几十~几百 KB)
    2. 申请 Colab GPU (`colab new -s fish9b --gpu A100`)
    3. 挂载 Google Drive (`colab drivemount -s fish9b`)
    4. 按需在远端安装 ms-swift / peft / qwen-vl-utils 并运行 colab_runner.py
    5. 若是 fill_qwen 或 preflight，仅把生成的 JSONL/JSON 小文件拉回本地
    6. finally: 强制执行 `colab stop -s fish9b` 销毁释放 GPU！
    """
    logs = []
    # Step 1: 同步本地轻量 JSONL 到 Google Drive
    from gdrive_bridge import sync_data_to_drive, load_drive_ids, find_file_in_folder, run_gdrive
    logs.append("=== Step 1: 同步本地轻量 JSONL 至 Google Drive ===")
    logs.append(sync_data_to_drive())

    started = False
    try:
        logs.append(f"\n=== Step 2: 申请 Colab {gpu} 实例 ({SESSION_NAME}) ===")
        new_cmd = [COLAB_BIN, "new", "-s", SESSION_NAME, "--gpu", gpu]
        if gpu.upper() in ("A100", "H100"):
            new_cmd.append("--high-mem")
        res_new = run_cmd(new_cmd, check=True)
        started = True
        logs.append(res_new.stdout)

        logs.append("\n=== Step 3: 挂载 Google Drive (/content/drive) ===")
        subprocess.run([COLAB_BIN, "drivemount", "-s", SESSION_NAME], check=True)

        logs.append("\n=== Step 4: 上传并执行远端任务脚本 ===")
        run_cmd([COLAB_BIN, "upload", "-s", SESSION_NAME, str(RUNNER_SCRIPT), "/content/colab_runner.py"], check=True)

        if task in ("train_run1", "train_run1b"):
            logs.append("安装 ms-swift / peft / qwen_vl_utils (走 Google 机房内网)...")
            run_cmd([COLAB_BIN, "install", "-s", SESSION_NAME, "ms-swift", "peft", "qwen-vl-utils", "accelerate", "bitsandbytes"], check=True)
        else:
            run_cmd([COLAB_BIN, "install", "-s", SESSION_NAME, "peft", "qwen-vl-utils", "accelerate", "huggingface_hub"], check=True)

        remote_py = (
            f"import subprocess\n"
            f"subprocess.run(['python3', '/content/colab_runner.py', '--task', '{task}', '--lr', '{lr}', '--epochs', '{epochs}'], check=True)\n"
        )
        res_exec = subprocess.run(
            [COLAB_BIN, "exec", "-s", SESSION_NAME],
            input=remote_py,
            text=True,
            capture_output=True,
            check=True,
        )
        logs.append(res_exec.stdout)

        # Step 5: 仅拉回轻量的 JSONL / JSON 结果文件（绝不拉大模型）
        ids = load_drive_ids()
        out_folder_id = ids["outputs_folder_id"]
        if task == "fill_qwen":
            fid = find_file_in_folder(out_folder_id, "dataset_master_filled.jsonl")
            if fid:
                local_master = BASE_DIR / "data" / "workbench" / "dataset_master.jsonl"
                run_gdrive(["readonly", "download", fid, "--out", str(local_master)])
                logs.append(f"已将补齐的 Qwen 原答拉回本地工作台: {local_master}")
        elif task == "preflight":
            fid = find_file_in_folder(out_folder_id, "preflight_report.json")
            if fid:
                local_rep = BASE_DIR / "data" / "exports" / "preflight_report.json"
                run_gdrive(["readonly", "download", fid, "--out", str(local_rep)])
                logs.append(f"已拉回训前核验报告: {local_rep}")

    finally:
        if started:
            print(stop_session(SESSION_NAME), flush=True)

    return "\n".join(logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大肥鱼-9B 本地省流 Colab CLI 编排器")
    parser.add_argument("action", choices=["usage", "sessions", "stop", "preflight", "fill_qwen", "train_run1", "train_run1b"])
    parser.add_argument("--gpu", type=str, default="A100", help="GPU 类型 (T4 / L4 / A100)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()

    if args.action == "usage":
        print(get_usage())
    elif args.action == "sessions":
        print(get_sessions())
    elif args.action == "stop":
        print(stop_session())
    else:
        print(run_remote_job(task=args.action, gpu=args.gpu, lr=args.lr, epochs=args.epochs))
