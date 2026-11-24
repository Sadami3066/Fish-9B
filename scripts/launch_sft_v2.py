#!/usr/bin/env python3
"""Launch SFT v2 on a mounted Colab runtime and monitor its exit status."""

import argparse
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path


COLAB = os.environ.get("FISH9B_COLAB_BIN", "colab")
DRIVE_ROOT = os.environ.get("FISH9B_DRIVE_ROOT", "/content/drive/MyDrive/Fish-9B")
TRAINER = Path(__file__).resolve().parents[1] / "colab" / "train_sft_v2.py"
ACCELERATORS = {
    "A100hm": ["--gpu", "A100", "--high-mem"],
    "A100": ["--gpu", "A100"],
    "H100": ["--gpu", "H100", "--high-mem"],
    "G4": ["--gpu", "G4"],
    "L4": ["--gpu", "L4"],
}


def remote_python(session: str, code: str, timeout: int = 50) -> str:
    result = subprocess.run(
        [COLAB, "exec", "-s", session, "--timeout", str(timeout - 5)],
        input=code,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout


def run_job(session: str, run_id: str, train_args: list[str], max_hours: float) -> None:
    remote_script = f"/content/train_sft_v2_{run_id}.py"
    subprocess.run([COLAB, "upload", "-s", session, str(TRAINER), remote_script], check=True)
    argv = ["python3", "-u", remote_script, *train_args]
    log_path = str(Path(DRIVE_ROOT) / "outputs" / f"sft_v2_{run_id}.log")
    done_path = f"/tmp/fish9b_sft_{run_id}.json"
    remote_job = f"""
import os, json, subprocess
from pathlib import Path
env = dict(os.environ, FISH9B_DRIVE_ROOT={DRIVE_ROOT!r})
argv = {argv!r}
with open({log_path!r}, 'x', encoding='utf-8') as log:
    result = subprocess.run(argv + ['--check_only'], env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
flag = Path({done_path!r})
temp = flag.with_suffix('.tmp')
temp.write_text(json.dumps({{'returncode': result.returncode}}))
temp.replace(flag)
"""
    start = f"""
from pathlib import Path
import subprocess
root = Path({DRIVE_ROOT!r})
if root.is_dir():
    (root / 'outputs').mkdir(exist_ok=True)
    subprocess.Popen(['python3', '-c', {remote_job!r}], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print('FISH9B_JOB_STARTED')
else:
    print('FISH9B_DRIVE_UNAVAILABLE')
"""
    if "FISH9B_JOB_STARTED" not in remote_python(session, start):
        raise RuntimeError(f"找不到 Colab 中的项目目录 {DRIVE_ROOT}。请检查 Drive 挂载。")

    print(f"训练及预检日志: {log_path}", flush=True)
    deadline = time.monotonic() + max_hours * 3600
    while time.monotonic() < deadline:
        status = remote_python(session, f"""
from pathlib import Path
flag = Path({done_path!r})
print('FISH9B_JOB_STATUS ' + (flag.read_text() if flag.is_file() else 'running'))
""")
        marker = next((line for line in status.splitlines() if line.startswith("FISH9B_JOB_STATUS ")), "")
        if marker and not marker.endswith("running"):
            details = json.loads(marker.split(" ", 1)[1])
            if details["returncode"]:
                raise RuntimeError(f"预检或训练失败（退出码 {details['returncode']}），见 {log_path}")
            return
        time.sleep(15)
    raise TimeoutError(f"训练超过 {max_hours:g} 小时，见 {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="xiaoyu")
    parser.add_argument("--cands", default="A100hm,A100,G4,L4", help="按尝试顺序列出 GPU")
    parser.add_argument("--wait_min", type=float, default=25)
    parser.add_argument("--max_hours", type=float, default=2.5)
    parser.add_argument("--train_args", default="", help="转给 train_sft_v2.py 的参数")
    parser.add_argument("--unsloth", action="store_true", help="安装并使用 Unsloth")
    parser.add_argument("--reuse", action="store_true", help="使用已挂载 Drive 的现有会话")
    parser.add_argument("--no_autostop", action="store_true", help="保留由本命令创建的会话")
    args = parser.parse_args()
    candidates = args.cands.split(",")
    if set(candidates) - ACCELERATORS.keys():
        parser.error("--cands 包含未知 GPU")
    if args.wait_min <= 0 or args.max_hours <= 0:
        parser.error("--wait_min 和 --max_hours 必须大于 0")
    train_args = shlex.split(args.train_args)
    if args.unsloth:
        train_args = ["--backend", "unsloth", *train_args]
    run_id = uuid.uuid4().hex[:12]
    created = False

    try:
        if not args.reuse:
            deadline = time.monotonic() + args.wait_min * 60
            while not created and time.monotonic() < deadline:
                for name in candidates:
                    result = subprocess.run(
                        [COLAB, "new", "-s", args.session, *ACCELERATORS[name]], check=False
                    )
                    if result.returncode == 0:
                        created = True
                        break
                if not created:
                    time.sleep(min(30, max(0, deadline - time.monotonic())))
            if not created:
                raise RuntimeError("等待期间没有申请到可用 GPU")
            subprocess.run([COLAB, "drivemount", "-s", args.session], check=True)

        packages = ["transformers>=5,<6", "peft", "accelerate", "tensorboard"]
        if args.unsloth:
            packages.append("unsloth")
        subprocess.run([COLAB, "install", "-s", args.session, *packages], check=True)
        run_job(args.session, run_id, train_args, args.max_hours)
        print("训练完成。", flush=True)
    finally:
        if created and not args.no_autostop:
            stopped = subprocess.run([COLAB, "stop", "-s", args.session], check=False)
            if stopped.returncode:
                print(f"Colab 停机失败，请检查会话 {args.session}。", flush=True)


if __name__ == "__main__":
    main()
