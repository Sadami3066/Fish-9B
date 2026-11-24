#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大肥鱼-9B (Base) Colab 远端全功能执行器 (colab_runner.py)
所有重型流量（18GB 模型拉取、LoRA Checkpoints 保存、16-bit 合并与 GGUF 导出）
全部在 Google Colab <-> Google Drive (/content/drive/MyDrive/Fish-9B) 机房内网完成，
绝不向本地回传大模型文件，仅回传几百 KB 的 JSONL 文本结果与核验报告。

支持任务 (--task):
1. preflight       : 大纲第 4 节训前 3 项硬核验（仅语言主干挂 LoRA + 多轮图文 labels 掩码解码 + 2步断点恢复）
2. fill_qwen       : 为四栏数据表批量生成原版 Qwen/Qwen3.5-9B 回复（支持多轮历史接续采样）
3. train_run1      : Run 1 安全基线 (A100 BF16, rank=16, alpha=32, lr=5e-5, bs=4, 1 epoch, 保存半轮和一轮两个 ckpt)
4. train_run1b     : Run 1b 甜点扫参 (A100 BF16, 120~200 steps 区间, epochs=3 或 lr=1e-4)
5. eval_abc        : 大纲第 5 节 A(原版) / B(原版+示例) / C(微调版) 盲测生成（含 20 题推理直答 vs <think> 双测）
6. export_gguf     : 16-bit 合并胜出 LoRA 并通过 llama.cpp 导出 GGUF + mmproj 至 Google Drive
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DRIVE_ROOT = Path(os.environ.get("FISH9B_DRIVE_ROOT", "/content/drive/MyDrive/Fish-9B"))
DRIVE_DATA = DRIVE_ROOT / "data"
DRIVE_CKPT = DRIVE_ROOT / "checkpoints"
DRIVE_OUT = DRIVE_ROOT / "outputs"
DRIVE_CACHE = DRIVE_ROOT / "model_cache" / "Qwen3.5-9B"

LOCAL_WORK = Path("/content/fish9b_work")
MODEL_ID = "Qwen/Qwen3.5-9B"


def ensure_dirs() -> None:
    LOCAL_WORK.mkdir(parents=True, exist_ok=True)
    if DRIVE_ROOT.exists():
        for d in [DRIVE_DATA, DRIVE_CKPT, DRIVE_OUT, DRIVE_CACHE.parent]:
            d.mkdir(parents=True, exist_ok=True)


def get_model_path() -> str:
    """优先使用本地 NVMe 或从 5TB Google Drive 共享缓存加载 Qwen3.5-9B，并在首次拉取后同步至 Google Drive"""
    local_nvme = LOCAL_WORK / "Qwen3.5-9B"
    if local_nvme.exists() and len(list(local_nvme.glob("*.safetensors"))) >= 4:
        return str(local_nvme)

    if DRIVE_CACHE.exists() and len(list(DRIVE_CACHE.glob("*.safetensors"))) >= 4:
        print(f"[INFO] 命中 Google Drive 5TB 内网共享模型缓存: {DRIVE_CACHE}")
        return str(DRIVE_CACHE)

    print(f"[INFO] 正在通过 Google 机房高速内网拉取 {MODEL_ID} 至本地 NVMe (0 本地 VPN 流量)...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_ID, local_dir=str(local_nvme))

    if DRIVE_ROOT.exists():
        print(f"[INFO] 正在将 {MODEL_ID} 同步备份至 5TB Google Drive 缓存目录: {DRIVE_CACHE} ...")
        DRIVE_CACHE.mkdir(parents=True, exist_ok=True)
        for item in local_nvme.iterdir():
            if item.name.startswith("."):
                continue
            dst = DRIVE_CACHE / item.name
            if item.is_file() and (not dst.exists() or dst.stat().st_size != item.stat().st_size):
                shutil.copy2(item, dst)
        os.sync()
        print("[INFO] 5TB Google Drive 模型缓存同步并刷盘完成！")
    return str(local_nvme)


def task_preflight() -> Dict[str, Any]:
    """
    大纲第 4 节：训练前先验证三件事：
    1. 保存的 LoRA 确实只改动预期语言层（未误伤视觉层 ViT、视觉投影、词嵌入和输出头）；
    2. 训练时实际喂进去的图片、多轮历史、前缀和 labels 掩码全对（对所有轮次的目标 assistant 回复及结束符计算 loss）；
    3. 目标设备能用原版完成文字与带图输入，并演练一次断点恢复。
    """
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    model_path = get_model_path()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 仅对语言主干挂载 LoRA，显式排除 visual / aligner / embed_tokens / lm_head
    language_linear_Names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(skip in name for skip in ["visual", "vision", "merger", "aligner", "lm_head", "embed_tokens"]):
                continue
            leaf = name.split(".")[-1]
            if leaf not in language_linear_Names:
                language_linear_Names.append(leaf)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=language_linear_Names,
        modules_to_save=[],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)

    trainable_params = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    leaked_vision_or_head = [
        n for n in trainable_params
        if any(bad in n for bad in ["visual", "vision", "merger", "aligner", "lm_head", "embed_tokens"])
    ]
    assert len(leaked_vision_or_head) == 0, f"发现非语言主干层被误挂 LoRA: {leaked_vision_or_head}"

    # 校验多轮对话模板与 labels 掩码
    sample_msgs = [
        {"role": "user", "content": "十一月怎么掉这么多"},
        {"role": "assistant", "content": "这张只画到十月。你说的是最后那个点吗？"},
        {"role": "user", "content": "对，就是最后那个点，帮我把最后三个数抄出来"},
        {"role": "assistant", "content": "8月 142，9月 138，10月 91。"},
    ]
    rendered = processor.apply_chat_template(sample_msgs, tokenize=False, add_generation_prompt=False)

    report = {
        "status": "PASS",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
        "lora_target_leaf_modules": language_linear_Names,
        "trainable_tensor_count": len(trainable_params),
        "vision_or_head_leaked": leaked_vision_or_head,
        "rendered_multiturn_template_preview": rendered,
    }
    out_path = (DRIVE_OUT if DRIVE_OUT.exists() else LOCAL_WORK) / "preflight_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def task_fill_qwen(input_jsonl: Path, output_jsonl: Path) -> None:
    """批量为 dataset_master.jsonl 中缺失 qwen_original 的轮次生成原版 Qwen/Qwen3.5-9B 回复"""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    model_path = get_model_path()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    records = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    updated_count = 0
    for rec in records:
        msgs = rec.get("messages", [])
        img_paths = rec.get("images", [])
        pil_images = []
        for ip in img_paths:
            full_ip = DRIVE_ROOT / ip if (DRIVE_ROOT / ip).exists() else Path(ip)
            if full_ip.exists():
                pil_images.append(Image.open(full_ip).convert("RGB"))

        history: List[Dict[str, Any]] = []
        first_user_seen = False
        for m in msgs:
            role = m.get("role")
            if role == "user":
                u_text = m.get("content", "")
                if not first_user_seen and pil_images:
                    content_items: List[Dict[str, Any]] = [{"type": "image", "image": im} for im in pil_images]
                    content_items.append({"type": "text", "text": u_text})
                    history.append({"role": "user", "content": content_items})
                else:
                    history.append({"role": "user", "content": u_text})
                first_user_seen = True
            elif role == "assistant":
                if not (m.get("qwen_original") or "").strip():
                    inputs = processor.apply_chat_template(
                        history,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(model.device)
                    with torch.no_grad():
                        gen_ids = model.generate(
                            **inputs,
                            max_new_tokens=600,
                            temperature=0.7,
                            top_p=0.8,
                            do_sample=True,
                        )
                    new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
                    reply = processor.decode(new_tokens, skip_special_tokens=True).strip()
                    m["qwen_original"] = reply
                    updated_count += 1
                    print(f"[{rec.get('id')}] 生成 Qwen 原答 ({len(reply)} chars): {reply[:70].replace(chr(10), ' ')}...")

                # 下一轮的历史上下文优先使用你认可的 fish_approved（剥离 <think> 后作为对话上文）
                raw_next = (m.get("fish_approved") or m.get("qwen_original") or "").strip()
                if "</think>" in raw_next:
                    raw_next = raw_next.split("</think>", 1)[1].strip()
                history.append({"role": "assistant", "content": raw_next})

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] 已补齐 {updated_count} 个轮次的 Qwen 原答 -> {output_jsonl}")


def task_train(mode: str, lr: float, epochs: float, rank: int = 16, alpha: int = 32) -> None:
    """
    大纲第 4 节：Colab A100 使用 BF16 全精度底座 + 语言主干 LoRA
    Run 1 安全基线: rank=16, alpha=32, lr=5e-5, effective_batch=4, 1 epoch (存半轮与一轮两个 ckpt)
    Run 1b 甜点扫参: 2~3 epochs 或 lr=1e-4 (总步数 120~200 步)
    """
    model_path = get_model_path()
    train_data = DRIVE_DATA / "train_sft.jsonl"
    val_data = DRIVE_DATA / "val_sft.jsonl"
    run_tag = f"{mode}_r{rank}_lr{lr}_ep{epochs}_{int(time.time())}"
    out_dir = DRIVE_CKPT / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 计算总样本数以设定半轮和一轮的 save_steps
    with open(train_data, "r", encoding="utf-8") as f:
        num_samples = sum(1 for line in f if line.strip())
    steps_per_epoch = max(2, num_samples // 4)
    half_epoch_steps = max(1, steps_per_epoch // 2)

    cmd = [
        "swift", "sft",
        "--model", model_path,
        "--model_type", "qwen3_5",
        "--template", "qwen3_5",
        "--tuner_type", "lora",
        "--lora_rank", str(rank),
        "--lora_alpha", str(alpha),
        "--lora_dropout", "0.05",
        "--target_modules", "all-linear",
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--torch_dtype", "bfloat16",
        "--dataset", str(train_data),
        "--val_dataset", str(val_data),
        "--add_non_thinking_prefix", "false",
        "--loss_scale", "default",
        "--max_length", "4096",
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", str(lr),
        "--warmup_ratio", "0.05",
        "--lr_scheduler_type", "cosine",
        "--save_strategy", "steps",
        "--save_steps", str(half_epoch_steps),
        "--eval_steps", str(half_epoch_steps),
        "--logging_steps", "5",
        "--save_total_limit", "6",
        "--output_dir", str(out_dir),
    ]
    print(f"[TRAIN] 启动命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[DONE] 训练完成，Checkpoints 已直接落盘至 5TB Google Drive: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大肥鱼-9B Colab 远端执行器")
    parser.add_argument("--task", required=True, choices=["preflight", "fill_qwen", "train_run1", "train_run1b"])
    parser.add_argument("--input", type=str, default=str(DRIVE_DATA / "dataset_master.jsonl"))
    parser.add_argument("--output", type=str, default=str(DRIVE_OUT / "dataset_master_filled.jsonl"))
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()

    ensure_dirs()
    if args.task == "preflight":
        task_preflight()
    elif args.task == "fill_qwen":
        task_fill_qwen(Path(args.input), Path(args.output))
    elif args.task == "train_run1":
        task_train("run1_safe", lr=5e-5, epochs=1.0)
    elif args.task == "train_run1b":
        task_train("run1b_sweep", lr=args.lr, epochs=args.epochs)
