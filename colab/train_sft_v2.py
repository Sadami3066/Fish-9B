#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小鱼 (Fish-9B) SFT v2 训练脚本 —— Colab A100, bf16 LoRA, assistant-only loss
自包含: transformers + peft (不依赖 ms-swift 参数名), 手工拼 ChatML 保证每一轮 <think> 都保留。

用法 (Colab 上, Drive 已挂载):
  python train_sft_v2.py --check_only          # 只做数据/掩码/模板核验 (无需 GPU 权重)
  python train_sft_v2.py                       # 正式训练 (默认超参见下)
  python train_sft_v2.py --lr 5e-5 --epochs 2  # 保守档
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

ROOT = Path(os.environ.get("FISH9B_DRIVE_ROOT", "/content/drive/MyDrive/Fish-9B"))
DEFAULT_TRAIN = ROOT / "data/train_ready_550/fish9b_xiaoyu_sft_523_train.jsonl"
DEFAULT_VAL = ROOT / "data/train_ready_550/fish9b_xiaoyu_sft_27_val.jsonl"
MODEL = ROOT / "model_cache/Qwen3.5-9B"
CKPT = ROOT / "checkpoints"

p = argparse.ArgumentParser()
p.add_argument("--train_file", type=str, default=str(DEFAULT_TRAIN))
p.add_argument("--val_file", type=str, default=str(DEFAULT_VAL))
p.add_argument("--model_path", default=str(MODEL))
p.add_argument("--output_dir", type=Path, help="本次训练的输出目录，默认写入 Drive 下新的 checkpoint 目录")
p.add_argument("--resume_from_checkpoint", type=Path, help="包含优化器和训练状态的 checkpoint 目录")
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--epochs", type=float, default=3.0)
p.add_argument("--rank", type=int, default=32)
p.add_argument("--alpha", type=int, default=64)
p.add_argument("--dropout", type=float, default=0.05)
p.add_argument("--max_len", type=int, default=4096)
p.add_argument("--bs", type=int, default=2)          # per-device
p.add_argument("--ga", type=int, default=4)          # effective batch = bs*ga = 8
p.add_argument("--seed", type=int, default=42)
p.add_argument("--tag", type=str, default="xiaoyu_sft_v2")
p.add_argument("--backend", choices=["hf", "unsloth"], default="hf")
p.add_argument("--optim", type=str, default="adamw_torch")
p.add_argument("--nothink", action="store_true", help="强制非思考模板 (enable_thinking=False, 不强制 <think>)")
p.add_argument("--check_only", action="store_true")
args = p.parse_args()
TRAIN, VAL = Path(args.train_file), Path(args.val_file)
MODEL = args.model_path

if args.backend == "unsloth":
    from unsloth import FastLanguageModel  # 必须先于 transformers/peft 导入
import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
IM_START, IM_END = "<|im_start|>", "<|im_end|>"

# ---------- 推理侧前缀: 训练必须与推理一致 ----------
probe_kw = {"enable_thinking": False} if args.nothink else {}
probe = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True, **probe_kw)
GEN_PREFIX_THINK = (not args.nothink) and probe.endswith("<think>\n")
NOTHINK_SUFFIX = "<think>\n\n</think>\n\n"
print("[tpl] generation prompt tail:", repr(probe[-60:]), "| nothink =", args.nothink, "| think-in-prompt =", GEN_PREFIX_THINK)
if args.nothink:
    assert probe.endswith(NOTHINK_SUFFIX), f"非思考模板结尾非预期: {probe[-60:]!r}"


def load(path):
    rows = []
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            rows.append(json.loads(ln)["messages"])
    return rows


def encode(messages):
    """空 system; 仅 assistant 内容(+<|im_end|>)参与 loss; 支持思考保留与 enable_thinking=False 两种模式。"""
    ids, labels = [], []
    assert messages[0]["role"] == "user", "首条必须是 user (空 system)"
    last_user_idx = max(idx for idx, msg in enumerate(messages) if msg["role"] == "user")
    for idx, m in enumerate(messages):
        if m["role"] == "user":
            t = f"{IM_START}user\n{m['content']}{IM_END}\n{IM_START}assistant\n"
            a = tok(t, add_special_tokens=False)["input_ids"]
            ids += a; labels += [-100] * len(a)
        elif m["role"] == "assistant":
            c = m["content"]
            if args.nothink:
                assert "<think>" not in c and "</think>" not in c, f"nothink 数据不应含 <think>: {c[:40]!r}"
                if idx > last_user_idx:
                    # 精确对齐 Qwen3.5 chat_template.jinja(enable_thinking=False): 最后一轮前导空 <think>\n\n</think>\n\n 属于 prompt 侧
                    a = tok(NOTHINK_SUFFIX, add_special_tokens=False)["input_ids"]
                    ids += a; labels += [-100] * len(a)
            else:
                assert c.startswith("<think>") and "</think>" in c, f"缺 <think>: {c[:40]!r}"
                if GEN_PREFIX_THINK:
                    # 推理时 prompt 已含 "<think>\n"; 训练时把它挂在 prompt 侧(不算 loss), 其余算 loss
                    head = "<think>\n"
                    body = c[len("<think>"):].lstrip("\n")
                    a = tok(head, add_special_tokens=False)["input_ids"]
                    ids += a; labels += [-100] * len(a)
                    c = body
            b = tok(c + IM_END, add_special_tokens=False)["input_ids"]
            ids += b; labels += b
            n = tok("\n", add_special_tokens=False)["input_ids"]
            ids += n; labels += [-100] * len(n)
        else:
            raise ValueError(f"unexpected role {m['role']}")
    if args.nothink:
        ref = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        assert tok.decode(ids) == ref, f"nothink 编码与官方 jinja 不一致!\nGOT: {tok.decode(ids)!r}\nREF: {ref!r}"
    return ids, labels


def build(rows, name):
    out, lens, dropped = [], [], 0
    for msgs in rows:
        i, l = encode(msgs)
        if len(i) > args.max_len:
            dropped += 1
            continue
        out.append({"input_ids": i, "labels": l}); lens.append(len(i))
    lens_s = sorted(lens)
    if not lens_s:
        raise ValueError(f"{name} 数据为空，或全部超过 --max_len={args.max_len}")
    print(f"[data] {name}: {len(out)} 条 (超长丢弃 {dropped}) | tokens mean={sum(lens)/len(lens):.0f} "
          f"p95={lens_s[int(len(lens)*.95)]} max={lens_s[-1]} | loss tokens="
          f"{sum(sum(1 for x in o['labels'] if x!=-100) for o in out)}")
    return out


train_rows, val_rows = load(TRAIN), load(VAL)
print(f"[data] raw train={len(train_rows)} val={len(val_rows)}")
train_ds, val_ds = build(train_rows, "train"), build(val_rows, "val")

# ---------- 掩码肉眼核验 ----------
ex = max(train_ds, key=lambda o: o["labels"].count(-100))   # 取一条多轮样本
print("\n[mask] 被训练部分 (只应是各轮 assistant <think>…</think>正文<|im_end|>):")
seg, cur = [], []
for t, l in zip(ex["input_ids"], ex["labels"]):
    if l != -100: cur.append(t)
    elif cur: seg.append(cur); cur = []
if cur: seg.append(cur)
for s in seg:
    print("  ▶", repr(tok.decode(s))[:160])
print("[mask] 完整文本:\n", tok.decode(ex["input_ids"])[:600], "\n")
assert not any("user" in tok.decode(s)[:8] for s in seg), "user token 泄漏进 loss!"
if args.check_only:
    print("[check_only] 通过, 未加载模型。"); sys.exit(0)

# ---------- 模型 + LoRA (全部线性层, 仅语言主干) ----------
from transformers import AutoModelForImageTextToText, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

SKIP = ("visual", "vision", "merger", "aligner", "lm_head", "embed_tokens")
if args.backend == "unsloth":
    model, _tk = FastLanguageModel.from_pretrained(model_name=str(MODEL), max_seq_length=args.max_len,
                                                   load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    base = model
else:
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(MODEL), dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    base = model
model.config.use_cache = False

targets = sorted({n for n, m in base.named_modules()
                  if isinstance(m, torch.nn.Linear) and not any(s_ in n for s_ in SKIP)})
leaf = sorted({n.split(".")[-1] for n in targets})
print(f"[lora] 目标线性层 {len(targets)} 个, 叶子类型: {leaf}")
assert any(x in leaf for x in ("q_proj", "gate_proj")), "没匹配到 attention/mlp 线性层"

if args.backend == "unsloth":
    model = FastLanguageModel.get_peft_model(model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                                             target_modules=leaf, use_gradient_checkpointing="unsloth", random_state=args.seed)
    FastLanguageModel.for_training(model)
else:
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                     target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
leaked = [n for n, q in model.named_parameters() if q.requires_grad and any(s_ in n for s_ in SKIP)]
assert not leaked, f"视觉/embedding/head 被误训: {leaked[:3]}"
model.print_trainable_parameters()


def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([b["input_ids"] + [pad] * (L - len(b["input_ids"])) for b in batch])
    lab = torch.tensor([b["labels"] + [-100] * (L - len(b["labels"])) for b in batch])
    att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (L - len(b["input_ids"])) for b in batch])
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


import inspect
steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.bs * args.ga)))
total_steps = max(1, int(steps_per_epoch * args.epochs))
eval_every = max(1, steps_per_epoch // 2)
out_dir = args.output_dir or CKPT / f"{args.tag}_r{args.rank}_lr{args.lr}_ep{args.epochs:g}_{time.strftime('%m%d_%H%M%S')}"
tb_root = Path(os.environ.get("FISH9B_TB_DIR", "/content/tb_logs" if Path("/content").is_dir() else "tb_logs"))
ta_sig = inspect.signature(TrainingArguments.__init__).parameters
ta_kw = dict(
    output_dir=str(out_dir), num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs,
    gradient_accumulation_steps=args.ga, learning_rate=args.lr,
    lr_scheduler_type="cosine", warmup_steps=max(1, int(total_steps * 0.05)), weight_decay=0.0, max_grad_norm=1.0,
    bf16=True, logging_steps=2, logging_first_step=True,
    eval_strategy="steps", eval_steps=eval_every,          # 每半个 epoch 看一次 val loss
    save_strategy="steps", save_steps=eval_every, save_total_limit=8,
    seed=args.seed, data_seed=args.seed, report_to="tensorboard",
    logging_dir=str(tb_root / out_dir.name),
    optim=args.optim, remove_unused_columns=False, group_by_length=False,
)
dropped_ta = sorted(k for k in ta_kw if k not in ta_sig)
if dropped_ta:
    print(f"[train] 跳过当前 transformers 不再支持的 TrainingArguments 字段: {dropped_ta}", flush=True)
targs = TrainingArguments(**{k: v for k, v in ta_kw.items() if k in ta_sig})
tb_root.mkdir(parents=True, exist_ok=True)
tb_link = tb_root / out_dir.name
if "logging_dir" not in ta_sig and not tb_link.exists() and not tb_link.is_symlink():
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_link.symlink_to(out_dir.resolve() / "runs", target_is_directory=True)
trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate)
print(f"[train] steps/epoch≈{steps_per_epoch}, total≈{total_steps}, out={out_dir}", flush=True)
trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
model.save_pretrained(str(out_dir / "final_adapter"))
tok.save_pretrained(str(out_dir / "final_adapter"))
json.dump(trainer.state.log_history, open(out_dir / "log_history.json", "w"), ensure_ascii=False, indent=1)
os.sync()
print("[DONE] adapter ->", out_dir / "final_adapter", flush=True)
