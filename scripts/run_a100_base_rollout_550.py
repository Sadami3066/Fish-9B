#!/usr/bin/env python3
"""
Fish-9B 固定上下文逐轮批量采集脚本 (Teacher-Forced Turn Rollout)
================================================================================
功能：
1. 读取指定的对话 JSONL，默认使用训练子集，拆成带完整历史上下文的独立推理请求；
2. 历史轮次仅保留干净的 assistant 正文（不含旧 20 字 <think>），完全契合 Qwen3.5
   多轮原生聊天模板；
3. 按 Prompt 长度排序 + 左填充批量并发推理（默认 batch_size=16，A100 80GB 可开 32），
   自动开启 Qwen3.5 原生 Thinking 模式（enable_thinking=True）；
4. 每完成一个 batch 立即增量落盘，支持 Colab 断连后断点续传；
5. 全部跑完后自动按原对话顺序拼回完整多轮 JSONL，方便后续“保留原版真 <think> +
   改写小鱼正文”以及顺手产出 On-Policy DPO 负样本。

Colab A100 一键运行示例：
    python /content/drive/MyDrive/Fish-9B/scripts/run_a100_base_rollout_550.py \
        --batch_size 24 \
        --max_new_tokens 2048
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

DRIVE_ROOT = os.environ.get("FISH9B_DRIVE_ROOT", "/content/drive/MyDrive/Fish-9B")

RE_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    return RE_THINK_BLOCK.sub("", text or "").strip()


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unpack_teacher_forced_tasks(rows: List[dict]) -> List[dict]:
    """将 550 条对话拆解为 1,348 个带固定历史上下文的独立推理任务。"""
    tasks = []
    for row_idx, row in enumerate(rows):
        msgs = row["messages"]
        assert len(msgs) >= 2 and len(msgs) % 2 == 0, f"Row {row_idx} 消息轮次异常"

        clean_history = []
        turn_idx = 0
        for i in range(0, len(msgs), 2):
            u_msg = msgs[i]
            a_msg = msgs[i + 1]
            assert u_msg["role"] == "user" and a_msg["role"] == "assistant"

            u_text = u_msg["content"]
            a_body = strip_think(a_msg["content"])

            # 当前轮输入 = 全部历史轮次（assistant 仅留正文） + 当前 user 提问
            prompt_messages = list(clean_history) + [{"role": "user", "content": u_text}]
            tasks.append(
                {
                    "task_key": f"r{row_idx:04d}_t{turn_idx:02d}",
                    "row_idx": row_idx,
                    "turn_idx": turn_idx,
                    "num_turns_in_row": len(msgs) // 2,
                    "user": u_text,
                    "xiaoyu_reply": a_body,
                    "prompt_messages": prompt_messages,
                }
            )

            clean_history.append({"role": "user", "content": u_text})
            clean_history.append({"role": "assistant", "content": a_body})
            turn_idx += 1
    return tasks


def load_completed_keys(turns_out_path: str) -> Dict[str, dict]:
    done = {}
    if not os.path.exists(turns_out_path):
        return done
    with open(turns_out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "task_key" in obj:
                    done[obj["task_key"]] = obj
            except json.JSONDecodeError:
                continue
    return done


def usable(record: dict) -> bool:
    return bool(record.get("base_reply", "").strip()) and not record.get("truncated", False)


def parse_thinking_and_reply(raw_text: str, prompt_ends_with_think: bool) -> Tuple[str, str]:
    """拆分 Qwen3.5 原生输出的 <think> 与最终正文。"""
    text = raw_text.strip()
    if prompt_ends_with_think and not text.startswith("<think>"):
        text = "<think>\n" + text

    if "</think>" in text:
        think_part, reply_part = text.split("</think>", 1)
        think_part = think_part.replace("<think>", "", 1).strip()
        reply_part = reply_part.strip()
        return think_part, reply_part

    # 极端情况：未闭合 </think>（通常是触发了 max_new_tokens 截断）
    if text.startswith("<think>"):
        return text.replace("<think>", "", 1).strip(), ""
    return "", text


def load_model_and_tokenizer(model_path: str, backend: str, max_seq_length: int):
    if backend == "unsloth":
        try:
            from unsloth import FastLanguageModel
            import torch

            print(f"[load] 使用 Unsloth bf16 加载原版模型: {model_path}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=False,
                load_in_16bit=True,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        except Exception as e:
            print(f"[warn] Unsloth 加载失败 ({e})，自动回退到 Transformers bf16...")

    import torch
    from transformers import AutoTokenizer

    print(f"[load] 使用 Transformers bf16 加载原版模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = None
    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def render_prompt(tokenizer, prompt_messages: List[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[Tuple[str, bool, int]]:
    import torch

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            top_k=top_k if temperature > 0 else None,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )

    results = []
    for row_ids in gen_ids:
        new_ids = row_ids[input_len:]
        token_ids = new_ids.tolist()
        stop = next((i for i, token in enumerate(token_ids) if token in (eos_ids or [])), None)
        gen_len = stop + 1 if stop is not None else len(token_ids)
        truncated = stop is None and gen_len >= max_new_tokens
        decoded = tokenizer.decode(token_ids[:gen_len], skip_special_tokens=True)
        results.append((decoded, truncated, gen_len))
    return results


def reassemble_conversations(turns_out_path: str, assembled_out_path: str, rows: List[dict]):
    """按 row_idx + turn_idx 拼回完整的 550 条多轮对话。"""
    done_map = load_completed_keys(turns_out_path)
    grouped = defaultdict(list)
    for rec in done_map.values():
        if not usable(rec):
            continue
        grouped[rec["row_idx"]].append(rec)

    think_lens = []
    reply_lens = []
    assembled_rows = 0

    temp_path = assembled_out_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        for row_idx, row in enumerate(rows):
            turns = sorted(grouped.get(row_idx, []), key=lambda x: x["turn_idx"])
            if [t["turn_idx"] for t in turns] != list(range(len(row["messages"]) // 2)):
                continue
            assembled_rows += 1
            clean_turns = []
            for t in turns:
                think_lens.append(t["base_think_chars"])
                reply_lens.append(t["base_reply_chars"])
                clean_turns.append(
                    {
                        "turn_idx": t["turn_idx"],
                        "user": t["user"],
                        "xiaoyu_reply": t["xiaoyu_reply"],
                        "base_thinking": t["base_thinking"],
                        "base_reply": t["base_reply"],
                        "base_think_chars": t["base_think_chars"],
                        "base_reply_chars": t["base_reply_chars"],
                        "gen_tokens": t["gen_tokens"],
                        "truncated": t["truncated"],
                    }
                )
            f.write(
                json.dumps(
                    {
                        "row_idx": row_idx,
                        "num_turns": len(clean_turns),
                        "turns": clean_turns,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    os.replace(temp_path, assembled_out_path)

    n = len(think_lens)
    if n > 0:
        avg_th = sum(think_lens) / n
        avg_rp = sum(reply_lens) / n
        med_th = sorted(think_lens)[n // 2]
        med_rp = sorted(reply_lens)[n // 2]
        print("\n================ 采集完成统计 ================")
        print(f"拼回对话数: {assembled_rows} / {len(rows)} 条（总轮次: {n}）")
        print(f"原版 <think> 字数: 均值={avg_th:.1f} | 中位数={med_th}")
        print(f"原版 正文    字数: 均值={avg_rp:.1f} | 中位数={med_rp}")
        print(f"逐轮原始文件: {turns_out_path}")
        print(f"拼回多轮文件: {assembled_out_path}")
        print("==============================================")


def main():
    parser = argparse.ArgumentParser(description="A100 固定上下文逐轮批量采集原版 Qwen3.5-9B 输出")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(DRIVE_ROOT, "model_cache/Qwen3.5-9B"),
        help="原版 Qwen3.5-9B 权重路径（本地缓存或 HF repo id）",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(DRIVE_ROOT, "data/train_ready_550_nothink/fish9b_xiaoyu_sft_523_train_nothink.jsonl"),
        help="输入 JSONL，支持 nothink 版或带 think 版",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(DRIVE_ROOT, "outputs"),
        help="结果输出目录",
    )
    parser.add_argument("--backend", type=str, default="unsloth", choices=["unsloth", "transformers"])
    parser.add_argument("--batch_size", type=int, default=16, help="A100 40GB 推荐 16，80GB 可设 24~32")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="调试用：只跑前 N 轮（0 表示全量）")
    parser.add_argument("--check_only", action="store_true", help="检查输入和拆轮，无需加载模型")
    args = parser.parse_args()

    if min(args.batch_size, args.max_new_tokens, args.max_seq_length) <= 0 or args.limit < 0:
        parser.error("batch_size、max_new_tokens、max_seq_length 必须大于零，limit 不能为负数")

    rows = load_jsonl(args.input)
    all_tasks = unpack_teacher_forced_tasks(rows)
    if not all_tasks:
        parser.error("没有可采集的对话")
    if args.check_only:
        print(f"[data] 输入对话 {len(rows)} 条 -> 总轮次 {len(all_tasks)}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    turns_out_path = os.path.join(args.out_dir, "base_qwen35_9b_rollouts_turns.jsonl")
    assembled_out_path = os.path.join(args.out_dir, "base_qwen35_9b_rollouts_550_assembled.jsonl")

    done_map = load_completed_keys(turns_out_path)
    tasks_by_key = {task["task_key"]: task for task in all_tasks}
    identity_fields = ("row_idx", "turn_idx", "user", "xiaoyu_reply")
    for key, record in done_map.items():
        task = tasks_by_key.get(key)
        if task is None or any(record.get(field) != task[field] for field in identity_fields):
            parser.error("输出目录含其他输入文件的采集结果，请指定新的 --out_dir")
    pending = [t for t in all_tasks if not usable(done_map.get(t["task_key"], {}))]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"[data] 输入对话 {len(rows)} 条 -> 总轮次 {len(all_tasks)} | 已完成 {len(done_map)} | 待跑 {len(pending)}")
    if not pending:
        print("[info] 所有轮次已跑完，直接重新拼装多轮文件...")
        if args.limit == 0:
            reassemble_conversations(turns_out_path, assembled_out_path, rows)
        return

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.backend, args.max_seq_length)
    import torch
    from tqdm import tqdm
    # 兼容 VLM processor 或纯 Tokenizer
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # 检测模板尾部是否自带 <think>\n
    sample_prompt = render_prompt(tok, [{"role": "user", "content": "hi"}])
    prompt_ends_with_think = "<think>" in sample_prompt[-20:]
    print(f"[tpl] prompt 尾部自带 <think>: {prompt_ends_with_think} | tail={sample_prompt[-36:]!r}")

    # 预渲染并按长度升序排序，大幅减少 batch 内 padding 浪费
    for t in pending:
        p_txt = render_prompt(tok, t["prompt_messages"])
        t["prompt_text"] = p_txt
        t["prompt_chars"] = len(p_txt)
    pending.sort(key=lambda x: x["prompt_chars"])

    t0 = time.time()
    total_gen_tokens = 0
    idx = 0
    cur_bs = args.batch_size

    pbar = tqdm(total=len(pending), desc="A100 Rollout", unit="turn")
    with open(turns_out_path, "a", encoding="utf-8") as fout:
        while idx < len(pending):
            batch = pending[idx : idx + cur_bs]
            prompts = [x["prompt_text"] for x in batch]
            try:
                outputs = generate_batch(
                    model=model,
                    tokenizer=tok,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_bs == 1:
                    raise
                cur_bs = max(1, cur_bs // 2)
                print(f"\n[oom] 显存不足，自动降 batch_size -> {cur_bs} 并重试...")
                continue

            for task, (raw_text, truncated, gen_len) in zip(batch, outputs):
                think_txt, reply_txt = parse_thinking_and_reply(raw_text, prompt_ends_with_think)
                total_gen_tokens += gen_len
                rec = {
                    "task_key": task["task_key"],
                    "row_idx": task["row_idx"],
                    "turn_idx": task["turn_idx"],
                    "user": task["user"],
                    "xiaoyu_reply": task["xiaoyu_reply"],
                    "base_thinking": think_txt,
                    "base_reply": reply_txt,
                    "base_think_chars": len(think_txt),
                    "base_reply_chars": len(reply_txt),
                    "gen_tokens": gen_len,
                    "truncated": truncated,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            idx += len(batch)
            pbar.update(len(batch))
            elapsed = max(1e-3, time.time() - t0)
            pbar.set_postfix(tok_s=f"{total_gen_tokens / elapsed:.0f}", bs=cur_bs)
    pbar.close()

    elapsed = time.time() - t0
    print(f"\n[done] 推理耗时 {elapsed / 60:.1f} 分钟 | 总生成 {total_gen_tokens} tokens ({total_gen_tokens / max(1e-3, elapsed):.0f} tok/s)")
    failed = sum(not usable(rec) for rec in load_completed_keys(turns_out_path).values())
    print(f"[quality] 截断或正文为空的轮次: {failed}（下次运行会重试）")
    if args.limit == 0:
        reassemble_conversations(turns_out_path, assembled_out_path, rows)


if __name__ == "__main__":
    main()
