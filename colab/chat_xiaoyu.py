#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小鱼 (Fish-9B) Agent 框架模拟对话台。使用 --share 可生成 Gradio 分享链接。
支持:
1. 自定义 System Prompt + OpenAI Function Calling Tools JSON Schema
2. 严格对齐 Qwen3.5 原生 chat_template.jinja (<tools> / <tool_call><function=...><parameter=...> / <tool_response>)
3. 截获模型输出的 <tool_call>, 支持手动注入 Tool Response 或启用自动 Mock 规则多步执行
4. 实时折叠查看底层完整 Raw ChatML Prompt, 热切换各 Checkpoint 与原版 Qwen3.5-9B 对比
"""
import argparse
import json
import os
import re
import threading
from pathlib import Path

DEFAULT_TOOLS_JSON = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取工作区指定路径的文件内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件绝对或相对路径"},
                        "start_line": {"type": "integer", "description": "起始行号(可选)"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "在沙盒终端中执行 shell 命令并返回输出",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string", "description": "要执行的 bash 命令"},
                    },
                    "required": ["cmd"],
                },
            },
        },
    ],
    ensure_ascii=False,
    indent=2,
)

DEFAULT_MOCK_JSON = json.dumps(
    {
        "read_file": {"ok": True, "content": "def hello():\n    return 'world'\n"},
        "run_shell": {"exit_code": 0, "stdout": "ALL TESTS PASSED (12/12)\n"},
    },
    ensure_ascii=False,
    indent=2,
)

PRESET_PROMPTS = {
    "空 (SFT 默认无 System)": "",
    "小鱼人设 (精简版)": "你是小鱼。说话直接、敏锐、自然，看清重点再开口，不讲客服套话。",
    "编码 Agent 框架": (
        "你是一个严谨又嘴硬心软的编程助手。调用工具前先想清楚最少需要查什么；"
        "拿到工具结果后再下结论，不要编造没看见的文件内容。"
    ),
}

CKPT_ROOT = Path(os.environ.get("FISH9B_DRIVE_ROOT", "/content/drive/MyDrive/Fish-9B")) / "checkpoints"

p = argparse.ArgumentParser()
p.add_argument("--run_dir", default="auto", help="auto=在 checkpoint 目录自动选最新训练")
p.add_argument("--compare_adapter", type=Path, help="额外加载一份兼容的 LoRA adapter 作为对照")
p.add_argument("--port", type=int, default=7860)
p.add_argument("--share", action="store_true", help="显式开启 Gradio 公网分享")
args = p.parse_args()

from unsloth import FastLanguageModel
import gradio as gr
import torch
from transformers import TextIteratorStreamer


def resolve_run_dir(raw: str) -> Path:
    if raw and raw != "auto":
        return Path(raw)
    if CKPT_ROOT.is_dir():
        runs = [
            path for path in CKPT_ROOT.iterdir()
            if path.is_dir() and (any(path.glob("checkpoint-*")) or (path / "final_adapter").is_dir())
        ]
        if runs:
            return max(runs, key=lambda path: path.stat().st_mtime)
    raise SystemExit(f"在 {CKPT_ROOT} 中未找到训练输出，请通过 --run_dir 指定。")


def discover_checkpoints(run_dir: Path) -> list[tuple[str, str, Path]]:
    eval_by_step = {}
    ep_by_step = {}
    lh = run_dir / "log_history.json"
    if lh.exists():
        try:
            for item in json.loads(lh.read_text(encoding="utf-8")):
                if "eval_loss" in item and "step" in item:
                    eval_by_step[int(item["step"])] = float(item["eval_loss"])
                if "epoch" in item and "step" in item:
                    ep_by_step[int(item["step"])] = float(item["epoch"])
        except Exception:
            pass
    ckpts = []
    for d in sorted(run_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 999999):
        step_str = d.name.split("-")[-1]
        step = int(step_str) if step_str.isdigit() else -1
        ep = ep_by_step.get(step)
        ev = eval_by_step.get(step)
        meta = []
        if ep is not None:
            meta.append(f"ep{ep:.2g}")
        if ev is not None:
            meta.append(f"val={ev:.3f}")
        tag = f" ({' · '.join(meta)})" if meta else ""
        ckpts.append((f"{d.name}{tag}", f"step{step}", d, ev))
    if (run_dir / "final_adapter").exists():
        ckpts.append(("final_adapter (结束)", "final", run_dir / "final_adapter", None))
    if not ckpts:
        raise RuntimeError(f"在 {run_dir} 未找到任何 checkpoint")
    # 优先把 eval_loss 最低的放到第一位作为默认加载; 若没有 eval_loss 则用最后一个
    scored = [c for c in ckpts if c[3] is not None]
    best = min(scored, key=lambda c: c[3]) if scored else ckpts[-1]
    ordered = [(best[0] + " [推荐]", "default", best[2])]
    for label, name, path, _ in ckpts:
        if path != best[2]:
            ordered.append((label, name, path))
    if args.compare_adapter:
        ordered.append((f"对照 adapter ({args.compare_adapter.name})", "compare_adapter", args.compare_adapter))
    return ordered


RUN = resolve_run_dir(args.run_dir)
CHOICES = discover_checkpoints(RUN)
print(f"[chat] 使用目录: {RUN}", flush=True)
print(f"[chat] 默认加载: {CHOICES[0][0]} -> {CHOICES[0][2]}", flush=True)

model, tok = FastLanguageModel.from_pretrained(
    model_name=str(CHOICES[0][2]),
    max_seq_length=4096,
    load_in_4bit=False,
    load_in_16bit=True,
)
loaded = {CHOICES[0][0]: CHOICES[0][1]}
for label, name, path in CHOICES[1:]:
    if path.exists():
        try:
            model.load_adapter(str(path), adapter_name=name)
            loaded[label] = name
            print(f"[chat] 已挂载适配器: {label}", flush=True)
        except Exception as e:
            print(f"[chat] 跳过 {label}: {e}", flush=True)

FastLanguageModel.for_inference(model)
tokenizer = getattr(tok, "tokenizer", tok)
BASE_LABEL = "原版 Qwen3.5-9B (关闭 LoRA 对比)"
ALL_LABELS = list(loaded.keys()) + [BASE_LABEL]
THINK_MODES = [
    "💬 关思考闲聊 (enable_thinking=False)",
    "🧠 开思考写代码/推理 (enable_thinking=True)",
]
GEN_LOCK = threading.Lock()


def parse_json_safe(raw: str, expect_type):
    raw = (raw or "").strip()
    if not raw:
        return expect_type(), ""
    try:
        val = json.loads(raw)
        if not isinstance(val, expect_type):
            return None, f"JSON 类型必须是 {expect_type.__name__}"
        return val, ""
    except Exception as e:
        return None, f"JSON 解析错误: {e}"


def parse_tool_calls(text: str) -> list[dict]:
    """解析 Qwen3.5 原生 XML (<function=...>...</function>) 或 JSON 格式的 <tool_call>。"""
    calls = []
    for blk in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL):
        fn_m = re.search(r"<function=([^\s>]+)>\s*(.*?)\s*</function>", blk, flags=re.DOTALL)
        if fn_m:
            name = fn_m.group(1).strip()
            body = fn_m.group(2)
            params = {}
            for k, v in re.findall(r"<parameter=([^\s>]+)>\n?(.*?)\n?</parameter>", body, flags=re.DOTALL):
                v_s = v.strip()
                try:
                    params[k.strip()] = json.loads(v_s)
                except Exception:
                    params[k.strip()] = v_s
            calls.append({"name": name, "arguments": params, "raw": blk})
        else:
            try:
                obj = json.loads(blk)
                calls.append({"name": obj.get("name", "?"), "arguments": obj.get("arguments", obj), "raw": blk})
            except Exception:
                calls.append({"name": "raw", "arguments": blk, "raw": blk})
    return calls


def build_prompt(messages: list[dict], tools: list[dict] | None, keep_all_think: bool, enable_thinking: bool) -> str:
    """
    复用 Qwen3.5 原生 chat_template.jinja 处理 system + <tools> + <tool_response> + enable_thinking。
    当 enable_thinking=True 且 keep_all_think=True 时, 早期 assistant 轮也保留 <think>。
    """
    msgs = [dict(m) for m in messages]
    if (not enable_thinking) or (not keep_all_think):
        return tokenizer.apply_chat_template(
            msgs,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    sys_msgs = [m for m in msgs if m["role"] == "system"]
    probe = sys_msgs + [{"role": "user", "content": "__PROBE__"}]
    hdr_full = tokenizer.apply_chat_template(
        probe, tools=tools if tools else None, tokenize=False, add_generation_prompt=False
    )
    hdr = hdr_full.split("<|im_start|>user\n__PROBE__<|im_end|>\n")[0]
    parts = [hdr]
    i = 0
    while i < len(msgs):
        m = msgs[i]
        r = m["role"]
        c = (m.get("content") or "").strip()
        if r == "system":
            i += 1
        elif r == "user":
            parts.append(f"<|im_start|>user\n{c}<|im_end|>\n")
            i += 1
        elif r == "assistant":
            parts.append(f"<|im_start|>assistant\n{c}<|im_end|>\n")
            i += 1
        elif r == "tool":
            parts.append("<|im_start|>user")
            while i < len(msgs) and msgs[i]["role"] == "tool":
                tc = (msgs[i].get("content") or "").strip()
                parts.append(f"\n<tool_response>\n{tc}\n</tool_response>")
                i += 1
            parts.append("<|im_end|>\n")
        else:
            i += 1
    parts.append("<|im_start|>assistant\n<think>\n")
    return "".join(parts)


def clean_specials(s: str) -> str:
    for sp in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        s = s.replace(sp, "")
    return s


def _fence_tool_calls(text: str) -> str:
    # 防止 Markdown/HTML 吞掉 <tool_call><function=...><parameter=...> 标签
    text = re.sub(r"(<tool_call>.*?</tool_call>)", r"\n```xml\n\1\n```\n", text, flags=re.DOTALL)
    if "<tool_call>" in text and "</tool_call>" not in text.split("<tool_call>")[-1]:
        head, tail = text.rsplit("<tool_call>", 1)
        text = f"{head}\n```xml\n<tool_call>{tail}\n```"
    return text


def as_display(raw_assistant: str, show_think: bool) -> str:
    if not show_think:
        body = raw_assistant.split("</think>", 1)[1].lstrip("\n") if "</think>" in raw_assistant else raw_assistant
        return _fence_tool_calls(body)
    if "</think>" in raw_assistant:
        head, tail = raw_assistant.split("</think>", 1)
        return f"{head}</think>{_fence_tool_calls(tail)}"
    return _fence_tool_calls(raw_assistant)


def stream_one_turn(messages, tools, ckpt_label, temp, top_p, max_tokens, keep_all_think, enable_thinking):
    prompt = build_prompt(messages, tools, keep_all_think, enable_thinking)
    with GEN_LOCK:
        inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to("cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        raw_eos = tokenizer.eos_token_id
        stop_ids = list(raw_eos) if isinstance(raw_eos, (list, tuple)) else ([raw_eos] if raw_eos is not None else [])
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0 and im_end not in stop_ids:
            stop_ids.append(im_end)

        gen_kw = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=int(max_tokens),
            do_sample=float(temp) > 0,
            temperature=max(float(temp), 1e-5),
            top_p=float(top_p),
            eos_token_id=stop_ids,
            use_cache=True,
        )
        gen_err = []

        def _run():
            try:
                if ckpt_label == BASE_LABEL:
                    with model.disable_adapter():
                        with torch.inference_mode():
                            model.generate(**gen_kw)
                else:
                    model.set_adapter(loaded.get(ckpt_label, "default"))
                    with torch.inference_mode():
                        model.generate(**gen_kw)
            except Exception as e:
                gen_err.append(repr(e))
                print("[chat][GEN_ERR]", repr(e), flush=True)
            finally:
                streamer.end()

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        acc = "<think>\n" if enable_thinking else ""
        for piece in streamer:
            piece = clean_specials(piece)
            if not piece:
                continue
            acc += piece
            yield acc, prompt
        th.join()
        if gen_err:
            acc += f"\n\n[生成出错] {gen_err[0]}"
            yield acc, prompt


def sync_system_msg(msgs: list[dict], sys_prompt: str) -> list[dict]:
    msgs = [dict(m) for m in (msgs or []) if m.get("role") != "system"]
    if (sys_prompt or "").strip():
        msgs.insert(0, {"role": "system", "content": sys_prompt.strip()})
    return msgs


def run_agent_step(
    chat_hist, raw_msgs, sys_prompt, tools_str, ckpt_label, think_mode, temp, top_p, max_tokens, show_think, keep_all_think, auto_mock, mock_str
):
    enable_thinking = "True" in (think_mode or "")
    tools, t_err = parse_json_safe(tools_str, list)
    if t_err:
        chat_hist.append({"role": "assistant", "content": f"❌ Tools JSON 格式有误: {t_err}"})
        yield chat_hist, raw_msgs, f"Tools JSON 错误: {t_err}", "", ""
        return
    mocks, m_err = parse_json_safe(mock_str, dict)
    if mocks is None:
        mocks = {}

    raw_msgs = sync_system_msg(raw_msgs, sys_prompt)
    last_prompt = ""
    for step_idx in range(4):  # 自动 Mock 最多连续 4 步防止死循环
        chat_hist.append({"role": "assistant", "content": ""})
        acc = ""
        for acc, last_prompt in stream_one_turn(raw_msgs, tools, ckpt_label, temp, top_p, max_tokens, keep_all_think, enable_thinking):
            chat_hist[-1]["content"] = as_display(acc, show_think)
            yield chat_hist, raw_msgs, "生成中…", "", last_prompt

        acc = acc.strip()
        raw_msgs.append({"role": "assistant", "content": acc})
        calls = parse_tool_calls(acc)
        if not calls:
            yield chat_hist, raw_msgs, "完成 (无挂起 tool_call)", "", last_prompt
            return

        pretty_calls = json.dumps(calls, ensure_ascii=False, indent=2)
        if auto_mock and all(c["name"] in mocks for c in calls):
            # 自动把所有 mock 返回值作为 tool 消息回传并继续下一轮
            for c in calls:
                res = mocks[c["name"]]
                res_s = json.dumps(res, ensure_ascii=False) if not isinstance(res, str) else res
                raw_msgs.append({"role": "tool", "content": res_s})
                chat_hist.append({"role": "user", "content": f"🛠️ **[Auto Tool `{c['name']}` 返回]**\n```json\n{res_s}\n```"})
            yield chat_hist, raw_msgs, f"已自动回传 {len(calls)} 个 Mock 工具结果，继续生成…", "", last_prompt
            continue
        else:
            # 停下来等待用户在下方回传 Tool Response
            first_name = calls[0]["name"]
            prefill = mocks.get(first_name, {"status": "ok", "result": f"模拟 {first_name} 返回结果"})
            prefill_s = json.dumps(prefill, ensure_ascii=False, indent=2) if not isinstance(prefill, str) else prefill
            status_msg = f"⏸️ 模型发起了 {len(calls)} 个 Tool Call，请在下方编辑返回值并点【回传 Tool Response】:\n{pretty_calls}"
            yield chat_hist, raw_msgs, status_msg, prefill_s, last_prompt
            return


def on_user_send(
    user_msg, chat_hist, raw_msgs, sys_prompt, tools_str, ckpt_label, think_mode, temp, top_p, max_tokens, show_think, keep_all_think, auto_mock, mock_str
):
    if not (user_msg or "").strip():
        yield "", chat_hist, raw_msgs, "", "", ""
        return
    chat_hist = list(chat_hist or [])
    raw_msgs = sync_system_msg(raw_msgs, sys_prompt)
    raw_msgs.append({"role": "user", "content": user_msg})
    chat_hist.append({"role": "user", "content": user_msg})
    yield "", chat_hist, raw_msgs, "准备生成…", "", ""
    for c, r, st, pre, prm in run_agent_step(
        chat_hist, raw_msgs, sys_prompt, tools_str, ckpt_label, think_mode, temp, top_p, max_tokens, show_think, keep_all_think, auto_mock, mock_str
    ):
        yield "", c, r, st, pre, prm


def on_tool_send(
    tool_resp_text, chat_hist, raw_msgs, sys_prompt, tools_str, ckpt_label, think_mode, temp, top_p, max_tokens, show_think, keep_all_think, auto_mock, mock_str
):
    tool_resp_text = (tool_resp_text or "").strip()
    if not tool_resp_text:
        yield chat_hist, raw_msgs, "请先输入 Tool Response 内容", tool_resp_text, ""
        return
    chat_hist = list(chat_hist or [])
    raw_msgs = sync_system_msg(raw_msgs, sys_prompt)
    raw_msgs.append({"role": "tool", "content": tool_resp_text})
    chat_hist.append({"role": "user", "content": f"🛠️ **[Tool Response]**\n```\n{tool_resp_text}\n```"})
    yield chat_hist, raw_msgs, "已喂回 Tool Response，模型继续推理中…", "", ""
    for c, r, st, pre, prm in run_agent_step(
        chat_hist, raw_msgs, sys_prompt, tools_str, ckpt_label, think_mode, temp, top_p, max_tokens, show_think, keep_all_think, auto_mock, mock_str
    ):
        yield c, r, st, pre, prm


def on_clear():
    return [], [], "已清空对话", "", ""


def switch_preset_chat():
    return THINK_MODES[0], PRESET_PROMPTS["空 (SFT 默认无 System)"], "", 1024


def switch_preset_code():
    return THINK_MODES[1], PRESET_PROMPTS["编码 Agent 框架"], DEFAULT_TOOLS_JSON, 1536


with gr.Blocks(title="小鱼 Fish-9B · 非思考/思考双模式 & 框架模拟沙盒") as demo:
    gr.Markdown(
        "## 小鱼 `Fish-9B` · 非思考/思考双模式 & Agent 框架模拟台\n"
        "支持切换 **💬 关思考闲聊 (`enable_thinking=False`)** 与 **🧠 开思考写代码 (`enable_thinking=True`)**；"
        "同时支持自由配置 `System Prompt`、`Tools (JSON Schema)` 及手动/Auto-Mock 回传工具结果。"
    )
    with gr.Row():
        ckpt_dd = gr.Dropdown(choices=ALL_LABELS, value=ALL_LABELS[0], label="选择模型 / Checkpoint", scale=3)
        think_mode_rb = gr.Radio(choices=THINK_MODES, value=THINK_MODES[0], label="对话模式 (Thinking 开关)", scale=3)
    with gr.Row():
        mode_chat_btn = gr.Button("一键切换: 💬 关思考闲聊 (空System+无Tools)", scale=2)
        mode_code_btn = gr.Button("一键切换: 🧠 开思考写代码 (编码Agent+Tools)", scale=2)
        show_think_cb = gr.Checkbox(value=True, label="显示 <think>", scale=1)
        keep_think_cb = gr.Checkbox(value=False, label="历史保留 <think>", scale=1)

    with gr.Accordion("框架配置: System Prompt & Tools Schema & Auto-Mock", open=False):
        with gr.Row():
            preset_dd = gr.Dropdown(choices=list(PRESET_PROMPTS.keys()), value="空 (SFT 默认无 System)", label="System 预设", scale=2)
            load_tools_btn = gr.Button("载入示例 Tools", scale=1)
            clear_tools_btn = gr.Button("清空 Tools (纯对话)", scale=1)
        with gr.Row():
            sys_box = gr.Textbox(
                value="",
                label="System Prompt",
                lines=5,
                placeholder="输入任意 System Prompt（留空则无 system 块）",
                scale=1,
            )
            tools_box = gr.Textbox(
                value="",
                label="Tools 定义 (OpenAI JSON 列表, 留空则不注入 <tools>)",
                lines=8,
                scale=1,
            )
            mock_box = gr.Textbox(
                value=DEFAULT_MOCK_JSON,
                label="Mock 工具返回值映射 (工具名 -> JSON/文本)",
                lines=8,
                scale=1,
            )
        auto_mock_cb = gr.Checkbox(value=False, label="勾选后命中 Mock 映射自动回传 Tool Response 并继续生成（不勾选则暂停等你手填）")

    bot = gr.Chatbot(height=460, reasoning_tags=[("<think>", "</think>")])
    raw_state = gr.State([])

    with gr.Row():
        user_box = gr.Textbox(placeholder="跟小鱼聊两句，或切到开思考写代码模式让她写段代码…", show_label=False, scale=6, autofocus=True)
        send_btn = gr.Button("发送 User", variant="primary", scale=1)
        clr_btn = gr.Button("清空对话", scale=1)

    with gr.Group():
        tool_status = gr.Textbox(label="Tool Call 状态 / 拦截到的调用参数", lines=4, interactive=False)
        with gr.Row():
            tool_resp_box = gr.Textbox(label="Tool Response 输入区 (JSON 或纯文本)", lines=3, scale=5, placeholder='例如: {"status": "ok", "data": ...}')
            tool_send_btn = gr.Button("📤 回传 Tool Response 并继续", variant="secondary", scale=1)

    with gr.Accordion("调试: 查看底层真实 ChatML Prompt 与生成参数", open=False):
        with gr.Row():
            temp_sl = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            topp_sl = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            max_sl = gr.Slider(128, 2048, value=1024, step=64, label="Max New Tokens")
        raw_prompt_view = gr.Textbox(label="最近一次送入模型的完整 Raw Prompt", lines=10, interactive=False)

    preset_dd.change(lambda k: PRESET_PROMPTS.get(k, ""), inputs=[preset_dd], outputs=[sys_box])
    load_tools_btn.click(lambda: DEFAULT_TOOLS_JSON, outputs=[tools_box])
    clear_tools_btn.click(lambda: "", outputs=[tools_box])
    mode_chat_btn.click(switch_preset_chat, outputs=[think_mode_rb, sys_box, tools_box, max_sl])
    mode_code_btn.click(switch_preset_code, outputs=[think_mode_rb, sys_box, tools_box, max_sl])

    send_inputs = [user_box, bot, raw_state, sys_box, tools_box, ckpt_dd, think_mode_rb, temp_sl, topp_sl, max_sl, show_think_cb, keep_think_cb, auto_mock_cb, mock_box]
    send_outputs = [user_box, bot, raw_state, tool_status, tool_resp_box, raw_prompt_view]
    user_box.submit(on_user_send, send_inputs, send_outputs)
    send_btn.click(on_user_send, send_inputs, send_outputs)

    tool_inputs = [tool_resp_box, bot, raw_state, sys_box, tools_box, ckpt_dd, think_mode_rb, temp_sl, topp_sl, max_sl, show_think_cb, keep_think_cb, auto_mock_cb, mock_box]
    tool_outputs = [bot, raw_state, tool_status, tool_resp_box, raw_prompt_view]
    tool_send_btn.click(on_tool_send, tool_inputs, tool_outputs)
    clr_btn.click(on_clear, outputs=[bot, raw_state, tool_status, tool_resp_box, raw_prompt_view])

print("[chat] 正在启动双模式 + Tool-Call 框架模拟器 Gradio 服务...", flush=True)
demo.queue().launch(server_name="127.0.0.1", server_port=args.port, share=args.share, show_error=True)
