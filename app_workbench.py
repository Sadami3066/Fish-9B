#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大肥鱼-9B (Base) 本地 Gradio 四栏数据工作台 & 盲测面板 (app_workbench.py)
启动: python app_workbench.py
"""

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from audit_dataset import (
    MASTER_PATH,
    audit_and_export,
    estimate_tokens,
    format_report_markdown,
    load_master_dataset,
    save_master_dataset,
)
from gdrive_bridge import status_all_accounts, sync_data_to_drive

IMAGES_DIR = BASE_DIR / "data" / "images"

EDIT_TAG_CHOICES = [
    "开头两句",
    "措辞去油",
    "信息取舍",
    "中后轮防回弹",
    "言外之意重写",
    "精确交付去废话",
]

TASK_TYPE_CHOICES = [
    "多轮编程排错与纠偏",
    "多模态看图纠错与深挖",
    "多轮日常吐槽与闲聊",
    "多模态截屏排错多轮",
    "复杂概念完整长答",
    "技术选型完整长答",
    "精确交付_JSON",
    "精确交付_代码",
    "精确交付-命令编写",
    "多模态看图与不确定性表达",
    "单轮看图-界面找茬",
    "日常半截话快问快答",
]

MAX_TURNS = 5


def compress_and_save_image(temp_img_path: Optional[str], group_id: str) -> Optional[str]:
    """将上传的图片等比缩放至最长边 ≤1280px 并保存为轻量 JPEG (~80KB)，大幅节省 VPN 上传流量"""
    if not temp_img_path or not os.path.exists(temp_img_path):
        return None
    group_clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in (group_id or "default"))
    target_dir = IMAGES_DIR / group_clean
    target_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(temp_img_path) as im:
        im_rgb = im.convert("RGB")
        w, h = im_rgb.size
        max_side = 1280
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            im_rgb = im_rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        digest = hashlib.md5(im_rgb.tobytes()[:4096]).hexdigest()[:10]
        out_path = target_dir / f"img_{digest}.jpg"
        im_rgb.save(out_path, format="JPEG", quality=85, optimize=True)

    return str(out_path.relative_to(BASE_DIR))


def get_record_choices() -> List[str]:
    recs = load_master_dataset()
    choices = []
    for r in recs:
        rid = r.get("id", "unknown")
        mode = "🙈盲写" if r.get("write_mode") == "blind_first" else "✏️改写"
        msgs = r.get("messages", [])
        asst = [m for m in msgs if m.get("role") == "assistant"]
        done = "✅" if (asst and all(bool((m.get("fish_approved") or "").strip()) for m in asst)) else "⏳"
        task_t = r.get("task_type", "")[:12]
        first_user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")[:22].replace("\n", " ")
        choices.append(f"{rid} | {done} {mode} ({len(asst)}轮·{task_t}) | {first_user}")
    return choices


def load_sample_to_ui(selected_label: Optional[str], reveal_qwen_override: bool) -> List[Any]:
    choices = get_record_choices()
    if not selected_label:
        selected_label = choices[0] if choices else "seed_001"
    selected_id = selected_label.split("|")[0].strip()

    recs = load_master_dataset()
    rec = next((r for r in recs if r.get("id") == selected_id), None)
    if not rec and recs:
        rec = recs[0]
        selected_id = rec.get("id", "seed_001")
    if not rec:
        rec = {
            "id": selected_id,
            "source_group_id": f"grp_{selected_id}",
            "split": "train",
            "write_mode": "blind_first",
            "task_type": TASK_TYPE_CHOICES[0],
            "is_fewshot": False,
            "images": [],
            "edit_tags": [],
            "edit_note": "",
            "messages": [{"role": "user", "content": ""}, {"role": "assistant", "qwen_original": "", "fish_approved": "", "turn_note": ""}],
        }

    img_rel = rec.get("images", [None])[0] if rec.get("images") else None
    img_abs = str(BASE_DIR / img_rel) if (img_rel and (BASE_DIR / img_rel).exists()) else None

    is_blind = rec.get("write_mode") == "blind_first"
    msgs = rec.get("messages", [])
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    u_curr = None
    for m in msgs:
        if m.get("role") == "user":
            u_curr = m
        elif m.get("role") == "assistant":
            pairs.append((u_curr or {"role": "user", "content": ""}, m))
            u_curr = None

    turn_outputs = []
    for i in range(MAX_TURNS):
        if i < len(pairs):
            u_m, a_m = pairs[i]
            u_txt = u_m.get("content", "")
            q_txt = a_m.get("qwen_original", "")
            f_txt = a_m.get("fish_approved", "")
            n_txt = a_m.get("turn_note", "")
            if is_blind and not f_txt.strip() and not reveal_qwen_override and q_txt.strip():
                q_display = "🙈 【盲写保护中】请先在右侧独立写出你的理想回复并点击保存，或勾选上方「揭晓 Qwen 原答」后再查看。"
            else:
                q_display = q_txt
            turn_outputs.extend([u_txt, q_display, f_txt, n_txt])
        else:
            turn_outputs.extend(["", "", "", ""])

    matched_choice = next((c for c in choices if c.startswith(f"{selected_id} |")), selected_label)
    return [
        gr.update(choices=choices, value=matched_choice),
        rec.get("id", ""),
        rec.get("source_group_id", ""),
        rec.get("split", "train"),
        rec.get("write_mode", "blind_first"),
        rec.get("task_type", TASK_TYPE_CHOICES[0]),
        bool(rec.get("is_fewshot", False)),
        img_abs,
        rec.get("edit_tags", []),
        rec.get("edit_note", ""),
        *turn_outputs,
    ]


def step_sample(current_label: Optional[str], delta: int, reveal_qwen_override: bool) -> List[Any]:
    choices = get_record_choices()
    if not choices:
        return load_sample_to_ui(None, reveal_qwen_override)
    curr_idx = 0
    if current_label:
        curr_id = current_label.split("|")[0].strip()
        for idx, c in enumerate(choices):
            if c.startswith(f"{curr_id} |"):
                curr_idx = idx
                break
    next_idx = (curr_idx + delta) % len(choices)
    return load_sample_to_ui(choices[next_idx], reveal_qwen_override)


def save_sample_from_ui(
    sample_id: str,
    group_id: str,
    split: str,
    write_mode: str,
    task_type: str,
    is_fewshot: bool,
    uploaded_img: Optional[str],
    edit_tags: List[str],
    edit_note: str,
    *turn_values: Any,
) -> Tuple[Any, str, str]:
    sample_id = (sample_id or "").strip() or f"seed_{random.randint(100, 999)}"
    group_id = (group_id or "").strip() or f"grp_{sample_id}"

    recs = load_master_dataset()
    existing = next((r for r in recs if r.get("id") == sample_id), None)
    old_pairs: List[Dict[str, Any]] = []
    if existing:
        old_pairs = [m for m in existing.get("messages", []) if m.get("role") == "assistant"]

    images_list = existing.get("images", []) if existing else []
    if uploaded_img:
        saved_rel = compress_and_save_image(uploaded_img, group_id)
        if saved_rel:
            images_list = [saved_rel]

    new_msgs = []
    for i in range(MAX_TURNS):
        u_txt = (turn_values[i * 4 + 0] or "").strip()
        q_txt = (turn_values[i * 4 + 1] or "").strip()
        f_txt = (turn_values[i * 4 + 2] or "").strip()
        n_txt = (turn_values[i * 4 + 3] or "").strip()

        if not u_txt and not f_txt and not q_txt:
            continue

        if q_txt.startswith("🙈 【盲写保护中】") and i < len(old_pairs):
            q_txt = old_pairs[i].get("qwen_original", "")

        new_msgs.append({"role": "user", "content": u_txt})
        new_msgs.append({
            "role": "assistant",
            "qwen_original": q_txt,
            "fish_approved": f_txt,
            "turn_note": n_txt,
        })

    new_rec = {
        "id": sample_id,
        "source_group_id": group_id,
        "split": split,
        "write_mode": write_mode,
        "task_type": task_type,
        "is_fewshot": bool(is_fewshot),
        "images": images_list,
        "edit_tags": list(edit_tags or []),
        "edit_note": (edit_note or "").strip(),
        "messages": new_msgs,
    }

    replaced = False
    for idx, r in enumerate(recs):
        if r.get("id") == sample_id:
            recs[idx] = new_rec
            replaced = True
            break
    if not replaced:
        recs.append(new_rec)

    save_master_dataset(recs)
    rep = audit_and_export(recs, export=True)
    choices = get_record_choices()
    target_choice = next((c for c in choices if c.startswith(f"{sample_id} |")), choices[0] if choices else None)

    total_toks = sum(estimate_tokens(m.get("fish_approved", "")) for m in new_msgs if m.get("role") == "assistant")
    num_turns = sum(1 for m in new_msgs if m.get("role") == "assistant")
    status_msg = f"✅ 已保存样本 `{sample_id}`（共 {num_turns} 轮，大肥鱼回复约 {total_toks} tokens），并自动更新 SFT/DPO 导出表！"
    return gr.update(choices=choices, value=target_choice), status_msg, format_report_markdown(rep)


def parse_human_blocks(raw_text: str, start_idx: int) -> List[Dict[str, Any]]:
    """
    解析易读的多行 ===SAMPLE=== 块格式，彻底避免网页端 AI 在单行 JSONL 中写长文/长代码时转义失败。
    支持格式:
    ===SAMPLE===
    task_type: ...
    images: ...
    edit_tags: 开头两句, 信息取舍
    edit_note: ...
    [USER]
    ...
    [FISH]
    ...
    [NOTE]
    ...
    """
    import re
    chunks = re.split(r"^===+SAMPLE[^=\n]*===+\s*$", raw_text.strip(), flags=re.M)
    results = []
    seq = start_idx
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or "[USER]" not in chunk or "[FISH]" not in chunk:
            continue
        # 提取头部 meta 与第一个 [USER] 之前的部分
        header_part, body_part = chunk.split("[USER]", 1)
        body_part = "[USER]" + body_part

        meta: Dict[str, str] = {}
        for line in header_part.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()

        sid = meta.get("id") or f"seed_{seq:03d}"
        seq += 1
        grp = meta.get("source_group_id") or f"grp_{sid}"
        task_t = meta.get("task_type") or "日常半截话快问快答"
        imgs = [x.strip() for x in re.split(r"[,，]", meta.get("images", "")) if x.strip() and x.strip().lower() != "none"]
        raw_tags = [x.strip() for x in re.split(r"[,，/]", meta.get("edit_tags", "")) if x.strip()]
        valid_tags = [t for t in raw_tags if t in EDIT_TAG_CHOICES] or ["开头两句", "信息取舍"]
        edit_note = meta.get("edit_note") or ""

        # 解析多轮 [USER] / [FISH] / [NOTE]
        tokens = re.split(r"^\[(USER|FISH|ASSISTANT|NOTE)\]\s*$", body_part, flags=re.M)
        msgs = []
        curr_user = None
        curr_fish = None
        curr_note = ""

        idx = 1
        while idx < len(tokens):
            tag = tokens[idx].strip().upper()
            content = tokens[idx + 1].strip() if (idx + 1) < len(tokens) else ""
            idx += 2

            if tag == "USER":
                if curr_user is not None and curr_fish is not None:
                    msgs.append({"role": "user", "content": curr_user})
                    msgs.append({"role": "assistant", "qwen_original": "", "fish_approved": curr_fish, "turn_note": curr_note})
                    curr_fish = None
                    curr_note = ""
                curr_user = content
            elif tag in ("FISH", "ASSISTANT"):
                curr_fish = content
            elif tag == "NOTE":
                curr_note = content

        if curr_user is not None and curr_fish is not None:
            msgs.append({"role": "user", "content": curr_user})
            msgs.append({"role": "assistant", "qwen_original": "", "fish_approved": curr_fish, "turn_note": curr_note})

        if msgs:
            results.append({
                "id": sid,
                "source_group_id": grp,
                "split": meta.get("split", "train"),
                "write_mode": meta.get("write_mode", "edit_after_qwen"),
                "task_type": task_t,
                "is_fewshot": False,
                "images": imgs,
                "edit_tags": valid_tags,
                "edit_note": edit_note,
                "messages": msgs,
            })
    return results


def bulk_import_jsonl(raw_text: str) -> Tuple[Any, str, str]:
    recs = load_master_dataset()
    existing_ids = {r.get("id") for r in recs}
    added = 0

    text_clean = (raw_text or "").strip()
    if "===SAMPLE" in text_clean or ("[USER]" in text_clean and "[FISH]" in text_clean):
        parsed_blocks = parse_human_blocks(text_clean, len(recs) + 1)
        for obj in parsed_blocks:
            oid = obj.get("id") or f"seed_{len(recs) + 1:03d}"
            while oid in existing_ids:
                oid = f"seed_{len(recs) + 1 + added:03d}"
            obj["id"] = oid
            obj["source_group_id"] = obj.get("source_group_id") or f"grp_{oid}"
            recs.append(obj)
            existing_ids.add(oid)
            added += 1
    else:
        for line in text_clean.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            oid = obj.get("id") or f"seed_{len(recs) + 1:03d}"
            if oid in existing_ids:
                oid = f"seed_{len(recs) + 1:03d}"
                obj["id"] = oid
            if not obj.get("source_group_id"):
                obj["source_group_id"] = f"grp_{oid}"
            recs.append(obj)
            existing_ids.add(oid)
            added += 1

    save_master_dataset(recs)
    rep = audit_and_export(recs, export=True)
    choices = get_record_choices()
    return (
        gr.update(choices=choices, value=choices[-1] if choices else None),
        f"✅ 成功批量导入 {added} 条样本（当前库中共 {len(recs)} 条）！",
        format_report_markdown(rep),
    )


def create_new_blank_sample() -> Tuple[str, str, str]:
    recs = load_master_dataset()
    next_num = len(recs) + 1
    sid = f"seed_{next_num:03d}"
    return sid, f"grp_{sid}", f"✨ 已准备新样本 ID `{sid}`，请在下方填写对话后点击「💾 保存当前样本」。"


def build_ui() -> gr.Blocks:
    init_choices = get_record_choices()
    init_val = init_choices[0] if init_choices else None

    with gr.Blocks(title="大肥鱼-9B (Base) 四栏数据工作台 & 盲测中心") as demo:
        gr.Markdown(
            """
            # 🐟 大肥鱼-9B（Base）通用图文表达底座 · 四栏可视化编辑工作台
            **当前状态**：已为你导入 `seed_001` ~ `seed_020` 共 **20 条**候选对话（含 10 组 3~4 轮多轮对话 + 4 条深度长答 + 4 条精确交付 + 看图纠错）。
            你可以直接点 **「⬅️ 上一条 / ➡️ 下一条」** 逐条可视化审阅修改，改完点左侧 **「💾 保存当前样本」** 即可！
            """
        )

        with gr.Tabs():
            # ==================== TAB 1: 四栏数据工作台 ====================
            with gr.Tab("📝 1. 四栏可视化编辑工作台"):
                with gr.Row():
                    btn_prev = gr.Button("⬅️ 上一条", scale=1)
                    sample_dropdown = gr.Dropdown(
                        choices=init_choices,
                        value=init_val,
                        label="选择样本（✅=已有大肥鱼回复，⏳=待写，🙈=盲写，✏️=改写）",
                        scale=5,
                    )
                    btn_next = gr.Button("➡️ 下一条", scale=1)
                    btn_new = gr.Button("➕ 新建样本", scale=1)
                    reveal_chk = gr.Checkbox(label="👁️ 揭晓盲写原答", value=False, scale=1)

                status_banner = gr.Markdown("✅ **已加载 `seed_001` ~ `seed_020`（共 20 条）**，可直接翻页修改。")

                with gr.Accordion("📥 批量粘贴 JSONL 一键追加导入（把其他网页 AI 生成的 JSONL 贴进来自动编号入库）", open=False):
                    bulk_box = gr.Textbox(label="粘贴 JSONL（每行一个 JSON）", lines=4, placeholder='{"id": "...", "messages": [...]}')
                    btn_bulk = gr.Button("📥 立即批量导入到工作台", variant="secondary")

                with gr.Row():
                    with gr.Column(scale=1):
                        btn_save = gr.Button("💾 保存当前样本修改", variant="primary")
                        inp_id = gr.Textbox(label="样本 ID", value="seed_001")
                        inp_group = gr.Textbox(label="防泄漏分组 (source_group_id，同图/同题变体填同一组)", value="grp_oom_backward")
                        inp_split = gr.Radio(choices=["train", "eval_sealed"], value="train", label="划分 (train=训练 / eval_sealed=封存测试)")
                        inp_mode = gr.Radio(
                            choices=["blind_first", "edit_after_qwen"],
                            value="blind_first",
                            label="模式 (blind_first=盲写 / edit_after_qwen=看原答改)",
                        )
                        inp_task = gr.Dropdown(choices=TASK_TYPE_CHOICES, value=TASK_TYPE_CHOICES[0], allow_custom_value=True, label="场景分类")
                        inp_fewshot = gr.Checkbox(label="⭐ 选入 B 组 Few-shot 精选示例池 (4~6 条)", value=True)
                        inp_img = gr.Image(type="filepath", label="附带图片（上传后自动压缩为 ~80KB 轻量 JPEG）")
                        inp_tags = gr.CheckboxGroup(choices=EDIT_TAG_CHOICES, label="📌 这次具体动了哪里？（统计数据配方）", value=["开头两句", "信息取舍", "中后轮防回弹"])
                        inp_note = gr.Textbox(label="整体改写备注 / 训练意图", lines=3)

                    with gr.Column(scale=3):
                        turn_Widgets: List[Any] = []
                        for t in range(1, MAX_TURNS + 1):
                            with gr.Accordion(f"第 {t} 轮对话 (Turn {t})", open=(t <= 4)):
                                u_box = gr.Textbox(label=f"[栏1] 第 {t} 轮用户输入", lines=2)
                                with gr.Row():
                                    q_box = gr.Textbox(label=f"[栏2] Qwen 原答 (可留空，稍后由 Colab Qwen3.5-9B 批量自动填入)", lines=4)
                                    f_box = gr.Textbox(label=f"[栏3] 🐟 你认可的大肥鱼回答 (直接在这里微调修改！)", lines=4)
                                n_box = gr.Textbox(label=f"[栏4] 第 {t} 轮修改点备注", lines=1)
                                turn_Widgets.extend([u_box, q_box, f_box, n_box])

            # ==================== TAB 2: 数据配方与四项偏差体检 ====================
            with gr.Tab("📊 2. 数据配方与「四项偏差」体检看板"):
                btn_refresh_audit = gr.Button("🔄 立即刷新四项偏差体检并导出 SFT / DPO 数据集", variant="primary")
                audit_md = gr.Markdown(format_report_markdown(audit_and_export(load_master_dataset(), export=True)))

            with gr.Tab("☁️ 3. Google Drive 同步"):
                gr.Markdown(
                    "可选功能。请先按 README 配置本地 Drive CLI 和目标目录。"
                )
                with gr.Row():
                    btn_drive_status = gr.Button("检查 Drive 连接")
                    btn_push_data = gr.Button("同步 JSONL 到 Drive", variant="primary")
                drive_log_box = gr.Markdown()

            # ==================== TAB 4: A/B/C 盲测打分面板 ====================
            with gr.Tab("⚖️ 4. A / B / C 盲测验收面板（大纲第 5 节）"):
                gr.Markdown(
                    """
                    ### 🙈 大纲第 5 节：B（原版 + 4~6 条精选示例） vs C（大肥鱼-Base 微调版）盲选面板
                    - 待第一轮训练 Checkpoint 产出并跑完 `eval_abc` 后，在此处对 40 道开放图文题与 10 组 5~8 轮长聊进行盲评打分。
                    """
                )

        all_form_outputs = [sample_dropdown, inp_id, inp_group, inp_split, inp_mode, inp_task, inp_fewshot, inp_img, inp_tags, inp_note, *turn_Widgets]

        # 页面打开时自动填充第一条样本
        demo.load(
            fn=load_sample_to_ui,
            inputs=[sample_dropdown, reveal_chk],
            outputs=all_form_outputs,
        )

        sample_dropdown.change(
            fn=load_sample_to_ui,
            inputs=[sample_dropdown, reveal_chk],
            outputs=all_form_outputs,
        )
        reveal_chk.change(
            fn=load_sample_to_ui,
            inputs=[sample_dropdown, reveal_chk],
            outputs=all_form_outputs,
        )
        btn_prev.click(
            fn=lambda cur, rev: step_sample(cur, -1, rev),
            inputs=[sample_dropdown, reveal_chk],
            outputs=all_form_outputs,
        )
        btn_next.click(
            fn=lambda cur, rev: step_sample(cur, 1, rev),
            inputs=[sample_dropdown, reveal_chk],
            outputs=all_form_outputs,
        )

        btn_new.click(
            fn=create_new_blank_sample,
            inputs=[],
            outputs=[inp_id, inp_group, status_banner],
        )

        btn_save.click(
            fn=save_sample_from_ui,
            inputs=[inp_id, inp_group, inp_split, inp_mode, inp_task, inp_fewshot, inp_img, inp_tags, inp_note, *turn_Widgets],
            outputs=[sample_dropdown, status_banner, audit_md],
        )

        btn_bulk.click(
            fn=bulk_import_jsonl,
            inputs=[bulk_box],
            outputs=[sample_dropdown, status_banner, audit_md],
        )

        btn_refresh_audit.click(
            fn=lambda: format_report_markdown(audit_and_export(load_master_dataset(), export=True)),
            inputs=[],
            outputs=[audit_md],
        )

        btn_drive_status.click(fn=status_all_accounts, inputs=[], outputs=[drive_log_box])
        btn_push_data.click(fn=sync_data_to_drive, inputs=[], outputs=[drive_log_box])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("FISH9B_PORT", "7860")), show_error=True)
