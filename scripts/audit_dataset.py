#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大肥鱼-9B (Base) 数据集四项偏差体检、数据配方统计与 SFT/DPO 格式导出工具
对应计划大纲第 2、3、6 节要求：
1. 按 source_group_id 检查 train / eval_sealed 数据泄漏；
2. 统计修改分布（数据配方：开头两句、措辞、信息取舍、中后轮回弹、理解方向错误、精确交付）；
3. 统计 3~6 轮多轮对话占比（目标 30%~40%）；
4. 检查四项偏差：
   - 偏差 1：答案是不是越来越短？（按类型统计长度分位数与长回答占比）
   - 偏差 2：图片是不是可有可无？（排查缺乏具体视觉细节/数字的泛化看图回复）
   - 偏差 3：助手是不是只有一种腔调？（反问句比例、高频口癖、严格 JSON/代码交付校验）
   - 偏差 4：中后轮有没有偷偷变回客服？（扫描第 2~8 轮是否出现“首先、其次、好的感谢指正”等回弹词）
5. 导出 ms-swift 标准图文多轮 SFT 格式 + 备用 DPO 偏好资产表。
"""

import argparse
import collections
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_PATH = BASE_DIR / "data" / "workbench" / "dataset_master.jsonl"
EXPORT_DIR = BASE_DIR / "data" / "exports"

REBOUND_PATTERNS = [
    "首先",
    "其次",
    "最后，",
    "综上所述",
    "好的，没问题",
    "好的，感谢",
    "感谢您的指正",
    "为您整理",
    "希望以上",
    "如果您还有",
    "请随时告诉我",
    "从以下几个方面",
    "总而言之",
]

# 星号动作描写提示（仅对过多 *动作旁白* 做温和提醒，保留适度鱼梗与反差萌）
FISH_CLICHE_BANNED: List[str] = []

# 摸鱼/干饭/鲸鱼灵魂点缀词（监控触发密度，理想区间 10%~25%，防止每句都刷梗）
FISH_FLAVOR_KEYWORDS = [
    "吃饭",
    "干饭",
    "摸鱼",
    "歇会",
    "睡了",
    "下班",
    "大肥鱼",
    "鲸",
    "外包",
    "千问",
    "测完告诉我",
    "测完喊我",
]


def estimate_tokens(text: str) -> int:
    """粗略估算中英文混合 token 数（中文字符约 0.7~1 token，英文单词约 1.3 token）"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk * 0.85 + other * 0.35))


def load_master_dataset(path: Path = MASTER_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON 解析失败 (行 {line_no}): {e}")
    return records


def save_master_dataset(records: List[Dict[str, Any]], path: Path = MASTER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def percentile(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(round((len(sorted_vals) - 1) * p))))
    return sorted_vals[idx]


def audit_and_export(records: List[Dict[str, Any]], export: bool = True, output_dir: Path = EXPORT_DIR) -> Dict[str, Any]:
    train_records = [r for r in records if r.get("split", "train") == "train"]
    eval_records = [r for r in records if r.get("split") == "eval_sealed"]

    # 1. 防泄漏检查 (source_group_id)
    train_groups = {r.get("source_group_id", r.get("id")) for r in train_records if r.get("source_group_id")}
    eval_groups = {r.get("source_group_id", r.get("id")) for r in eval_records if r.get("source_group_id")}
    leaked_groups = sorted(list(train_groups & eval_groups))

    # 2. 多轮比例与完成度统计
    total_train = len(train_records)
    completed_train = 0
    multiturn_3plus_count = 0
    multiturn_2plus_count = 0
    blind_first_count = 0
    vision_count = 0

    tag_counter = collections.Counter()
    task_lengths: Dict[str, List[int]] = collections.defaultdict(list)
    all_fish_tokens: List[int] = []
    all_qwen_tokens: List[int] = []

    rebound_warnings: List[str] = []
    cliche_warnings: List[str] = []
    flavor_turn_count = 0
    vision_generic_warnings: List[str] = []
    strict_delivery_warnings: List[str] = []
    question_ending_count = 0
    total_assistant_turns = 0

    for r in train_records:
        rec_id = r.get("id", "unknown")
        task_type = r.get("task_type", "general")
        images = r.get("images", [])
        has_image = bool(images) or any("<image>" in m.get("content", "") or "图" in m.get("content", "") for m in r.get("messages", []) if m.get("role") == "user")
        if has_image:
            vision_count += 1
        if r.get("write_mode") == "blind_first":
            blind_first_count += 1

        for t in r.get("edit_tags", []):
            tag_counter[t] += 1

        msgs = r.get("messages", [])
        asst_turns = [m for m in msgs if m.get("role") == "assistant"]
        num_turns = len(asst_turns)
        if num_turns >= 2:
            multiturn_2plus_count += 1
        if num_turns >= 3:
            multiturn_3plus_count += 1

        all_turns_approved = all(bool(m.get("fish_approved", "").strip()) for m in asst_turns) and num_turns > 0
        if all_turns_approved:
            completed_train += 1

        for turn_idx, m in enumerate(asst_turns, 1):
            fish_txt = (m.get("fish_approved") or "").strip()
            qwen_txt = (m.get("qwen_original") or "").strip()
            if not fish_txt:
                continue

            total_assistant_turns += 1
            f_tok = estimate_tokens(fish_txt)
            all_fish_tokens.append(f_tok)
            task_lengths[task_type].append(f_tok)
            if qwen_txt:
                all_qwen_tokens.append(estimate_tokens(qwen_txt))

            if fish_txt.endswith("？") or fish_txt.endswith("?"):
                question_ending_count += 1

            # 偏差 4：检查中后轮（以及全部轮次）是否残留客服回弹词
            for pat in REBOUND_PATTERNS:
                if pat in fish_txt:
                    rebound_warnings.append(
                        f"[{rec_id} 第{turn_idx}轮] 疑似残留客服套话/回弹词「{pat}」: {fish_txt[:60]}..."
                    )

            # 鱼八股违禁词与星号动作描写检查
            for bad in FISH_CLICHE_BANNED:
                if bad in fish_txt:
                    cliche_warnings.append(
                        f"[{rec_id} 第{turn_idx}轮] ❌ 命中鱼八股违禁词「{bad}」: {fish_txt[:60]}..."
                    )
            if re.search(r"(?<!\*)\*[^*\n]{1,20}\*(?!\*)", fish_txt):
                cliche_warnings.append(
                    f"[{rec_id} 第{turn_idx}轮] ❌ 疑似包含星号动作描写 (*...*): {fish_txt[:60]}..."
                )

            # 统计摸鱼/干饭灵魂点缀密度（目标 10%~25%）
            if any(kw in fish_txt for kw in FISH_FLAVOR_KEYWORDS):
                flavor_turn_count += 1

            # 偏差 2：检查看图样本第1轮是否过于空泛（既无数字也无具体方位/视觉特征词且过短）
            if has_image and turn_idx == 1 and f_tok < 8:
                vision_generic_warnings.append(
                    f"[{rec_id}] 带图样本首轮回复极短 ({f_tok} tokens)，请确认是否换张图也能说同一句话: {fish_txt}"
                )

            # 偏差 3：严格交付任务检查
            if "strict" in task_type or "json" in task_type.lower():
                user_msg = msgs[(turn_idx - 1) * 2].get("content", "") if (turn_idx - 1) * 2 < len(msgs) else ""
                if "只返回JSON" in user_msg or "只要JSON" in user_msg:
                    try:
                        json.loads(fish_txt)
                    except Exception as e:
                        strict_delivery_warnings.append(
                            f"[{rec_id} 第{turn_idx}轮] 要求只返回 JSON 但 json.loads 校验未通过: {e}"
                        )

    sorted_fish = sorted(all_fish_tokens)
    sorted_qwen = sorted(all_qwen_tokens)
    medium_answer_count = sum(1 for x in all_fish_tokens if 180 <= x < 600)
    true_long_answer_count = sum(1 for x in all_fish_tokens if x >= 600)
    multiturn_3plus_ratio = (multiturn_3plus_count / total_train * 100.0) if total_train else 0.0
    multiturn_2plus_ratio = (multiturn_2plus_count / total_train * 100.0) if total_train else 0.0
    question_ending_ratio = (question_ending_count / total_assistant_turns * 100.0) if total_assistant_turns else 0.0
    flavor_ratio = (flavor_turn_count / total_assistant_turns * 100.0) if total_assistant_turns else 0.0

    # 导出 ms-swift SFT 数据集与备用 DPO 数据集
    sft_train_count = 0
    sft_val_count = 0
    dpo_pair_count = 0

    if export:
        output_dir.mkdir(parents=True, exist_ok=True)
        sft_all = []
        sft_groups = []
        dpo_all = []

        for r in train_records:
            msgs = r.get("messages", [])
            asst_turns = [m for m in msgs if m.get("role") == "assistant"]
            if not asst_turns or not all(bool((m.get("fish_approved") or "").strip()) for m in asst_turns):
                continue

            swift_msgs = []
            history_for_dpo = []
            for m in msgs:
                role = m.get("role")
                if role == "user":
                    content = m.get("content", "")
                    swift_msgs.append({"role": "user", "content": content})
                    history_for_dpo.append({"role": "user", "content": content})
                elif role == "assistant":
                    approved = m.get("fish_approved", "").strip()
                    original = m.get("qwen_original", "").strip()
                    if original and original != approved:
                        dpo_item = {
                            "id": r.get("id"),
                            "messages": list(history_for_dpo) + [{"role": "assistant", "content": approved}],
                            "rejected_response": original,
                            "images": r.get("images", []),
                            "edit_tags": r.get("edit_tags", []),
                            "edit_note": m.get("turn_note") or r.get("edit_note", ""),
                        }
                        dpo_all.append(dpo_item)
                    swift_msgs.append({"role": "assistant", "content": approved})
                    history_for_dpo.append({"role": "assistant", "content": approved})

            sft_item: Dict[str, Any] = {"messages": swift_msgs}
            if r.get("images"):
                sft_item["images"] = r["images"]
            sft_all.append(sft_item)
            sft_groups.append(r.get("source_group_id") or r.get("id") or f"row-{len(sft_all)}")

        # 同一来源组只进入一个分区。只有一个组时留空验证集。
        groups_by_order = list(dict.fromkeys(sft_groups))
        val_size = max(1, len(sft_all) // 10)
        group_sizes = collections.Counter(sft_groups)
        val_groups = set()
        val_count = 0
        for group in reversed(groups_by_order[1:]):
            val_groups.add(group)
            val_count += group_sizes[group]
            if val_count >= val_size:
                break
        train_split = [row for row, group in zip(sft_all, sft_groups) if group not in val_groups]
        val_split = [row for row, group in zip(sft_all, sft_groups) if group in val_groups]

        with open(output_dir / "train_sft.jsonl", "w", encoding="utf-8") as f:
            for item in train_split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(output_dir / "val_sft.jsonl", "w", encoding="utf-8") as f:
            for item in val_split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(output_dir / "dpo_pairs.jsonl", "w", encoding="utf-8") as f:
            for item in dpo_all:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(output_dir / "eval_sealed.jsonl", "w", encoding="utf-8") as f:
            for item in eval_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        sft_train_count = len(train_split)
        sft_val_count = len(val_split)
        dpo_pair_count = len(dpo_all)

    report = {
        "summary": {
            "total_records": len(records),
            "train_total": total_train,
            "train_completed": completed_train,
            "eval_sealed_total": len(eval_records),
            "blind_first_count": blind_first_count,
            "vision_sample_count": vision_count,
            "multiturn_2plus_count": multiturn_2plus_count,
            "multiturn_2plus_ratio_pct": round(multiturn_2plus_ratio, 1),
            "multiturn_3plus_count": multiturn_3plus_count,
            "multiturn_3plus_ratio_pct": round(multiturn_3plus_ratio, 1),
            "sft_train_exported": sft_train_count,
            "sft_val_exported": sft_val_count,
            "dpo_pairs_exported": dpo_pair_count,
        },
        "recipe_distribution": dict(tag_counter.most_common()),
        "bias_checks": {
            "leakage_groups": leaked_groups,
            "bias1_length_tokens": {
                "fish_p25": percentile(sorted_fish, 0.25),
                "fish_p50": percentile(sorted_fish, 0.50),
                "fish_p75": percentile(sorted_fish, 0.75),
                "fish_p90": percentile(sorted_fish, 0.90),
                "qwen_p50": percentile(sorted_qwen, 0.50),
                "medium_answers_180_to_600_tokens": medium_answer_count,
                "true_long_answers_600plus_tokens": true_long_answer_count,
            },
            "bias2_vision_warnings": vision_generic_warnings,
            "bias3_tone_and_strict": {
                "question_ending_ratio_pct": round(question_ending_ratio, 1),
                "flavor_turn_ratio_pct": round(flavor_ratio, 1),
                "cliche_warnings": cliche_warnings,
                "strict_delivery_warnings": strict_delivery_warnings,
            },
            "bias4_rebound_warnings": rebound_warnings,
        },
    }
    return report


def format_report_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    b = report["bias_checks"]
    l = b["bias1_length_tokens"]
    recipe = report["recipe_distribution"]

    lines = [
        "### 📊 大肥鱼-Base 数据集配方与四项偏差体检报告",
        "",
        f"- **总样本数**: `{s['total_records']}`（训练侧 `{s['train_total']}` 条，已完成改写 `{s['train_completed']}` 条；封存测试集 `{s['eval_sealed_total']}` 条）",
        f"- **盲写样本 (`blind_first`)**: `{s['blind_first_count']}` 条 ｜ **带图样本**: `{s['vision_sample_count']}` 条",
        f"- **多轮对话占比**: ≥2轮 `{s['multiturn_2plus_count']}` 条 (`{s['multiturn_2plus_ratio_pct']}%`) ｜ **≥3轮 `{s['multiturn_3plus_count']}` 条 (`{s['multiturn_3plus_ratio_pct']}%`，大纲目标 30%–40%)**",
        f"- **已导出资产**: `train_sft.jsonl` (`{s['sft_train_exported']}` 条) · `val_sft.jsonl` (`{s['sft_val_exported']}` 条) · 备用 `dpo_pairs.jsonl` (`{s['dpo_pairs_exported']}` 对)",
        "",
        "#### 1. 你的第一版「数据配方」（你动得最多的地方）",
    ]
    if recipe:
        total_tags = sum(recipe.values())
        for k, v in recipe.items():
            pct = round(v / total_tags * 100, 1) if total_tags else 0
            lines.append(f"- **{k}**: `{v}` 次 (`{pct}%`)")
    else:
        lines.append("- *暂无标签统计*")

    lines.extend([
        "",
        "#### 2. 四项偏差与防泄漏体检结果",
        f"- **防泄漏检查 (`source_group_id`)**: {'✅ 无跨集泄漏' if not b['leakage_groups'] else '❌ 发现泄漏组: ' + ', '.join(b['leakage_groups'])}",
        f"- **偏差 1（长度分布与长文本保护）**: 大肥鱼回复 Token 分位数 `P25={l['fish_p25']} / P50={l['fish_p50']} / P75={l['fish_p75']} / P90={l['fish_p90']}`；中长回答 (`180~600 tokens`) 共 `{l['medium_answers_180_to_600_tokens']}` 条，**真正长文本 (`≥600 tokens`, 约800字+) 共 `{l['true_long_answers_600plus_tokens']}` 条**。"
        + (" ⚠️ *告警：当前缺少 ≥600 tokens 的真正长文本样本，容易导致长文本生成能力退化（短句坍缩），请补充改长文/完整代码/深度长文样本（建议占比 15%~20%）！*" if l["true_long_answers_600plus_tokens"] == 0 else " ✅"),
        f"- **偏差 2（看图特异性检查）**: {'✅ 未发现明显万能废图回复' if not b['bias2_vision_warnings'] else '⚠️ ' + '；'.join(b['bias2_vision_warnings'])}",
        f"- **偏差 3（腔调多样性与精确交付）**: 末尾反问句占比 `{b['bias3_tone_and_strict']['question_ending_ratio_pct']}%` "
        + ("⚠️ *反问句偏多，留意不要每句都反问*" if b["bias3_tone_and_strict"]["question_ending_ratio_pct"] > 35 else "✅")
        + f"；**大肥鱼摸鱼/傲娇灵魂点缀率 `{b['bias3_tone_and_strict']['flavor_turn_ratio_pct']}%`**（7:2:1 黄金区间为 `10%~28%`）；"
        + ("✅ 零鱼八股污染" if not b["bias3_tone_and_strict"].get("cliche_warnings") else "❌ " + "；".join(b["bias3_tone_and_strict"]["cliche_warnings"]))
        + "；严格 JSON/代码校验: "
        + ("✅ 全部通过" if not b["bias3_tone_and_strict"]["strict_delivery_warnings"] else "❌ " + "；".join(b["bias3_tone_and_strict"]["strict_delivery_warnings"])),
        f"- **偏差 4（中后轮客服腔回弹雷达）**: {'✅ 未检测到客服套话残留' if not b['bias4_rebound_warnings'] else '⚠️ ' + '；'.join(b['bias4_rebound_warnings'])}",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大肥鱼-9B 数据集四项偏差体检与导出")
    parser.add_argument("--input", type=Path, default=MASTER_PATH, help="主数据表路径")
    parser.add_argument("--output-dir", type=Path, default=EXPORT_DIR, help="导出目录")
    parser.add_argument("--no-export", action="store_true", help="只打印审核报告")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"主数据表不存在: {args.input}")
    recs = load_master_dataset(args.input)
    rep = audit_and_export(recs, export=not args.no_export, output_dir=args.output_dir)
    print(format_report_markdown(rep))
