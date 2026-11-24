# 数据格式

## 工作台源记录

本地默认位置是 `data/workbench/dataset_master.jsonl`。每行一段对话，用户消息用 `content`，助手候选分别用 `qwen_original` 和 `fish_approved`。

```json
{"id":"demo_001","source_group_id":"demo_group_001","split":"train","write_mode":"rewrite","task_type":"日常问答","messages":[{"role":"user","content":"水在标准大气压下多少度沸腾？"},{"role":"assistant","qwen_original":"在标准大气压下，水的沸点为 100°C。","fish_approved":"100°C，前提是标准大气压。"}]}
```

相关样本沿用同一个 `source_group_id`，用 `split: "eval_sealed"` 标记封存评测。当前工作台支持本地图片引用，实际训练时检查所用训练入口是否支持图文格式。

## 文本 SFT 和逐轮采集

`colab/train_sft_v2.py` 与 `scripts/run_a100_base_rollout_550.py` 读取标准 `messages` 结构，文本消息按照 `user` / `assistant` 交替。SFT v2 直接读取文本 `content`，带图训练可使用 `colab/colab_runner.py` 的 ms-swift 路径。

```json
{"messages":[{"role":"user","content":"水在标准大气压下多少度沸腾？"},{"role":"assistant","content":"100°C，前提是标准大气压。"}]}
```

纯正文训练用 `--nothink`。保留思考的训练集每条助手消息使用闭合的 `<think>…</think>`，后面接正文。训练与验证分别传文件，采集后改写的内容按原始对话组重做分组检查。候选 DPO 对需先核验改写在相同上下文中优于原始回答。
