# Fish-9B · 小鱼实验基础设施

Fish-9B 是一套用于 Qwen3.5-9B 对话数据整理、LoRA 微调与推理评测的实验工具。数据工作台在本地运行，模型权重、训练集、检查点和实验输出通过 Google Drive 在 Colab 会话之间保留。

## 本地数据工作台

需要 Python 3.10+。工作台不加载模型。

```bash
git clone https://github.com/Sadami3066/Fish-9B.git
cd Fish-9B
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app_workbench.py
```

打开 `http://127.0.0.1:7860`。工作台支持导入对话、并排编辑原答与改写、保存图片引用，以及审核并导出 SFT 和候选 DPO 数据。从空目录启动时，可以在界面新建样本。自定义端口用 `FISH9B_PORT`。

```bash
python scripts/audit_dataset.py --input data/workbench/dataset_master.jsonl --no-export
python scripts/audit_dataset.py --input data/workbench/dataset_master.jsonl --output-dir data/exports
```

工作台记录与 SFT 输入使用不同的 JSONL 结构，见 [数据格式](docs/data-format.md)。

## Colab 与 Google Drive

先在 Colab 挂载自己的 Drive，再克隆仓库。在 Colab 的 Python 单元运行：

```python
from google.colab import drive
drive.mount('/content/drive')
```

在运行训练或推理命令的同一个 shell 中设置 `FISH9B_DRIVE_ROOT`。默认值是 `/content/drive/MyDrive/Fish-9B`。在 Notebook 里也可以通过 `os.environ` 设置，使后续命令继承。

```text
Fish-9B/                    # Drive 项目目录
├── data/                   # 训练、验证和采集输入
├── model_cache/Qwen3.5-9B/ # 官方模型分片、配置和 tokenizer
├── checkpoints/            # LoRA adapter 与训练状态
└── outputs/                # 训练日志、采集结果和评测输出
```

首次缓存权重时，在 Colab 下载到 Drive，后续会话可以复用：

```bash
python -m pip install huggingface_hub
hf download Qwen/Qwen3.5-9B --local-dir "${FISH9B_DRIVE_ROOT:-/content/drive/MyDrive/Fish-9B}/model_cache/Qwen3.5-9B"
```

### SFT

在支持 BF16 的 GPU 上安装训练依赖，指定各自的训练集、验证集和输出目录。先跑 `--check_only` 校验数据与 assistant-only loss 掩码，再去掉它正式训练。

```bash
python -m pip install 'transformers>=5,<6' peft accelerate tensorboard
root=${FISH9B_DRIVE_ROOT:-/content/drive/MyDrive/Fish-9B}
python colab/train_sft_v2.py \
  --train_file "$root/data/train.jsonl" --val_file "$root/data/val.jsonl" \
  --output_dir "$root/checkpoints/my-run" --check_only
```

带 `<think>` 的文本集使用默认模板，纯正文集加 `--nothink`。可选 `--backend unsloth`（另装 `unsloth`）。恢复训练时指定完整的 `--resume_from_checkpoint`，并沿用同一组数据和超参数。

也可在本地主机用已认证的 [Colab CLI](https://github.com/googlecolab/google-colab-cli) 启动。若二进制不在 `PATH`，设置 `FISH9B_COLAB_BIN` 指向它。启动器会对它创建的会话执行自动停机，`--reuse` 适用于已经挂载 Drive 的现有会话。

```bash
python scripts/launch_sft_v2.py --session fish-sft --cands A100hm,A100 \
  --train_args='--lr 5e-5 --epochs 2'
```

### 推理与固定历史采集

在 GPU 会话安装与 CUDA 版本匹配的 [SGLang](https://docs.sglang.ai/) 后，在该会话的终端运行下列命令。端口默认是 `30000`，API 在会话内监听 `127.0.0.1`。

```bash
read -rs FISH9B_SGLANG_API_KEY
export FISH9B_SGLANG_API_KEY
bash colab/serve_sglang.sh
```

LoRA 对话台使用 `python colab/chat_xiaoyu.py --run_dir /path/to/run`（需要安装 `unsloth`、`gradio`）。工具响应由手工输入或内置模拟数据提供。

`scripts/run_a100_base_rollout_550.py` 从指定 JSONL 提取固定历史下的逐轮样本，直接启动 Transformers 或 Unsloth 模型。它适用于固定上下文对照，真实交互采集需要用户根据新回复继续对话。

```bash
root=${FISH9B_DRIVE_ROOT:-/content/drive/MyDrive/Fish-9B}
python scripts/run_a100_base_rollout_550.py --input /path/to/train.jsonl --check_only
python scripts/run_a100_base_rollout_550.py --input /path/to/train.jsonl \
  --backend transformers --batch_size 8 --limit 20 --out_dir "$root/outputs/rollout-smoke"
```

将同一输入与同一输出目录用于续跑。正式运行时移除 `--limit`；截断或正文为空的轮次会在后续运行中重试。固定历史脚本会单独加载模型，应在显存充足的会话里运行。

### 可选的本地 Drive 同步

`scripts/gdrive_bridge.py` 兼容具有 `readonly` 与 `mutate` 子命令的 AGY `gdrive` CLI。复制 `configs/drive_ids.example.json` 为 `configs/drive_ids.json`，填入自己的账号与目录 ID，再运行 `python scripts/gdrive_bridge.py status`。自定义 CLI 路径用 `FISH9B_GDRIVE_BIN`。也可以直接通过浏览器上传小型 JSONL。

## 仓库内容

- `app_workbench.py` 与 `scripts/audit_dataset.py` 提供编辑、检查与导出。
- `colab/` 包含 SFT、LoRA 对话调试与 SGLang 启动入口。
- `scripts/` 包含采集与 Colab 编排。`configs/` 提供提示词、评测配置和可选的 Drive 配置模板。
- 数据、权重、检查点、运行日志和账号配置留在 Git 之外。Colab 实例停机后需要重新安装运行时依赖。
