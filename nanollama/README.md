# NanoLlama-1B: 1.1B Parameter Language Model

Building a high-quality 1B parameter model with only 30B training tokens to prove that **curriculum design beats scale**.

## Quick Start

### 1. Environment Setup

```bash
# Navigate to project directory
cd nanollama

# Run setup script (creates conda env, installs dependencies)
./setup.sh

# Or manually:
conda create -n nanollama python=3.11 -y
conda activate nanollama
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### 2. Authentication

```bash
# HuggingFace (for Llama-3 tokenizer access)
huggingface-cli login
# Get token from: https://huggingface.co/settings/tokens

# Weights & Biases (for training monitoring)
wandb login
# Get token from: https://wandb.ai/authorize
```

### 3. Download Datasets

This will take **several hours** and require **~100-200GB disk space**.

```bash
# Download all datasets (recommended - run in background)
nohup python scripts/download_datasets.py --all > logs/download.log 2>&1 &

# Or download selectively:
python scripts/download_datasets.py --pretrain-only  # Just pre-training data
python scripts/download_datasets.py --eval-only      # Just evaluation benchmarks
python scripts/download_datasets.py --sft-only       # Just SFT data
python scripts/download_datasets.py --tokenizer-only # Just tokenizer

# Monitor progress:
tail -f logs/download.log
```

### 4. Verify Downloads

```bash
# Check what was downloaded
python scripts/download_datasets.py --all  # Will verify existing downloads

# Check cache size
du -sh data/cache
```

## What Gets Downloaded

### Pre-training Data (~30B tokens target)

**Phase 1 (0-80%): General Knowledge - 24B tokens**
- FineWeb-Edu (educational web content, score ≥ 3)
- Streamed during training, not fully downloaded upfront

**Phase 2 (80-90%): Long Context - 3B tokens**
- PG19 (books for narrative coherence)
- The Stack (Python, JavaScript, TypeScript code)

**Phase 3 (90-100%): Knowledge Crystallization - 3B tokens**
- OpenWebMath (mathematical content)
- StackExchange (curated Q&A)
- Wikipedia STEM articles (prepared separately)

### Evaluation Benchmarks
- MMLU (Massive Multitask Language Understanding)
- GSM8K (Grade School Math)
- HellaSwag (Commonsense Reasoning)
- ARC-Challenge (Science Q&A)
- TruthfulQA (Truthfulness)

### Fine-tuning Data

**SFT (Supervised Fine-Tuning) - 100K examples**
- OpenAssistant/oasst2 (50K chat examples)
- MetaMathQA (25K math with chain-of-thought)
- Gorilla APIBench (25K tool use examples)

**DPO (Direct Preference Optimization) - 50K pairs**
- Anthropic HH-RLHF (preference pairs)
- Intel Orca DPO (alternative)

## Expected Download Times

With good internet connection (~100 Mbps):

| Dataset | Size | Time |
|---------|------|------|
| Tokenizer | ~5 MB | < 1 min |
| FineWeb-Edu setup | Minimal | < 5 min |
| PG19 | ~11 GB | 15-30 min |
| The Stack (Python/JS/TS) | ~50 GB | 1-2 hours |
| OpenWebMath | ~30 GB | 30-60 min |
| StackExchange | ~5 GB | 10-20 min |
| Evaluation datasets | ~5 GB | 10-20 min |
| SFT datasets | ~10 GB | 15-30 min |
| DPO datasets | ~2 GB | 5-10 min |
| **Total** | **~110 GB** | **3-5 hours** |

## Troubleshooting

### HuggingFace Authentication Error

```bash
# Error: "Repo requires authentication"
# Solution: Login with token
huggingface-cli login

# For Llama-3 tokenizer, you may need to accept Meta's license:
# Visit: https://huggingface.co/meta-llama/Meta-Llama-3-8B
```

### Out of Disk Space

```bash
# Check available space
df -h

# Delete cache of unused datasets
rm -rf data/cache/*

# Use external drive for cache
export HF_HOME=/path/to/external/drive/hf_cache
python scripts/download_datasets.py --all
```

### Slow Downloads

```bash
# Use mirrors (if outside US)
export HF_ENDPOINT=https://hf-mirror.com

# Download in smaller batches
python scripts/download_datasets.py --eval-only      # Fast
python scripts/download_datasets.py --sft-only       # Fast
python scripts/download_datasets.py --pretrain-only  # Slow
```

### Failed Download Mid-Way

```bash
# Re-run the script - it will resume from cache
python scripts/download_datasets.py --all

# HuggingFace datasets library caches partial downloads
# and will resume automatically
```

## Next Steps

After datasets are downloaded:

1. **Decontamination** (Day 0 of timeline)
   - Run `scripts/decontaminate.py` to remove test set leakage

2. **Model Implementation** (Day -2 of timeline)
   - Implement `models/nanollama.py`

3. **Training Pipeline** (Day -1 of timeline)
   - Implement `train.py`, data loaders, schedulers

4. **Start Training** (Day 1)
   - Launch Phase 1 pre-training on 2× H100s

## Directory Structure

```
nanollama/
├── data/
│   ├── cache/              # HuggingFace datasets cache
│   └── contaminated_ids.json  # (Created during decontamination)
├── models/
│   ├── tokenizer/          # Llama-3 tokenizer
│   └── nanollama.py        # (To be implemented)
├── checkpoints/            # Training checkpoints
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── sft/
│   └── dpo/
├── configs/                # Training configs (YAML)
├── scripts/
│   └── download_datasets.py
├── evals/                  # Evaluation scripts
├── logs/                   # Training and download logs
└── results/                # Benchmark results
```

## Hardware Requirements

- **GPUs**: 2× NVIDIA H100 80GB (as specified)
- **RAM**: 128GB+ recommended
- **Disk**: 500GB+ (200GB datasets + 300GB checkpoints)
- **Network**: Fast connection for downloads

## Training Timeline

- **Week 0 (Days -3 to 0)**: Setup + Dataset Download ← **YOU ARE HERE**
- **Week 1-3 (Days 1-21)**: Pre-training (Phase 1, 2, 3)
- **Week 4 (Days 22-27)**: SFT, DPO, Evaluation, Deployment

See `Implementation Timeline - 3 Week Plan` for detailed schedule.

## Support

For issues:
1. Check `logs/dataset_download.log`
2. Verify HuggingFace authentication
3. Check disk space
4. Consult HuggingFace datasets documentation

---

**Status**: Dataset download in progress...
**Next**: Once complete, proceed to decontamination and model implementation
