# NanoLlama Data Directory

This directory contains all dataset-related files for the NanoLlama-1B project.

## Directory Structure

```
data/
├── README.md                        # This file
├── download_and_verify.py          # Dataset download script
├── HOW_TO_DOWNLOAD.md              # Download guide
├── ALL_DATASETS_READY.md           # Dataset status summary
├── fineweb_edu_validation.txt      # FineWeb-Edu validation marker
├── cache/                           # Organized datasets (71.16 GB)
│   ├── core/                        # Core datasets (8.7 MB)
│   │   ├── tokenizer/              # Llama-3 tokenizer
│   │   └── fineweb_edu/            # FineWeb-Edu streaming
│   ├── pretrain/                    # Pre-training datasets (54 GB)
│   │   ├── pg19_books/             # PG19 books
│   │   ├── the_stack_code/         # The Stack (Python, JS, TS)
│   │   ├── openwebmath/            # Math content
│   │   └── stackexchange/          # StackExchange Q&A
│   ├── evaluation/                  # Evaluation benchmarks (229 MB)
│   │   ├── mmlu/                   # MMLU test
│   │   ├── gsm8k/                  # GSM8K math
│   │   ├── hellaswag/              # HellaSwag reasoning
│   │   ├── arc_challenge/          # ARC-Challenge
│   │   └── truthfulqa/             # TruthfulQA
│   ├── sft/                         # Supervised fine-tuning (160 MB)
│   │   ├── openassistant/          # OpenAssistant chat
│   │   └── metamathqa/             # MetaMathQA
│   └── dpo/                         # DPO alignment (343 MB)
│       ├── hh_rlhf/                # HH-RLHF preferences
│       └── orca_dpo/               # Orca DPO pairs
└── logs/                            # Download logs
    ├── dataset_download_20251205_231313.log
    └── ...
```

## Quick Start

Download all datasets:
```bash
cd data
python3 download_and_verify.py
```

## What's Here

### Scripts
- **download_and_verify.py** - Robust download script with timestamped logging and verification

### Documentation
- **HOW_TO_DOWNLOAD.md** - Complete guide to downloading datasets
- **ALL_DATASETS_READY.md** - Status of all 15 datasets (✅ Complete)

### Data
- **cache/** - Organized datasets cache (71.16 GB)
  - **core/** (8.7 MB) - Tokenizer + FineWeb-Edu
  - **pretrain/** (54 GB) - PG19, The Stack, OpenWebMath, StackExchange
  - **evaluation/** (229 MB) - MMLU, GSM8K, HellaSwag, ARC, TruthfulQA
  - **sft/** (160 MB) - OpenAssistant, MetaMathQA
  - **dpo/** (343 MB) - HH-RLHF, Orca-DPO
- **logs/** - Timestamped download logs
- **fineweb_edu_validation.txt** - Marker file confirming FineWeb-Edu setup

## Datasets Included (15/15)

### Core (2/2)
1. ✓ Tokenizer (Llama-3) - 128,256 vocab
2. ✓ FineWeb-Edu (Phase 1) - Streaming, 24B tokens

### Pre-training (4/4)
3. ✓ PG19 (Books - Phase 2) - 28,602 books
4. ✓ The Stack (Code - Phase 2) - Python, JavaScript, TypeScript
5. ✓ OpenWebMath (Phase 3) - Math content
6. ✓ StackExchange (Phase 3) - 10.8M examples

### Evaluation (5/5)
7. ✓ MMLU - 14,042 test examples
8. ✓ GSM8K - 1,319 test examples
9. ✓ HellaSwag - 10,042 validation examples
10. ✓ ARC-Challenge - 1,172 test examples
11. ✓ TruthfulQA - 817 validation examples

### Fine-tuning (4/4)
12. ✓ OpenAssistant (SFT) - 128,575 train examples
13. ✓ MetaMathQA (SFT) - 395,000 examples
14. ✓ HH-RLHF (DPO) - 160,800 train examples
15. ✓ Orca-DPO (DPO) - 12,859 train examples

## Disk Usage

Total: **71.16 GB** (organized by purpose)

**By Category:**
- Pre-training: 54 GB (76%)
  - The Stack: ~40 GB
  - StackExchange: ~15 GB
  - OpenWebMath: ~10 GB
  - PG19: ~11 GB
- DPO: 343 MB
- SFT: 160 MB
- Evaluation: 229 MB
- Core: 8.7 MB

## Status

✅ **All datasets downloaded and verified**
- Ready for Phase 1 training
- Ready for all evaluation benchmarks
- Ready for SFT and DPO fine-tuning
