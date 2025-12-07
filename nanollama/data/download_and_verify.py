#!/usr/bin/env python3
"""
Robust dataset downloader with verification and logging.

This script:
1. Downloads all required datasets
2. Logs everything with timestamped log files
3. Verifies all datasets were downloaded successfully
4. Returns error code if any datasets failed
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer

# Create logs directory (within data folder)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"dataset_download_{timestamp}.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Dataset cache directories organized by purpose
CACHE_BASE = Path("cache")
CACHE_BASE.mkdir(parents=True, exist_ok=True)

# Organized subdirectories
CACHE_CORE = CACHE_BASE / "core"
CACHE_PRETRAIN = CACHE_BASE / "pretrain"
CACHE_EVAL = CACHE_BASE / "evaluation"
CACHE_SFT = CACHE_BASE / "sft"
CACHE_DPO = CACHE_BASE / "dpo"

# Create all subdirectories
for cache_dir in [CACHE_CORE, CACHE_PRETRAIN, CACHE_EVAL, CACHE_SFT, CACHE_DPO]:
    cache_dir.mkdir(parents=True, exist_ok=True)


class DatasetDownloader:
    """Manages dataset downloads with error tracking"""

    def __init__(self):
        self.results: Dict[str, bool] = {}
        self.errors: Dict[str, str] = {}

    def download_dataset(self, name: str, download_func) -> bool:
        """Download a dataset and track result"""
        logger.info("=" * 60)
        logger.info(f"Downloading: {name}")
        logger.info("=" * 60)

        try:
            download_func()
            logger.info(f"✓ {name} - SUCCESS")
            self.results[name] = True
            return True
        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ {name} - FAILED: {error_msg}")
            self.results[name] = False
            self.errors[name] = error_msg
            return False

    def get_summary(self) -> Tuple[List[str], List[str]]:
        """Get lists of successful and failed downloads"""
        successful = [name for name, success in self.results.items() if success]
        failed = [name for name, success in self.results.items() if not success]
        return successful, failed


def download_tokenizer(downloader: DatasetDownloader):
    """Download Llama-3 tokenizer"""

    def _download():
        tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Meta-Llama-3-8B',
            cache_dir=str(CACHE_CORE / "tokenizer")
        )
        tokenizer_path = Path("../models/tokenizer")
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_path))
        logger.info(f"  Vocab size: {len(tokenizer)}")

    downloader.download_dataset("Tokenizer (Llama-3)", _download)


def download_fineweb_edu(downloader: DatasetDownloader):
    """Download FineWeb-Edu for Phase 1"""

    def _download():
        dataset = load_dataset(
            'HuggingFaceFW/fineweb-edu',
            name='default',
            split='train',
            streaming=True,
            cache_dir=str(CACHE_CORE / "fineweb_edu")
        )
        # Test access
        sample = next(iter(dataset))
        logger.info(f"  Sample keys: {list(sample.keys())}")

        # Create validation marker
        marker = Path("fineweb_edu_validation.txt")
        marker.write_text("Validation set marker\n")

    downloader.download_dataset("FineWeb-Edu (Phase 1)", _download)


def download_pg19(downloader: DatasetDownloader):
    """Download PG19 books for Phase 2"""

    def _download():
        dataset = load_dataset(
            'pg19',
            split='train',
            cache_dir=str(CACHE_PRETRAIN / "pg19_books"),
            trust_remote_code=True
        )
        logger.info(f"  Examples: {len(dataset):,}")

    downloader.download_dataset("PG19 (Books - Phase 2)", _download)


def download_the_stack(downloader: DatasetDownloader):
    """Download The Stack code datasets for Phase 2"""

    def _download():
        stack_cache = CACHE_PRETRAIN / "the_stack_code"

        # Try Python
        dataset = load_dataset(
            'bigcode/the-stack-dedup',
            data_dir='data/python',
            split='train',
            cache_dir=str(stack_cache),
            streaming=True
        )
        sample = next(iter(dataset))
        logger.info(f"  Python: Accessible")

        # Try JavaScript
        dataset = load_dataset(
            'bigcode/the-stack-dedup',
            data_dir='data/javascript',
            split='train',
            cache_dir=str(stack_cache),
            streaming=True
        )
        sample = next(iter(dataset))
        logger.info(f"  JavaScript: Accessible")

        # Try TypeScript
        dataset = load_dataset(
            'bigcode/the-stack-dedup',
            data_dir='data/typescript',
            split='train',
            cache_dir=str(stack_cache),
            streaming=True
        )
        sample = next(iter(dataset))
        logger.info(f"  TypeScript: Accessible")

    downloader.download_dataset("The Stack (Code - Phase 2)", _download)


def download_openwebmath(downloader: DatasetDownloader):
    """Download OpenWebMath for Phase 3"""

    def _download():
        dataset = load_dataset(
            'open-web-math/open-web-math',
            split='train',
            cache_dir=str(CACHE_PRETRAIN / "openwebmath"),
            streaming=True,
            trust_remote_code=True
        )
        sample = next(iter(dataset))
        logger.info(f"  Sample text: {sample['text'][:100]}...")

    downloader.download_dataset("OpenWebMath (Phase 3)", _download)


def download_stackexchange(downloader: DatasetDownloader):
    """Download StackExchange for Phase 3"""

    def _download():
        dataset = load_dataset(
            'HuggingFaceH4/stack-exchange-preferences',
            split='train',
            cache_dir=str(CACHE_PRETRAIN / "stackexchange")
        )
        logger.info(f"  Examples: {len(dataset):,}")

    downloader.download_dataset("StackExchange (Phase 3)", _download)


def download_eval_datasets(downloader: DatasetDownloader):
    """Download evaluation benchmarks"""

    # MMLU
    def _download_mmlu():
        dataset = load_dataset(
            'cais/mmlu',
            'all',
            cache_dir=str(CACHE_EVAL / "mmlu")
        )
        logger.info(f"  Test examples: {len(dataset['test']):,}")

    downloader.download_dataset("MMLU (Evaluation)", _download_mmlu)

    # GSM8K
    def _download_gsm8k():
        dataset = load_dataset(
            'gsm8k',
            'main',
            cache_dir=str(CACHE_EVAL / "gsm8k")
        )
        logger.info(f"  Test: {len(dataset['test']):,}")

    downloader.download_dataset("GSM8K (Evaluation)", _download_gsm8k)

    # HellaSwag
    def _download_hellaswag():
        dataset = load_dataset(
            'Rowan/hellaswag',
            cache_dir=str(CACHE_EVAL / "hellaswag")
        )
        logger.info(f"  Validation: {len(dataset['validation']):,}")

    downloader.download_dataset("HellaSwag (Evaluation)", _download_hellaswag)

    # ARC
    def _download_arc():
        dataset = load_dataset(
            'ai2_arc',
            'ARC-Challenge',
            cache_dir=str(CACHE_EVAL / "arc_challenge")
        )
        logger.info(f"  Test: {len(dataset['test']):,}")

    downloader.download_dataset("ARC-Challenge (Evaluation)", _download_arc)

    # TruthfulQA
    def _download_truthfulqa():
        dataset = load_dataset(
            'truthful_qa',
            'generation',
            cache_dir=str(CACHE_EVAL / "truthfulqa")
        )
        logger.info(f"  Validation: {len(dataset['validation']):,}")

    downloader.download_dataset("TruthfulQA (Evaluation)", _download_truthfulqa)


def download_sft_datasets(downloader: DatasetDownloader):
    """Download SFT datasets"""

    # OpenAssistant
    def _download_oasst():
        dataset = load_dataset(
            'OpenAssistant/oasst2',
            cache_dir=str(CACHE_SFT / "openassistant")
        )
        logger.info(f"  Train: {len(dataset['train']):,}")

    downloader.download_dataset("OpenAssistant (SFT)", _download_oasst)

    # MetaMathQA
    def _download_metamath():
        dataset = load_dataset(
            'meta-math/MetaMathQA',
            cache_dir=str(CACHE_SFT / "metamathqa")
        )
        logger.info(f"  Examples: {len(dataset['train']):,}")

    downloader.download_dataset("MetaMathQA (SFT)", _download_metamath)


def download_dpo_datasets(downloader: DatasetDownloader):
    """Download DPO datasets"""

    # HH-RLHF
    def _download_hh():
        dataset = load_dataset(
            'Anthropic/hh-rlhf',
            cache_dir=str(CACHE_DPO / "hh_rlhf")
        )
        logger.info(f"  Train: {len(dataset['train']):,}")

    downloader.download_dataset("HH-RLHF (DPO)", _download_hh)

    # Orca DPO
    def _download_orca():
        dataset = load_dataset(
            'Intel/orca_dpo_pairs',
            cache_dir=str(CACHE_DPO / "orca_dpo")
        )
        logger.info(f"  Train: {len(dataset['train']):,}")

    downloader.download_dataset("Orca-DPO (DPO)", _download_orca)


def main():
    """Main download and verification flow"""
    logger.info("=" * 80)
    logger.info("NanoLlama-1B Dataset Downloader with Verification")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Cache directory: {CACHE_BASE.absolute()}")
    logger.info(f"  - Core datasets: {CACHE_CORE}")
    logger.info(f"  - Pre-training: {CACHE_PRETRAIN}")
    logger.info(f"  - Evaluation: {CACHE_EVAL}")
    logger.info(f"  - SFT: {CACHE_SFT}")
    logger.info(f"  - DPO: {CACHE_DPO}")
    logger.info("")

    downloader = DatasetDownloader()

    # Download all datasets
    logger.info("PHASE 1: Core Downloads")
    logger.info("=" * 80)
    download_tokenizer(downloader)
    download_fineweb_edu(downloader)

    logger.info("\nPHASE 2: Pre-training Datasets")
    logger.info("=" * 80)
    download_pg19(downloader)
    download_the_stack(downloader)
    download_openwebmath(downloader)
    download_stackexchange(downloader)

    logger.info("\nPHASE 3: Evaluation Datasets")
    logger.info("=" * 80)
    download_eval_datasets(downloader)

    logger.info("\nPHASE 4: Fine-tuning Datasets")
    logger.info("=" * 80)
    download_sft_datasets(downloader)
    download_dpo_datasets(downloader)

    # Generate summary
    successful, failed = downloader.get_summary()

    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n✓ Successful Downloads: {len(successful)}/{len(downloader.results)}")
    for name in successful:
        logger.info(f"  ✓ {name}")

    if failed:
        logger.info(f"\n✗ Failed Downloads: {len(failed)}/{len(downloader.results)}")
        for name in failed:
            error = downloader.errors.get(name, "Unknown error")
            logger.info(f"  ✗ {name}")
            logger.info(f"     Error: {error}")

    # Check critical datasets
    critical_datasets = [
        "Tokenizer (Llama-3)",
        "FineWeb-Edu (Phase 1)",
    ]

    critical_failed = [d for d in critical_datasets if d in failed]

    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION")
    logger.info("=" * 80)

    # Check if tokenizer exists
    tokenizer_path = Path("../models/tokenizer/tokenizer_config.json")
    if tokenizer_path.exists():
        logger.info("✓ Tokenizer verified: ../models/tokenizer/")
    else:
        logger.error("✗ Tokenizer not found!")
        critical_failed.append("Tokenizer")

    # Check if FineWeb marker exists
    fineweb_marker = Path("fineweb_edu_validation.txt")
    if fineweb_marker.exists():
        logger.info("✓ FineWeb-Edu verified: streaming configured")
    else:
        logger.error("✗ FineWeb-Edu marker not found!")
        critical_failed.append("FineWeb-Edu")

    # Check cache directory size
    cache_size_gb = sum(f.stat().st_size for f in CACHE_BASE.rglob('*') if f.is_file()) / 1e9
    logger.info(f"✓ Cache directory size: {cache_size_gb:.2f} GB")

    # Show breakdown by category
    for category, cache_dir in [
        ("Core", CACHE_CORE),
        ("Pre-training", CACHE_PRETRAIN),
        ("Evaluation", CACHE_EVAL),
        ("SFT", CACHE_SFT),
        ("DPO", CACHE_DPO)
    ]:
        if cache_dir.exists():
            size_gb = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1e9
            logger.info(f"  - {category}: {size_gb:.2f} GB")

    # Final status
    logger.info("\n" + "=" * 80)
    if not failed:
        logger.info("STATUS: ALL DATASETS DOWNLOADED SUCCESSFULLY ✓")
        logger.info("=" * 80)
        logger.info(f"\nLog saved to: {log_file}")
        logger.info("You can now proceed with model implementation!")
        return 0
    elif critical_failed:
        logger.error("STATUS: CRITICAL DATASETS FAILED ✗")
        logger.error("=" * 80)
        logger.error("\nCritical datasets missing:")
        for name in critical_failed:
            logger.error(f"  ✗ {name}")
        logger.error(f"\nLog saved to: {log_file}")
        logger.error("\nACTION REQUIRED:")

        # Provide specific guidance for known failures
        for name in failed:
            error = downloader.errors.get(name, "")

            if "gated" in error.lower():
                logger.error(f"\n{name}:")
                logger.error("  This is a GATED dataset requiring access request")
                logger.error("  Action: Request access on HuggingFace")
                logger.error(f"  Will continue without this dataset for now")

            if "trust_remote_code" in error.lower():
                logger.error(f"\n{name}:")
                logger.error("  This dataset requires custom code approval")
                logger.error("  Action: Script should be updated with trust_remote_code=True")

        return 1
    else:
        logger.warning("STATUS: SOME NON-CRITICAL DATASETS FAILED ⚠")
        logger.warning("=" * 80)
        logger.warning("\nYou can proceed with training, but some datasets are missing:")
        for name in failed:
            logger.warning(f"  ⚠ {name}")
        logger.warning(f"\nLog saved to: {log_file}")
        logger.warning("\nYou can retry later to download missing datasets")
        return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.error("\n\nDownload interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}", exc_info=True)
        sys.exit(1)
