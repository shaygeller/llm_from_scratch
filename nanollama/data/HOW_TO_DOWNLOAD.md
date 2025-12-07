# How to Download Datasets

## Quick Start

**Single command to download and verify all datasets:**

```bash
cd nanollama/data
python3 download_and_verify.py
```

That's it! The script will:
1. Download all 15 required datasets
2. Create a timestamped log file
3. Verify critical datasets
4. Report success or failure with details

## What the Script Does

### Automatic Features
- ✅ Downloads all datasets in correct order
- ✅ Creates timestamped logs: `logs/dataset_download_YYYYMMDD_HHMMSS.log` (within data folder)
- ✅ Tracks successes and failures
- ✅ Verifies tokenizer and FineWeb-Edu exist
- ✅ Reports cache directory size
- ✅ Returns exit code 0 (success) or 1 (failure)

### Error Handling
- Continues downloading even if some datasets fail
- Reports specific errors for each failed dataset
- Provides guidance for known issues (gated datasets, trust_remote_code)
- Distinguishes critical vs non-critical failures

## Exit Codes

**Exit Code 0 (Success)**
- All datasets downloaded successfully
- OR non-critical datasets failed but training can proceed

**Exit Code 1 (Failure)**
- Critical datasets failed (tokenizer, FineWeb-Edu)
- Training cannot proceed

## Log Files

Each run creates a new timestamped log within the data folder:
- `logs/dataset_download_20251205_231313.log`
- `logs/dataset_download_20251206_143022.log`
- etc.

Logs contain:
- Detailed download progress
- Success/failure for each dataset
- Error messages
- Final summary
- Verification results

## Checking Results

### During Download
Watch live progress:
```bash
cd nanollama/data
tail -f logs/dataset_download_*.log
```

### After Download
Check the summary at end of log:
```bash
cd nanollama/data
# View latest log
ls -t logs/dataset_download_*.log | head -1 | xargs cat

# Check exit code
echo $?  # 0 = success, 1 = failure
```

### Verify Downloaded Datasets
```bash
# Check cache size
du -sh data/cache

# Check tokenizer
ls -la models/tokenizer/

# Check FineWeb marker
cat data/fineweb_edu_validation.txt
```

## What Gets Downloaded

### Phase 1 (Core)
1. Tokenizer (Llama-3, 128K vocab)
2. FineWeb-Edu (streaming, 24B tokens)

### Phase 2 (Pre-training)
3. PG19 (28,602 books)
4. The Stack (Python, JS, TS code)
5. OpenWebMath (math content)
6. StackExchange (10.8M examples)

### Phase 3 (Evaluation)
7. MMLU (14K test examples)
8. GSM8K (1.3K test examples)
9. HellaSwag (10K validation)
10. ARC-Challenge (1.2K test)
11. TruthfulQA (817 validation)

### Phase 4 (Fine-tuning)
12. OpenAssistant (128K chat)
13. MetaMathQA (395K math)
14. HH-RLHF (160K DPO pairs)
15. Orca-DPO (12.8K DPO pairs)

## Disk Space

**Total required:** ~70-75 GB

**Breakdown:**
- The Stack: ~40 GB
- StackExchange: ~15 GB
- OpenWebMath: ~10 GB
- Others: ~5-10 GB

## Time Estimates

With good internet (100 Mbps):
- First run (all datasets): ~30-60 minutes
- Re-run (cached): ~2-3 minutes
- Verification only: <1 minute

## Troubleshooting

### "Gated dataset" Error
Some datasets require access request on HuggingFace:
1. Visit the dataset page (URL in error message)
2. Click "Request Access"
3. Wait for approval (~24 hours)
4. Re-run script

### "trust_remote_code" Error
This is a safety check for datasets with custom code:
- The script handles this automatically
- If you see this error, the script needs updating

### Network Timeout
If download times out:
```bash
# Re-run the script
python3 download_and_verify.py

# HuggingFace datasets library resumes from cache
# You won't re-download what you already have
```

### Disk Space Full
Check available space:
```bash
df -h
```

Need more space? You can:
1. Use external drive for cache:
   ```bash
   export HF_HOME=/path/to/external/drive/cache
   python3 download_and_verify.py
   ```

2. Download only critical datasets (modify script)

## Advanced Usage

### Re-download Specific Dataset
Edit `data/download_and_verify.py` and comment out datasets you don't need:

```python
# Download only what you need
download_tokenizer(downloader)
download_fineweb_edu(downloader)
# download_pg19(downloader)  # Skip this
# download_the_stack(downloader)  # Skip this
```

### Check Without Downloading
The script uses HuggingFace cache, so re-running is fast if data exists.

### Clean Cache and Re-download
```bash
# WARNING: This deletes all downloaded data!
rm -rf data/cache
python3 download_and_verify.py
```

## Integration with Training

The script is designed to be run before training:

```bash
# Step 1: Download datasets
cd data
python3 download_and_verify.py
if [ $? -eq 0 ]; then
    echo "✓ Datasets ready"
    # Step 2: Start training
    python3 train.py
else
    echo "✗ Dataset download failed"
    exit 1
fi
```

## Success Checklist

After running the script, verify:
- [ ] Exit code is 0
- [ ] Log shows "ALL DATASETS DOWNLOADED SUCCESSFULLY ✓"
- [ ] `models/tokenizer/` directory exists
- [ ] `data/fineweb_edu_validation.txt` exists
- [ ] Cache directory is ~70GB: `du -sh data/cache`
- [ ] 15/15 datasets in success list

If all checked, you're ready to proceed with training!

---

**Current Status:** ✅ All datasets already downloaded (71.16 GB)

You can verify by running:
```bash
cd data
python3 download_and_verify.py
```

Expected: Completes in <5 minutes (uses cached data)
