#!/bin/bash
# Quick start script to begin dataset downloads
# Run this in a tmux/screen session as it takes 3-5 hours

set -e

echo "========================================="
echo "NanoLlama Dataset Download"
echo "========================================="
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Check if logs directory exists
if [ ! -d "logs" ]; then
    echo "Creating logs directory..."
    mkdir -p logs
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data
fi

# Check HuggingFace authentication
echo "Checking HuggingFace authentication..."
if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo ""
    echo "⚠️  WARNING: Not logged in to HuggingFace"
    echo "   The Llama-3 tokenizer requires authentication."
    echo ""
    echo "   Please run: huggingface-cli login"
    echo "   Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Starting download in background..."
echo "This will take 3-5 hours depending on your connection."
echo ""
echo "Logs will be written to: logs/dataset_download.log"
echo ""

# Run download in background
nohup python3 scripts/download_datasets.py --all > logs/dataset_download.log 2>&1 &
pid=$!

echo "✓ Download started (PID: $pid)"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/dataset_download.log"
echo ""
echo "Check if still running:"
echo "  ps aux | grep download_datasets"
echo ""
echo "Kill if needed:"
echo "  kill $pid"
echo ""
