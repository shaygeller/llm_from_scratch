# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project builds a 1.1B parameter language model (NanoLlama-1B) from scratch to demonstrate that small models can achieve high reasoning capabilities through intelligent training strategies rather than just parameter count.

**Current Status:** Planning phase - PRD documents are complete, implementation has not yet begun.

## Four-Phase Architecture

The project follows a structured four-phase approach, each with specific technical requirements:

### Phase 1: Foundation & Pre-training
- Build decoder-only transformer with Llama-3 architecture (22 layers, 2048 hidden dim, ~1.1B params)
- Implement Grouped Query Attention (32 query heads, 4 KV heads) for memory efficiency
- Use RoPE positional embeddings, SwiGLU activation, RMSNorm
- Distributed training via PyTorch FSDP with BFloat16 precision
- Three-stage curriculum: General Knowledge (FineWeb-Edu) → Long Context Annealing (PG19 + The Stack) → Knowledge Crystallization (textbooks, papers)
- Context window expansion: 4K → 128K tokens during training
- WSD (Warmup-Stable-Decay) scheduler with max LR 3e-4
- Token packing pipeline to maximize compute efficiency

### Phase 2: SFT & Agent Reasoning
- Supervised fine-tuning with ReAct data mixture: 50% chat, 25% CoT math, 25% tool use
- Special tokens: `<|im_start|>`, `<|im_end|>`, `<|start_thought|>`, `<|end_thought|>`, `<|tool_call|>`
- Loss masking: compute loss only on assistant-generated tokens
- Grammar-constrained decoding via `outlines` library for valid JSON tool calls
- Agent loop: Thought → Tool Call → Python Execution → Tool Response → Final Answer

### Phase 3: Alignment (DPO) & Evaluation
- Direct Preference Optimization with β=0.1 (KL divergence penalty)
- Reference model: frozen copy of SFT model (offload to CPU for memory efficiency)
- N-gram decontamination: remove 13-gram overlaps between training and test sets (MMLU, GSM8K)
- Process-based evaluation: verify reasoning steps in `<|start_thought|>` blocks, not just final answers

### Phase 4: Deployment
- Export to GGUF format for edge deployment (llama.cpp/Ollama compatibility)
- Quantization targets: Q8_0 (server), Q4_K_M (laptops/phones)
- "Glass Box" UI: expose hidden thoughts and raw tool calls for debugging

## Critical Technical Constraints

### Architecture Decisions
- **GQA (32:4 ratio):** Reduces KV cache by 8x - essential for 1B models that are memory-bandwidth bound
- **BFloat16:** Required over FP16 to prevent loss instabilities (NaN/zero) due to wider dynamic range
- **FSDP FULL_SHARD:** Model + optimizer states require ~16GB VRAM in FP32 - sharding is mandatory
- **z-loss (1e-4):** Prevents logits from drifting to 10^4, which causes numerical errors in softmax
- **Token Packing:** Without packing, short documents waste 97%+ of compute in fixed-size windows

### Data Strategy Rationale
- **Phase 1 (0-80%):** FineWeb-Edu for general knowledge foundation
- **Phase 2 (80-90%):** PG19 + The Stack forces attention to track long-range dependencies (sequence length ramps 4K → 32K → 128K)
- **Phase 3 (90-100%):** High-quality sources during LR decay creates beneficial "overfitting" to logical patterns
- **Why this matters:** The curriculum mimics human learning (basics → complex skills → mastery)

### Training Stability
- **WSD over Cosine:** More flexible - can stop stable phase and trigger decay anytime; Cosine locks duration from step 1
- **Loss Masking:** Training on user tokens teaches the model to predict questions, not answer them
- **Reference Model CPU Offload:** DPO doubles memory; reference model doesn't need gradients, so keep in CPU RAM

## Key Dependencies (To Be Installed)

When implementing, the codebase will require:
- PyTorch 2.4+ (nightly builds for latest FSDP features)
- `torch.distributed.fsdp` for distributed training
- `outlines` library for grammar-constrained decoding
- Tokenizer (likely HuggingFace `transformers` for Llama-3 tokenizer)
- Data processing: libraries for MinHash deduplication, streaming pipelines
- Evaluation: MMLU, GSM8K benchmark datasets

## Implementation Guidelines

### Model Architecture (`model.py` or similar)
- Implement NanoLlama-1B following exact specs in PRD 1, Section 2.1
- RoPE base_theta=10,000.0 for base training, extension_theta=10,000,000.0 during annealing
- Pre-norm RMSNorm placement (before attention and FFN blocks)
- Grouped Query Attention with explicit 32:4 head configuration

### Training Loop (`train.py` or similar)
- FSDP configuration: `FULL_SHARD` strategy, BF16 mixed precision
- WSD scheduler: 2000-step warmup, stable at 3e-4, linear decay to 0
- z-loss implementation: `1e-4 * torch.log(torch.sum(torch.exp(logits) ** 2))`
- Curriculum switching logic at 80% and 90% of total steps
- Context window ramping during Phase 2 (80-90%)

### Data Pipeline (`data.py` or similar)
- Streaming data loader with on-the-fly tokenization
- Token packing implementation to fill 4K/32K/128K windows
- MinHash deduplication logic
- Dataset switching based on training phase percentage

### SFT Training (`sft_train.py` or similar)
- Special token definitions: `<|im_start|>`, `<|im_end|>`, `<|start_thought|>`, `<|end_thought|>`, `<|tool_call|>`
- Loss masking: zero out gradients for user tokens (only train on assistant responses)
- Data format validation: ensure correct token structure for chat, CoT, and tool use

### DPO Training (`dpo_train.py` or similar)
- Reference model initialization: `reference_model = copy.deepcopy(sft_model).requires_grad_(False)`
- Beta parameter: 0.1 (KL penalty coefficient)
- Memory optimization: move reference model to CPU if GPU memory constrained

### Evaluation (`eval.py` or similar)
- N-gram decontamination script: check 13-gram overlaps with MMLU/GSM8K before training
- Process evaluation: parse `<|start_thought|>` ... `<|end_thought|>` blocks and verify logical steps
- Benchmark metrics: MMLU accuracy, GSM8K accuracy with process verification

### Inference & Demo (`inference.py`, `demo_ui.py` or similar)
- Grammar-constrained decoding using `outlines` FSM for JSON tool calls
- Agent loop implementation with thought → tool call → execution → response flow
- UI must expose both `<|start_thought|>` content and raw JSON tool calls

## Design Principles

1. **Efficiency Over Scale:** Every architectural choice prioritizes compute/memory efficiency for 1B parameter constraints
2. **Quality Over Quantity:** Data curation and curriculum design matter more than dataset size
3. **Interpretability by Design:** Chain-of-thought and tool call transparency enable debugging
4. **Production-First:** Plan for quantization and edge deployment from the start, not as an afterthought
5. **Stability Through Simplicity:** DPO over RLHF, WSD over Cosine - prefer mathematically simpler, more stable approaches

## Common Pitfalls to Avoid

- Do NOT use FP16 precision (will cause loss instabilities) - always use BF16
- Do NOT train without loss masking on user tokens - model will learn to generate questions
- Do NOT use standard attention (32 KV heads) - must use GQA (4 KV heads) for memory efficiency
- Do NOT skip token packing - wastes 97% of compute on short documents
- Do NOT evaluate without N-gram decontamination - may measure memorization, not reasoning
- Do NOT use absolute positional embeddings - RoPE is essential for long-context performance
- Do NOT implement RLHF/PPO for alignment - DPO is simpler and more stable

## PRD Reference

All technical specifications are defined in four PRD files:
- `PRD 1: Phase 1 - Foundation & Pre-training` - Architecture, data curriculum, training loop
- `PRD 2: Phase 2 - SFT & Agent Reasoning` - ReAct mixture, formatting, grammar-constrained decoding
- `PRD 3: Phase 3 - Alignment (DPO) & Eval` - DPO implementation, evaluation methodology
- `PRD 4: Phase 4 - Deployment & Extras` - Quantization (GGUF), UI design

When implementing any component, consult the corresponding PRD section for detailed rationale and specifications.
