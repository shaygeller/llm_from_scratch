# Implementation Timeline: 3-Week Execution Plan

**Project:** NanoLlama-1B Training on 2× H100 GPUs
**Total Duration:** 24 days (Week 0 prep + 3 weeks training)
**Start Date:** [Fill in your start date]

---

## Week 0: Preparation & Setup (Days -3 to 0)

**Goal:** Complete all setup before GPU training begins. No GPU time wasted.

### Day -3: Environment & Dependencies

**Morning (4 hours):**
- [ ] Clone repository structure
  ```bash
  mkdir -p nanollama/{data,models,checkpoints,logs,evals,scripts}
  cd nanollama
  git init
  ```
- [ ] Set up Python environment
  ```bash
  conda create -n nanollama python=3.11
  conda activate nanollama
  pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
  pip install transformers datasets accelerate wandb
  pip install flash-attn --no-build-isolation
  pip install outlines datasketch sentencepiece
  ```
- [ ] Create `requirements.txt` and freeze versions
  ```bash
  pip freeze > requirements.txt
  ```

**Afternoon (4 hours):**
- [ ] Download datasets (start background processes, will take hours)
  ```python
  # scripts/download_data.py
  from datasets import load_dataset

  # FineWeb-Edu (large, stream and cache)
  ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)
  # Cache first 30B tokens worth locally

  # PG19
  ds = load_dataset('pg19', split='train')

  # The Stack
  ds = load_dataset('bigcode/the-stack-dedup',
                    data_dir='data/python', split='train')

  # OpenWebMath
  ds = load_dataset('open-web-math/open-web-math', split='train')

  # Download tokenizer
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
  tokenizer.save_pretrained('models/tokenizer')
  ```
- [ ] Set up Weights & Biases account
  ```bash
  wandb login
  wandb init -p nanollama-1b
  ```

**Evening (2 hours):**
- [ ] Verify GPU setup
  ```python
  import torch
  print(f"GPUs available: {torch.cuda.device_count()}")
  print(f"GPU 0: {torch.cuda.get_device_name(0)}")
  print(f"GPU 1: {torch.cuda.get_device_name(1)}")
  # Expected: 2× NVIDIA H100 80GB
  ```
- [ ] Test FSDP basic functionality
  ```python
  # scripts/test_fsdp.py
  # Simple 2-layer transformer with FSDP to verify setup works
  ```

**Deliverable:** ✅ Environment ready, datasets downloading

---

### Day -2: Model Implementation

**Morning (4 hours):**
- [ ] Implement core model architecture (`models/nanollama.py`)
  - [ ] `RMSNorm` layer
  - [ ] `RotaryEmbedding` (RoPE) with configurable theta
  - [ ] `GroupedQueryAttention` (32 Q heads, 4 KV heads)
  - [ ] `SwiGLU` FFN
  - [ ] `TransformerBlock` (pre-norm: RMSNorm → Attention → RMSNorm → FFN)
  - [ ] `NanoLlama` model wrapper with embedding + 22 layers + LM head

**Afternoon (4 hours):**
- [ ] Model testing & validation
  ```python
  # Test forward pass
  model = NanoLlama(vocab_size=32000, n_layers=22, hidden_dim=2048, ...)
  input_ids = torch.randint(0, 32000, (2, 1024))  # batch=2, seq=1024
  logits = model(input_ids)
  assert logits.shape == (2, 1024, 32000)

  # Count parameters
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total parameters: {total_params:,}")  # Should be ~1.1B
  ```
- [ ] Implement z-loss in loss function
  ```python
  # models/losses.py
  def compute_loss_with_z_penalty(logits, labels, z_weight=1e-4):
      # Cross-entropy loss
      ce_loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

      # z-loss: prevent logits from drifting to extreme values
      z_loss = z_weight * torch.log(torch.sum(torch.exp(logits) ** 2))

      return ce_loss + z_loss, {"ce_loss": ce_loss.item(), "z_loss": z_loss.item()}
  ```

**Evening (2 hours):**
- [ ] Implement FSDP wrapper (`training/fsdp_utils.py`)
  ```python
  from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
  from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

  def wrap_model_fsdp(model):
      mixed_precision_policy = MixedPrecision(
          param_dtype=torch.bfloat16,
          reduce_dtype=torch.bfloat16,
          buffer_dtype=torch.bfloat16,
      )

      model = FSDP(
          model,
          sharding_strategy=ShardingStrategy.FULL_SHARD,
          mixed_precision=mixed_precision_policy,
          use_orig_params=True,  # For optimizer state dict
          device_id=torch.cuda.current_device(),
      )
      return model
  ```

**Deliverable:** ✅ Model implementation complete, tests pass

---

### Day -1: Data Pipeline & Training Loop

**Morning (4 hours):**
- [ ] Implement streaming data loader with token packing (`data/dataloader.py`)
  ```python
  class PackedDataLoader:
      def __init__(self, dataset, tokenizer, seq_len=4096, batch_size=32):
          self.dataset = dataset
          self.tokenizer = tokenizer
          self.seq_len = seq_len
          self.batch_size = batch_size
          self.buffer = []  # Token buffer for packing

      def __iter__(self):
          for example in self.dataset:
              tokens = self.tokenizer.encode(example['text'])
              self.buffer.extend(tokens)

              # Yield batches when buffer is full
              while len(self.buffer) >= self.seq_len * self.batch_size:
                  batch = []
                  for _ in range(self.batch_size):
                      batch.append(self.buffer[:self.seq_len])
                      self.buffer = self.buffer[self.seq_len:]
                  yield torch.tensor(batch)
  ```
- [ ] Implement WSD scheduler (`training/schedulers.py`)
  ```python
  class WSDScheduler:
      def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps,
                   max_lr, min_lr):
          self.warmup_steps = warmup_steps
          self.stable_steps = stable_steps
          self.decay_steps = decay_steps
          self.max_lr = max_lr
          self.min_lr = min_lr
          self.current_step = 0

      def step(self):
          if self.current_step < self.warmup_steps:
              # Linear warmup
              lr = self.max_lr * (self.current_step / self.warmup_steps)
          elif self.current_step < self.warmup_steps + self.stable_steps:
              # Constant
              lr = self.max_lr
          else:
              # Linear decay
              decay_progress = (self.current_step - self.warmup_steps - self.stable_steps) / self.decay_steps
              lr = self.max_lr - (self.max_lr - self.min_lr) * decay_progress

          for param_group in self.optimizer.param_groups:
              param_group['lr'] = lr

          self.current_step += 1
          return lr
  ```

**Afternoon (4 hours):**
- [ ] Implement main training loop (`train.py`)
  - [ ] Distributed setup (rank, world_size)
  - [ ] Model initialization with FSDP
  - [ ] Optimizer (AdamW) setup
  - [ ] Checkpoint saving/loading logic
  - [ ] Logging to W&B
  - [ ] Gradient clipping
  - [ ] Validation loop

**Evening (2 hours):**
- [ ] Test training loop with dummy data (1000 steps)
  ```bash
  # Should complete without errors, verify:
  # - Loss decreases
  # - Gradients are finite
  # - Checkpoints save correctly
  # - W&B logs appear
  torchrun --nproc_per_node=2 train.py --test_mode --steps 1000
  ```

**Deliverable:** ✅ Training pipeline ready, smoke test passed

---

### Day 0: Decontamination & Final Checks

**Morning (4 hours):**
- [ ] Download evaluation datasets
  ```python
  # MMLU
  ds = load_dataset('cais/mmlu', 'all')

  # GSM8K
  ds = load_dataset('gsm8k', 'main')

  # HellaSwag
  ds = load_dataset('Rowan/hellaswag')

  # ARC
  ds = load_dataset('ai2_arc', 'ARC-Challenge')
  ```
- [ ] Implement decontamination script (`scripts/decontaminate.py`)
  ```python
  from datasketch import MinHash, MinHashLSH

  def find_contaminated_examples(train_dataset, test_datasets, n=13, threshold=0.8):
      # Build LSH index for test sets
      lsh = MinHashLSH(threshold=threshold, num_perm=128)

      # Index all test examples
      for test_name, test_ds in test_datasets.items():
          for idx, example in enumerate(test_ds):
              m = create_minhash(tokenize(example['text']), n=13)
              lsh.insert(f"{test_name}_{idx}", m)

      # Query for each training example
      contaminated = []
      for idx, train_ex in enumerate(train_dataset):
          m = create_minhash(tokenize(train_ex['text']), n=13)
          matches = lsh.query(m)
          if matches:
              contaminated.append(idx)

      return contaminated
  ```
- [ ] Run decontamination on FineWeb-Edu
  ```bash
  python scripts/decontaminate.py \
    --train_dataset fineweb-edu \
    --test_datasets mmlu,gsm8k,hellaswag,arc \
    --output data/contaminated_ids.json
  ```

**Afternoon (3 hours):**
- [ ] Implement evaluation scripts (`evals/`)
  - [ ] `eval_mmlu.py` - 5-shot MMLU evaluation
  - [ ] `eval_gsm8k.py` - 8-shot GSM8K with CoT
  - [ ] `eval_hellaswag.py` - 0-shot HellaSwag
  - [ ] `eval_perplexity.py` - Validation perplexity
  - [ ] `eval_needle.py` - Needle-in-haystack for long context

**Evening (1 hour):**
- [ ] Final checklist before training
  - [ ] ✅ 2× H100 GPUs available and idle
  - [ ] ✅ All datasets downloaded (check `~/datasets/` cache)
  - [ ] ✅ Decontamination complete
  - [ ] ✅ Training script tested
  - [ ] ✅ Checkpointing works
  - [ ] ✅ W&B logging configured
  - [ ] ✅ Evaluation scripts ready
  - [ ] ✅ `tmux` or `screen` session for long-running job
  - [ ] ✅ Disk space: >500GB available for checkpoints
  - [ ] ✅ Monitoring alerts set up (email/Slack for failures)

**Deliverable:** ✅ Ready to start training Day 1 morning

---

## Week 1: Phase 1 Pre-training Begins (Days 1-7)

**Training Config:** 4K context, batch size 32, seq_len 4096

### Day 1: Launch Training

**Morning (2 hours):**
- [ ] Launch Phase 1 training
  ```bash
  tmux new -s nanollama

  torchrun --nproc_per_node=2 train.py \
    --config configs/phase1_pretrain.yaml \
    --output_dir checkpoints/phase1 \
    --wandb_project nanollama-1b \
    --wandb_run_name phase1_pretrain
  ```
- [ ] Monitor first 500 steps
  - [ ] Loss starts at ~10-11 (log(32000))
  - [ ] Loss decreases steadily
  - [ ] No NaN/Inf
  - [ ] Throughput: ~0.15-0.2 steps/sec (5-7 sec/step)
  - [ ] GPU utilization >90%

**Afternoon (Check-in):**
- [ ] Verify W&B dashboard shows:
  - [ ] Loss curve decreasing
  - [ ] Learning rate ramping up (warmup phase)
  - [ ] Gradient norms <10
  - [ ] Tokens/sec >20K

**Evening (Check-in):**
- [ ] Status: ~2,500 steps complete (~3.3B tokens seen)
- [ ] Estimated time to 50K steps: ~3.5 days

**Deliverable:** ✅ Training running smoothly

---

### Days 2-4: Monitor & Validate

**Daily Check-ins (2× per day):**
- [ ] Morning: Check W&B for anomalies
  - [ ] Loss still decreasing
  - [ ] No gradient explosions
  - [ ] GPUs haven't crashed
- [ ] Evening: Check throughput
  - [ ] Still ~0.15-0.2 steps/sec
  - [ ] If slower, investigate (thermal throttling? network issues?)

**Day 4 Evening - Milestone: 50K Steps**
- [ ] Run first validation (estimated: ~6.6B tokens seen)
  ```bash
  python evals/eval_perplexity.py \
    --checkpoint checkpoints/phase1/step_50000 \
    --dataset fineweb-edu-validation
  ```
- [ ] Expected: PPL ~20-25
- [ ] **Go/No-Go Decision:**
  - ✅ If PPL <25: Continue
  - ❌ If PPL >30: Investigate (learning rate too high? data issue?)

**Deliverable:** ✅ 50K steps, PPL <25

---

### Days 5-7: Continue Training

**Day 5-6: Routine monitoring**
- [ ] Continue training toward 100K steps
- [ ] Daily checks: loss, gradients, throughput

**Day 7 Evening - Milestone: 100K Steps**
- [ ] Validation checkpoint (~13.1B tokens seen)
  ```bash
  python evals/eval_perplexity.py \
    --checkpoint checkpoints/phase1/step_100000
  ```
- [ ] Expected: PPL ~15-18
- [ ] **Go/No-Go:**
  - ✅ If PPL <18: On track
  - ⚠️ If PPL 18-22: Slower convergence, but acceptable
  - ❌ If PPL >22: Consider extending Phase 1 by 20K steps

**Week 1 Summary:**
- Steps completed: ~100K / 230K (43%)
- Tokens seen: ~13B / 30B (43%)
- Time remaining: 2 weeks for 130K steps ✅ On schedule

---

## Week 2: Phase 1 Completion (Days 8-14)

### Days 8-11: Continue Phase 1

**Daily routine:**
- [ ] Morning: W&B check
- [ ] Evening: Progress update (expect ~12K steps/day)

**Day 11 Evening - Milestone: 150K Steps**
- [ ] Validation (~19.7B tokens seen)
  ```bash
  python evals/eval_perplexity.py \
    --checkpoint checkpoints/phase1/step_150000
  ```
- [ ] Expected: PPL ~13-15
- [ ] **Go/No-Go:**
  - ✅ If PPL <15: Excellent
  - ⚠️ If PPL 15-17: Acceptable, continue
  - ❌ If PPL >17: May need to extend training

**Deliverable:** ✅ 150K steps, PPL trending toward <12

---

### Days 12-14: Approach Phase 1 End

**Day 13 Evening - Milestone: 184K Steps (80% complete)**
- [ ] **CRITICAL CHECKPOINT:** End of Phase 1, start of Phase 2
  ```bash
  python evals/eval_perplexity.py \
    --checkpoint checkpoints/phase1/step_184000
  ```
- [ ] Expected: PPL ~12-13
- [ ] Save extra checkpoint (backup)
  ```bash
  cp -r checkpoints/phase1/step_184000 checkpoints/phase1_FINAL_BACKUP
  ```

**Day 14: Transition to Phase 2**

**Morning (1 hour prep):**
- [ ] Stop training (graceful shutdown)
- [ ] Verify Phase 1 checkpoint integrity
- [ ] Update training config for Phase 2:
  ```yaml
  # configs/phase2_annealing.yaml
  sequence_length: 8192  # Starting at 8K
  batch_size_per_gpu: 8  # Halved due to 2× seq len
  global_batch_size: 16
  dataset: pg19_and_stack_mixed
  resume_from: checkpoints/phase1/step_184000
  start_step: 184000
  rope_theta: 10000.0  # Will ramp to 500000.0
  ```

**Afternoon (2 hours):**
- [ ] Launch Phase 2 training (8K context)
  ```bash
  torchrun --nproc_per_node=2 train.py \
    --config configs/phase2_annealing.yaml \
    --output_dir checkpoints/phase2 \
    --wandb_run_name phase2_annealing_8k
  ```
- [ ] Monitor first 100 steps at 8K context
  - [ ] Loss might spike slightly (new data distribution + longer context)
  - [ ] Should stabilize after ~500 steps
  - [ ] Throughput: ~0.08-0.12 steps/sec (slower due to 8K context)

**Week 2 Summary:**
- Steps completed: ~185K / 230K (80%)
- Tokens seen: ~24B / 30B (80%)
- Phase 1 COMPLETE ✅
- Phase 2 STARTED ✅

---

## Week 3: Phase 2 Annealing + Phase 3 Crystallization (Days 15-21)

### Days 15-16: Phase 2 - 8K Context Training

**Running Config:**
- Sequence length: 8192
- Batch size: 16 global
- Dataset: PG19 (50%) + The Stack (50%)
- Target: 10K steps (steps 184K → 194K)

**Daily monitoring:**
- [ ] Verify loss continues decreasing (might be higher due to new data)
- [ ] Check attention patterns (are later layers attending to full 8K?)
  ```python
  python scripts/visualize_attention.py \
    --checkpoint checkpoints/phase2/step_190000 \
    --input_length 8192
  ```

**Day 16 Evening - Milestone: 194K Steps**
- [ ] Completed 8K context phase
- [ ] **Next:** Transition to 16K context

---

### Days 17-18: Phase 2 - 16K Context Training

**Morning Day 17 (1 hour):**
- [ ] Update config for 16K context
  ```yaml
  sequence_length: 16384
  batch_size_per_gpu: 4  # Halved again
  global_batch_size: 8
  rope_theta: 500000.0  # Extended theta for 16K
  ```
- [ ] Resume training
  ```bash
  torchrun --nproc_per_node=2 train.py \
    --config configs/phase2_16k.yaml \
    --resume_from checkpoints/phase2/step_194000
  ```
- [ ] Monitor first 100 steps
  - [ ] Throughput: ~0.04-0.06 steps/sec (16K is expensive)
  - [ ] GPU memory: Should be <70GB per GPU with Flash Attention

**Day 18 Evening - Milestone: 207K Steps (90% total)**
- [ ] Phase 2 COMPLETE ✅
- [ ] Run needle-in-haystack test
  ```bash
  python evals/eval_needle.py \
    --checkpoint checkpoints/phase2/step_207000 \
    --context_length 16384 \
    --num_tests 100
  ```
- [ ] **Go/No-Go:**
  - ✅ If accuracy ≥85%: 16K context working
  - ⚠️ If accuracy 70-85%: Acceptable, model uses some context
  - ❌ If accuracy <70%: Bug in RoPE or attention masking

**Deliverable:** ✅ Phase 2 complete, 16K context validated

---

### Days 19-20: Phase 3 - Knowledge Crystallization

**Morning Day 19 (1 hour):**
- [ ] Update config for Phase 3
  ```yaml
  # Back to 4K for final phase
  sequence_length: 4096
  batch_size_per_gpu: 16
  global_batch_size: 32
  dataset: gold_sources_mixed  # OpenWebMath + StackExchange + STEM Wiki
  resume_from: checkpoints/phase2/step_207000
  ```
- [ ] Launch Phase 3
  ```bash
  torchrun --nproc_per_node=2 train.py \
    --config configs/phase3_crystallization.yaml \
    --output_dir checkpoints/phase3
  ```

**Target:** 23K steps (207K → 230K)
**Duration:** ~1.5-2 days

**Daily monitoring:**
- [ ] Loss should continue decreasing (high-quality data)
- [ ] Learning rate decaying linearly (decay phase of WSD)

**Day 20 Evening - Milestone: 230K Steps (100%)**
- [ ] **PRE-TRAINING COMPLETE** 🎉
- [ ] Final validation
  ```bash
  python evals/eval_perplexity.py \
    --checkpoint checkpoints/phase3/step_230000
  ```
- [ ] Expected: PPL <12 ✅
- [ ] **Go/No-Go for SFT:**
  - ✅ If PPL <12: Proceed to SFT
  - ⚠️ If PPL 12-13: Acceptable, proceed
  - ❌ If PPL >13: Consider extending 5K-10K steps

**Deliverable:** ✅ Pre-training complete, base model ready

---

### Day 21: SFT Preparation & Launch

**Morning (3 hours):**
- [ ] Prepare SFT dataset
  ```python
  # Load and format datasets
  chat_ds = load_dataset('OpenAssistant/oasst2')
  math_ds = load_dataset('meta-math/MetaMathQA-400K')
  tool_ds = generate_tool_use_examples()  # Or load from APIBench

  # Mix: 50% chat, 25% math, 25% tool
  sft_dataset = mix_datasets(chat_ds, math_ds, tool_ds, ratios=[0.5, 0.25, 0.25])

  # Apply chat template
  sft_dataset = sft_dataset.map(apply_chat_template)
  ```
- [ ] Add special tokens to tokenizer
  ```python
  tokenizer.add_special_tokens({
      'additional_special_tokens': [
          '<|im_start|>', '<|im_end|>',
          '<|start_thought|>', '<|end_thought|>',
          '<|tool_call|>', '<|tool_response|>'
      ]
  })
  # Resize model embeddings from 32000 → 32008
  model.resize_token_embeddings(len(tokenizer))
  ```
- [ ] Implement loss masking for assistant tokens only
  ```python
  def compute_sft_loss(logits, labels, attention_mask):
      # Mask user tokens (labels = -100 for ignored tokens)
      loss = F.cross_entropy(
          logits.view(-1, vocab_size),
          labels.view(-1),
          ignore_index=-100
      )
      return loss
  ```

**Afternoon (2 hours):**
- [ ] Launch SFT training
  ```bash
  torchrun --nproc_per_node=2 train_sft.py \
    --config configs/sft.yaml \
    --base_model checkpoints/phase3/step_230000 \
    --output_dir checkpoints/sft \
    --num_steps 12000 \
    --batch_size 8 \
    --learning_rate 1e-5
  ```
- [ ] Monitor first 500 steps
  - [ ] Loss should be lower (~2-3) than pre-training
  - [ ] Throughput: ~0.3-0.4 steps/sec (2K seq len, smaller batch)

**Evening:**
- [ ] SFT training progressing (will take ~2-3 days)

**Week 3 Summary:**
- Phase 2 complete ✅
- Phase 3 complete ✅
- Pre-training DONE ✅
- SFT started ✅

---

## Week 4: SFT, DPO, Evaluation & Deployment (Days 22-24+)

### Days 22-23: SFT Completion

**Routine monitoring:**
- [ ] Check loss converging (target: ~1.5-2.0 final loss)
- [ ] Spot-check generations every 2K steps
  ```python
  python scripts/test_generation.py \
    --checkpoint checkpoints/sft/step_6000 \
    --prompt "What is 15 * 24?"

  # Expected output:
  # <|start_thought|>
  # I need to multiply 15 by 24. Let me break it down:
  # 15 * 20 = 300
  # 15 * 4 = 60
  # Total: 300 + 60 = 360
  # <|end_thought|>
  # The answer is 360.
  ```

**Day 23 Evening - Milestone: SFT Complete (12K steps)**
- [ ] Save final SFT checkpoint
- [ ] Run preliminary benchmarks
  ```bash
  # MMLU
  python evals/eval_mmlu.py \
    --checkpoint checkpoints/sft/step_12000 \
    --shots 5

  # GSM8K
  python evals/eval_gsm8k.py \
    --checkpoint checkpoints/sft/step_12000 \
    --shots 8 \
    --cot True

  # Tool use accuracy
  python evals/eval_tool_use.py \
    --checkpoint checkpoints/sft/step_12000
  ```

**Expected Results:**
- MMLU: 28-35% (success if ≥30%)
- GSM8K: 12-20% (success if ≥15%)
- Tool use (syntax): ≥95% with grammar constraints
- Tool use (semantic): 55-70%

**Go/No-Go for DPO:**
- ✅ If MMLU ≥28% and GSM8K ≥12%: Proceed to DPO
- ⚠️ If below targets: Can skip DPO and iterate on SFT
- ❌ If MMLU <25%: Something wrong, debug before DPO

**Deliverable:** ✅ SFT complete, benchmarks meet minimum targets

---

### Day 24: DPO Training & Final Evaluation

**Morning (2 hours):**
- [ ] Prepare DPO dataset
  ```python
  dpo_ds = load_dataset('Anthropic/hh-rlhf')
  # Format: {chosen: "...", rejected: "..."}
  ```
- [ ] Launch DPO
  ```bash
  torchrun --nproc_per_node=2 train_dpo.py \
    --config configs/dpo.yaml \
    --base_model checkpoints/sft/step_12000 \
    --output_dir checkpoints/dpo \
    --num_steps 5000 \
    --batch_size 4 \
    --learning_rate 5e-7 \
    --beta 0.1
  ```
- [ ] Monitor first 200 steps
  - [ ] DPO loss decreasing (target: <0.5)
  - [ ] Reference model on CPU (saves VRAM)

**Afternoon (DPO continues, ~8-10 hours):**
- [ ] Meanwhile: Set up evaluation infrastructure
  ```bash
  # Install lm-eval-harness
  pip install lm-eval

  # Test evaluation
  lm_eval --model hf \
    --model_args pretrained=checkpoints/sft/step_12000 \
    --tasks mmlu \
    --batch_size 8
  ```

**Evening - Milestone: DPO Complete (5K steps)**
- [ ] Final model checkpoint saved at `checkpoints/dpo/step_5000`

---

### Days 25-26: Comprehensive Evaluation

**Full Benchmark Suite:**

```bash
# Core benchmarks
python evals/eval_mmlu.py --checkpoint checkpoints/dpo/step_5000 --shots 5
python evals/eval_gsm8k.py --checkpoint checkpoints/dpo/step_5000 --shots 8 --cot
python evals/eval_hellaswag.py --checkpoint checkpoints/dpo/step_5000 --shots 0
python evals/eval_arc.py --checkpoint checkpoints/dpo/step_5000 --shots 25

# Alignment benchmarks
python evals/eval_truthfulqa.py --checkpoint checkpoints/dpo/step_5000
python evals/eval_toxicity.py --checkpoint checkpoints/dpo/step_5000
python evals/eval_mt_bench.py --checkpoint checkpoints/dpo/step_5000

# Long context
python evals/eval_needle.py --checkpoint checkpoints/dpo/step_5000 --context 16384

# Tool use
python evals/eval_tool_use.py --checkpoint checkpoints/dpo/step_5000

# Perplexity
python evals/eval_perplexity.py --checkpoint checkpoints/dpo/step_5000
```

**Create Results Table:**

| Benchmark | Score | Target | Status |
|-----------|-------|--------|--------|
| MMLU 5-shot | 35.2% | ≥30% | ✅ Success |
| GSM8K 8-shot CoT | 18.7% | ≥15% | ✅ Success |
| HellaSwag 0-shot | 52.1% | ≥50% | ✅ Success |
| ARC-Challenge | 32.4% | ≥30% | ✅ Success |
| TruthfulQA | 36.8% | ≥35% | ✅ Success |
| Toxicity (max) | 0.24 | <0.30 | ✅ Success |
| Needle @ 16K | 87.0% | ≥85% | ✅ Success |
| Tool Syntax | 97.5% | ≥95% | ✅ Success |
| Tool Semantic | 64.2% | ≥60% | ✅ Success |
| **Final PPL** | **11.8** | **<12** | **✅ Success** |

**Deliverable:** ✅ Complete evaluation, results documented

---

### Day 27: Quantization & Deployment

**Morning (3 hours):**
- [ ] Convert to GGUF format
  ```bash
  # Install llama.cpp
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp && make

  # Convert model
  python convert-hf-to-gguf.py \
    --outfile nanollama-1b-f16.gguf \
    --outtype f16 \
    checkpoints/dpo/step_5000

  # Quantize to different precisions
  ./quantize nanollama-1b-f16.gguf nanollama-1b-q8_0.gguf q8_0
  ./quantize nanollama-1b-f16.gguf nanollama-1b-q4_k_m.gguf q4_k_m
  ```

**Afternoon (2 hours):**
- [ ] Test quantized models
  ```bash
  # Test Q8_0
  ./main -m nanollama-1b-q8_0.gguf \
    -p "What is 25 * 48?" \
    -n 256

  # Test Q4_K_M
  ./main -m nanollama-1b-q4_k_m.gguf \
    -p "Explain quantum entanglement" \
    -n 512
  ```
- [ ] Benchmark inference speed
  ```bash
  # CPU inference (M1/M2 Mac or x86)
  ./perplexity -m nanollama-1b-q4_k_m.gguf -f test.txt

  # Measure tokens/sec
  ```

**Evening (2 hours):**
- [ ] Build demo UI (optional)
  ```python
  # demo_ui.py using Gradio
  import gradio as gr
  from transformers import AutoModelForCausalLM, AutoTokenizer

  model = AutoModelForCausalLM.from_pretrained('checkpoints/dpo/step_5000')
  tokenizer = AutoTokenizer.from_pretrained('models/tokenizer')

  def chat(message, history):
      # Format with <|im_start|> tags
      # Generate response
      # Extract and display <|start_thought|> separately
      return response, thoughts

  gr.ChatInterface(
      chat,
      title="NanoLlama-1B Demo",
      description="1B parameter model with chain-of-thought reasoning"
  ).launch()
  ```

**Deliverable:** ✅ Model quantized, demo ready

---

## Final Deliverables Checklist

### Models
- [ ] `checkpoints/phase1/step_184000` - Base pre-trained (Phase 1)
- [ ] `checkpoints/phase3/step_230000` - Full pre-trained model
- [ ] `checkpoints/sft/step_12000` - SFT model
- [ ] `checkpoints/dpo/step_5000` - Final aligned model
- [ ] `nanollama-1b-q8_0.gguf` - 8-bit quantized
- [ ] `nanollama-1b-q4_k_m.gguf` - 4-bit quantized

### Evaluation Results
- [ ] `results/benchmark_results.json` - All benchmark scores
- [ ] `results/comparison_table.md` - vs TinyLlama, MobileLLM
- [ ] `results/qualitative_samples.txt` - Generation examples
- [ ] `results/attention_visualizations.png` - 16K context heatmaps

### Documentation
- [ ] `training_log.md` - Daily progress notes
- [ ] `hyperparameters.yaml` - Final config used
- [ ] `requirements.txt` - Exact package versions
- [ ] `README.md` - Model card with usage instructions

### Code
- [ ] `models/nanollama.py` - Model architecture
- [ ] `train.py` - Pre-training script
- [ ] `train_sft.py` - SFT script
- [ ] `train_dpo.py` - DPO script
- [ ] `evals/` - All evaluation scripts
- [ ] `demo_ui.py` - Gradio demo

### Reproducibility
- [ ] `configs/` - All training configs
- [ ] `scripts/decontaminate.py` - Decontamination script
- [ ] `data/contaminated_ids.json` - List of removed examples
- [ ] `wandb_run_urls.txt` - Links to W&B dashboards
- [ ] `git_commit_hash.txt` - Code version used

---

## Contingency Plans

### If Training Diverges (Loss → NaN)
1. Rollback to last good checkpoint (5K steps back)
2. Reduce learning rate by 2×
3. Increase gradient clipping (1.0 → 0.5)
4. Check for data corruption in current batch

### If GPU Crashes
1. Auto-resume from last checkpoint (already implemented)
2. Check for hardware issues (nvidia-smi, temps)
3. If persistent: reduce batch size or seq length

### If Benchmarks Below Target
- **MMLU <28%:** Extend Phase 3 by 10K steps (more crystallization)
- **GSM8K <12%:** Add more math data to SFT (increase CoT ratio to 40%)
- **Tool use <60%:** Generate more tool examples, improve prompts

### If Running Out of Time
**Priority order:**
1. Complete pre-training (230K steps) - NON-NEGOTIABLE
2. Run SFT (can reduce to 8K steps minimum)
3. Skip DPO if needed (SFT model is still usable)
4. Quantization can be done later offline

---

## Success Metrics Summary

**Minimum Viable Product (21 days):**
- ✅ Pre-training complete (230K steps, PPL <12)
- ✅ SFT complete (MMLU ≥28%, GSM8K ≥12%)
- ✅ Model runs on consumer hardware (Q4 quantized)

**Success (Research Validated - 24 days):**
- ✅ MMLU ≥35% with only 30B tokens (proves quality > quantity)
- ✅ GSM8K ≥20% (reasoning curriculum works)
- ✅ 16K context functional (needle test ≥85%)
- ✅ DPO improves alignment

**Stretch (Publication-Ready - 26 days):**
- ✅ MMLU ≥40% (matching 1.6B models)
- ✅ GSM8K ≥25%
- ✅ Comprehensive evaluation suite
- ✅ Demo UI with interpretable reasoning
- ✅ Writeup: "How Curriculum Design Outperforms Scale"

---

**Total Timeline: 24-27 days from Day 0**
**GPU Time: ~21 days active training**
**Success Probability: High (conservative targets, proven architecture)**

Ready to start Week 0 prep? 🚀
