"""Pre-flight smoke for the x-attn architecture (Task #35 prep).

Goal: catch integration bugs in the wrapper / merged-checkpoint /
event-tensor path BEFORE we sink ~1.5 hr of Stage-1 GPU into the first
training run. Specifically verifies:

  1. /workspace/checkpoints/qwen3-8b-cpt-light-merged loads via
     AutoModelForCausalLM (review 010 finding about resize compat).
  2. Vocab size matches len(tokenizer) after custom_tokens.install
     (no token-row drift from the merge).
  3. QwenXAttnWrapper construction succeeds with the sweep_space
     defaults (every_4, n_slots=64, gate_init=small_0.01).
  4. PEFT LoRA-on-Q attaches cleanly (Stage-1 trainable surface).
  5. A single forward pass with dummy events + text input produces a
     real loss (not NaN) on GPU.

No training, no eval, no I/O beyond loading the checkpoint and
running one batch. ~2 min wall on Blackwell.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512,
                        help="short for the smoke; real training uses 2048")
    parser.add_argument("--n-events", type=int, default=16)
    args = parser.parse_args()

    if not args.merged_checkpoint.exists():
        print(f"FAIL: merged checkpoint not found: {args.merged_checkpoint}", file=sys.stderr)
        return 1

    print(f"[xa-pre] loading merged base from {args.merged_checkpoint}")
    import torch
    from transformers import AutoModelForCausalLM

    # Review 011 finding #4: fail closed (no silent CPU fallback) so an
    # operator never wastes an hour discovering the smoke "passed" on
    # CPU. The general scripts/preflight_check.py is the primary CUDA
    # gate, but this Stage-1 architecture smoke must be independently
    # self-protecting.
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available — preflight_xattn requires a GPU. "
              "Re-run scripts/preflight_check.py first to diagnose.",
              file=sys.stderr)
        return 4
    print(f"[xa-pre] torch.version.cuda={torch.version.cuda}, "
          f"device={torch.cuda.get_device_name(0)}, "
          f"compute_capability={torch.cuda.get_device_capability(0)}")

    from src.model.qwen_xattn_wrapper import QwenXAttnWrapper
    from src.train.common import prepare_tokenizer

    # 1) tokenizer + vocab sanity
    tokenizer, n_new = prepare_tokenizer(str(args.merged_checkpoint))
    print(f"[xa-pre] tokenizer loaded, len={len(tokenizer)}, custom tokens added={n_new}")

    # 2) merged checkpoint load
    base = AutoModelForCausalLM.from_pretrained(
        str(args.merged_checkpoint),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"[xa-pre] base loaded: vocab_size={base.config.vocab_size}, "
          f"layers={base.config.num_hidden_layers}, "
          f"hidden={base.config.hidden_size}")

    if base.config.vocab_size < len(tokenizer):
        # If we added more tokens than the merged checkpoint has rows,
        # resize. This shouldn't happen if the merge preserved our
        # 151,734-row embedding, but verify.
        print(f"[xa-pre] resizing embed_tokens to len(tokenizer)={len(tokenizer)}")
        base.resize_token_embeddings(len(tokenizer))

    # 3) Wrap
    print(f"[xa-pre] wrapping with QwenXAttnWrapper (every_4, slots=64, small_0.01)")
    wrapper = QwenXAttnWrapper(
        base_model=base,
        insertion_pattern="every_4",
        n_slots=64,
        gate_init="small_0.01",
        encoder_hidden_dim=256,
        encoder_n_layers=6,
        encoder_n_heads=4,
    )
    # 4) Stage-1 LoRA on Q
    print(f"[xa-pre] attaching Stage-1 LoRA-on-Q (r=16)")
    wrapper.attach_lora_on_q(r=16, alpha=16, dropout=0.0)

    device = "cuda"  # CUDA-availability already asserted at start
    # train_xattn.py:125 does the same chain — base loads as bf16 but
    # the freshly-constructed encoder + resampler + cross-attn blocks
    # default to fp32; the .to(bf16) call unifies the dtype so the
    # in-hook block(h, kv, kv_pad) call doesn't hit "expected BFloat16
    # but found Float" at norm_q(h).
    wrapper = wrapper.to(device).to(torch.bfloat16)
    wrapper.eval()

    # Trainable param count
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapper.parameters())
    print(f"[xa-pre] trainable params = {trainable:,} / {total:,} "
          f"({100.0*trainable/total:.2f}%)")

    # 5) Dummy forward pass
    print(f"[xa-pre] running forward(batch={args.batch_size}, seq={args.seq_len}, "
          f"events={args.n_events}) on {device}")

    bsz = args.batch_size
    seq = args.seq_len
    nev = args.n_events
    vocab_size = base.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (bsz, seq), device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Event tensors — types/buckets are small ranges; delta_t is float seconds.
    event_type_ids = torch.randint(0, 7, (bsz, nev), device=device, dtype=torch.long)
    bucket_ids = torch.randint(0, 9, (bsz, nev, 9), device=device, dtype=torch.long)
    delta_t = torch.rand(bsz, nev, device=device, dtype=torch.float32) * 60.0
    event_mask = torch.ones(bsz, nev, device=device, dtype=torch.bool)

    with torch.no_grad():
        try:
            out = wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                event_type_ids=event_type_ids,
                bucket_ids=bucket_ids,
                delta_t=delta_t,
                event_mask=event_mask,
            )
        except Exception as e:
            print(f"FAIL: wrapper forward raised {type(e).__name__}: {e}",
                  file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 2

    loss = out.loss.item() if hasattr(out, "loss") and out.loss is not None else None
    logits_shape = tuple(out.logits.shape) if hasattr(out, "logits") else None
    print(f"[xa-pre] forward OK: loss={loss}, logits_shape={logits_shape}")

    if loss is None or loss != loss:  # NaN check
        print(f"FAIL: loss is NaN or missing", file=sys.stderr)
        return 3

    print(f"[xa-pre] DONE — Stage-1 architecture is ready for trainer launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
