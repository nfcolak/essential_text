"""Continued MLM pretraining of ``bert-base-uncased`` on financial text (controlled DAPT arm).

The vocabulary, architecture and tokenizer stay those of ``google-bert/bert-base-uncased`` (pinned revision); only
the pretraining text changes. Documents (one per line of ``--corpus``) are tokenized, concatenated and cut into
blocks of ``--seq-len`` word pieces, each wrapped in [CLS] ... [SEP]; 15% of the tokens are masked
(80/10/10, ``DataCollatorForLanguageModeling``). AdamW, linear warmup + decay, gradient clipping 1.0, fp16 autocast on
CUDA. One pass over the blocks unless ``--max-steps`` is smaller.

Runs on CUDA (Kaggle), MPS or CPU. Writes ``<out-dir>/model/`` (weights, config, tokenizer),
``<out-dir>/train_log.json`` and ``<out-dir>/progress.log``; ``--max-steps`` with a small value is the timing probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

BERT = "google-bert/bert-base-uncased"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_blocks(tokenizer, corpus: str, seq_len: int, max_blocks: int | None = None) -> np.ndarray:
    """(n_blocks, seq_len) int32 array of [CLS] + (seq_len - 2) pieces + [SEP]."""
    body = seq_len - 2
    buf: list[int] = []
    blocks: list[np.ndarray] = []
    batch: list[str] = []

    def flush(lines: list[str]) -> None:
        for ids in tokenizer(lines, add_special_tokens=False)["input_ids"]:
            buf.extend(ids)
        while len(buf) >= body:
            blocks.append(np.array([tokenizer.cls_token_id] + buf[:body] + [tokenizer.sep_token_id], dtype=np.int32))
            del buf[:body]

    with open(corpus, encoding="utf-8") as handle:
        for line in handle:
            batch.append(line.rstrip("\n"))
            if len(batch) >= 256:
                flush(batch)
                batch = []
                if max_blocks is not None and len(blocks) >= max_blocks:
                    break
    if batch:
        flush(batch)
    arr = np.stack(blocks) if blocks else np.zeros((0, seq_len), dtype=np.int32)
    return arr[:max_blocks] if max_blocks is not None else arr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=BERT)
    parser.add_argument("--revision", default=BERT_REVISION)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=None, help="cap (probe); default = one pass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    import torch
    from transformers import (BertForPreTraining, BertTokenizerFast, DataCollatorForLanguageModeling,
                              get_linear_schedule_with_warmup)

    os.makedirs(args.out_dir, exist_ok=True)
    progress = open(os.path.join(args.out_dir, "progress.log"), "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        progress.write(line + "\n")
        progress.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    tokenizer = BertTokenizerFast.from_pretrained(args.model, revision=args.revision)
    t0 = time.time()
    max_blocks = None if args.max_steps is None else args.max_steps * args.batch_size
    blocks = build_blocks(tokenizer, args.corpus, args.seq_len, max_blocks)
    n_blocks = len(blocks)
    steps_per_pass = n_blocks // args.batch_size
    total_steps = min(steps_per_pass, args.max_steps) if args.max_steps else steps_per_pass
    log(f"device={device} corpus_sha256={sha256(args.corpus)} blocks={n_blocks} tokens={n_blocks * args.seq_len} "
        f"steps={total_steps} tokenize_s={time.time() - t0:.1f} torch={torch.__version__}")

    # BertForPreTraining (not BertForMaskedLM) keeps the pretrained pooler in the saved checkpoint: the sequence
    # classifier of the fine-tuning recipe starts from that pooler, as it does for bert-base-uncased. Only the MLM
    # loss is trained (no next-sentence loss), so the pooler and NSP head keep their pretrained values.
    model = BertForPreTraining.from_pretrained(args.model, revision=args.revision).to(device)
    mlm_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    order = np.random.default_rng(args.seed).permutation(n_blocks)

    history = []
    model.train()
    started = time.time()
    window_loss, window_n = 0.0, 0
    for step in range(total_steps):
        idx = order[step * args.batch_size:(step + 1) * args.batch_size]
        batch = collator([{"input_ids": torch.as_tensor(blocks[i], dtype=torch.long)} for i in idx])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            labels = batch.pop("labels")
            logits = model(**batch).prediction_logits
            loss = mlm_loss(logits.view(-1, logits.size(-1)).float(), labels.view(-1))
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        window_loss += float(loss.item())
        window_n += 1
        if (step + 1) % args.log_every == 0 or step + 1 == total_steps:
            elapsed = time.time() - started
            entry = {"step": step + 1, "loss": window_loss / window_n, "lr": scheduler.get_last_lr()[0],
                     "elapsed_s": elapsed, "s_per_step": elapsed / (step + 1)}
            history.append(entry)
            eta = entry["s_per_step"] * (total_steps - step - 1)
            log(f"step {step + 1}/{total_steps} loss {entry['loss']:.4f} s/step {entry['s_per_step']:.3f} "
                f"eta_min {eta / 60:.1f}")
            window_loss, window_n = 0.0, 0

    model_dir = os.path.join(args.out_dir, "model")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    hashes = {f: sha256(os.path.join(model_dir, f)) for f in sorted(os.listdir(model_dir))}
    summary = {"args": vars(args), "device": str(device), "blocks": n_blocks, "steps": total_steps,
               "tokens_seen": total_steps * args.batch_size * args.seq_len, "corpus_sha256": sha256(args.corpus),
               "train_seconds": time.time() - started, "history": history, "model_files_sha256": hashes,
               "torch": torch.__version__}
    with open(os.path.join(args.out_dir, "train_log.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(f"DONE steps={total_steps} train_s={summary['train_seconds']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
