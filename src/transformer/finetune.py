"""Fine-tuning and zero-shot inference for the Step 2 transformer models.

Probability arrays always use the canonical column order [negative, neutral, positive].
"""
from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score

MODEL_REGISTRY = {
    "bert_base": "google-bert/bert-base-uncased",
    "finbert_mlm": "yiyanghkust/finbert-pretrain",
}
ZERO_SHOT_REGISTRY = {"prosus_finbert": "ProsusAI/finbert"}
MODEL_REVISIONS = {
    "google-bert/bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "yiyanghkust/finbert-pretrain": "88ab954a39ea6d3ce2b62cff086dd5ad1172c664",
    "ProsusAI/finbert": "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031",
}

CANONICAL_LABELS = ("negative", "neutral", "positive")


@dataclass
class FinetuneConfig:
    max_length: int = 128
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 5
    patience: int = 2
    grad_clip: float = 1.0
    seed: int = 42
    device: Optional[str] = None  # None = auto: cuda > mps > cpu

    def to_dict(self) -> dict:
        return asdict(self)


def select_device(preferred: Optional[str] = None) -> "torch.device":
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "manual_seed") and torch.backends.mps.is_available():
        mps.manual_seed(seed)


def _build_tokenizer(model_name_or_path: str, revision: Optional[str]):
    from transformers import AutoTokenizer, BertTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
    except Exception:
        # Repos with only vocab.txt (e.g. yiyanghkust/finbert-pretrain): the vocab is uncased.
        if os.path.isdir(model_name_or_path):
            vocab_file = os.path.join(model_name_or_path, "vocab.txt")
        else:
            from huggingface_hub import hf_hub_download

            vocab_file = hf_hub_download(model_name_or_path, "vocab.txt", revision=revision)
        return BertTokenizerFast(vocab_file=vocab_file, do_lower_case=True)


def load_model_and_tokenizer(model_name_or_path: str, num_labels: int = 3,
                             revision: Optional[str] = None):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels, revision=revision)
    tokenizer = _build_tokenizer(model_name_or_path, revision)
    return model, tokenizer


def _load_zero_shot(model_name_or_path: str, revision: Optional[str]):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, revision=revision)
    tokenizer = _build_tokenizer(model_name_or_path, revision)
    return model, tokenizer


def predict_proba(model, tokenizer, texts: List[str], config: FinetuneConfig, device) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), config.eval_batch_size):
            batch = list(texts[start:start + config.eval_batch_size])
            enc = tokenizer(batch, padding=True, truncation=True, max_length=config.max_length,
                            return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _macro_f1(y_true, prob: np.ndarray) -> float:
    return float(f1_score(y_true, prob.argmax(axis=1), labels=[0, 1, 2], average="macro",
                          zero_division=0))


def _make_optimizer(model, config: FinetuneConfig):
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    decay_params, plain_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (plain_params if any(nd in name for nd in no_decay) else decay_params).append(param)
    groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": plain_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def finetune_and_predict(model_name_or_path: str, train_texts, train_labels, val_texts, val_labels,
                         eval_sets: Dict[str, List[str]], config: FinetuneConfig,
                         revision: Optional[str] = None, model_loader=None) -> dict:
    from transformers import get_linear_schedule_with_warmup

    started = time.time()
    device = select_device(config.device)
    set_seed(config.seed)
    loader = model_loader or (lambda name, rev, for_zero_shot: load_model_and_tokenizer(name, 3, rev))
    model, tokenizer = loader(model_name_or_path, revision, False)
    model.to(device)

    train_texts = list(train_texts)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    n_train = len(train_texts)
    steps_per_epoch = math.ceil(n_train / config.batch_size)
    total_steps = steps_per_epoch * config.max_epochs
    optimizer = _make_optimizer(model, config)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.warmup_ratio * total_steps), total_steps)

    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()
        model.train()
        gen = torch.Generator()
        gen.manual_seed(config.seed + epoch)
        order = torch.randperm(n_train, generator=gen).tolist()
        loss_sum, n_batches = 0.0, 0
        for start in range(0, n_train, config.batch_size):
            idx = order[start:start + config.batch_size]
            enc = tokenizer([train_texts[i] for i in idx], padding=True, truncation=True,
                            max_length=config.max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = torch.as_tensor(train_labels[idx], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = model(**enc, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.item())
            n_batches += 1
        val_prob = predict_proba(model, tokenizer, val_texts, config, device)
        val_f1 = _macro_f1(val_labels, val_prob)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(n_batches, 1),
                        "val_macro_f1": val_f1, "seconds": time.time() - epoch_start})
        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state)
    val_prob = predict_proba(model, tokenizer, val_texts, config, device)
    eval_probs = {name: predict_proba(model, tokenizer, texts, config, device)
                  for name, texts in eval_sets.items()}
    return {
        "best_epoch": best_epoch,
        "history": history,
        "val_prob": val_prob,
        "eval_probs": eval_probs,
        "wall_seconds": time.time() - started,
        "device": str(device),
        "config": copy.deepcopy(config.to_dict()),
    }


def canonical_label_permutation(id2label: dict) -> List[int]:
    names = {int(k): str(v).strip().lower() for k, v in id2label.items()}
    if sorted(names.values()) != sorted(CANONICAL_LABELS) or len(names) != 3:
        raise ValueError(
            "id2label must name exactly negative/neutral/positive, got %r" % (id2label,))
    by_name = {name: col for col, name in names.items()}
    return [by_name[name] for name in CANONICAL_LABELS]


def zero_shot_predict(model_name_or_path: str, texts: List[str], config: FinetuneConfig,
                      revision: Optional[str] = None, model_loader=None) -> np.ndarray:
    device = select_device(config.device)
    loader = model_loader or (lambda name, rev, for_zero_shot: _load_zero_shot(name, rev))
    model, tokenizer = loader(model_name_or_path, revision, True)
    model.to(device)
    perm = canonical_label_permutation(model.config.id2label)
    prob = predict_proba(model, tokenizer, list(texts), config, device)
    return prob[:, perm]
