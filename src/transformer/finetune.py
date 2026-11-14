"""Fine-tuning and zero-shot inference for the Step 2 transformer models.

Probability arrays always use the canonical column order [negative, neutral, positive].
"""
from __future__ import annotations

import copy
import inspect
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
FINBERT_VOCAB_SIZE = 30873
FINBERT_PROBE_TERMS = ("profit", "ebitda", "depreciation")
TOKENIZER_CHECK_NAME = "tokenizer_check.json"


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


def _vocab_size_matches(tokenizer, expected_vocab_size: Optional[int]) -> bool:
    if expected_vocab_size is None:
        return True
    return len(tokenizer) == expected_vocab_size


def _bert_tokenizer_from_vocab_file(vocab_file: str):
    """Load the uncased BERT WordPiece file with either tokenizer argument name.

    Transformers 4 accepts ``vocab_file``. Transformers 5 renamed it to ``vocab`` and
    ignores ``vocab_file``, which otherwise builds a five-token tokenizer.
    """
    from transformers import BertTokenizerFast

    parameters = inspect.signature(BertTokenizerFast.__init__).parameters
    kwargs = {"do_lower_case": True}
    if "vocab_file" in parameters:
        kwargs["vocab_file"] = vocab_file
    elif "vocab" in parameters:
        kwargs["vocab"] = vocab_file
    else:
        raise RuntimeError(
            f"BertTokenizerFast cannot load a vocab file. Tried to read {vocab_file}."
        )
    return BertTokenizerFast(**kwargs)


def _resolve_vocab_file(model_name_or_path: str, revision: Optional[str]) -> str:
    if os.path.isdir(model_name_or_path):
        return os.path.join(model_name_or_path, "vocab.txt")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_name_or_path, "vocab.txt", revision=revision)


def _build_tokenizer(model_name_or_path: str, revision: Optional[str],
                     expected_vocab_size: Optional[int] = None):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
    except Exception:
        tokenizer = None
    if tokenizer is not None and _vocab_size_matches(tokenizer, expected_vocab_size):
        return tokenizer
    vocab_file = _resolve_vocab_file(model_name_or_path, revision)
    if not os.path.isfile(vocab_file):
        raise RuntimeError(
            f"No usable tokenizer for {model_name_or_path}: automatic loading "
            f"produced vocab size {None if tokenizer is None else len(tokenizer)}, "
            f"expected {expected_vocab_size}, and {vocab_file} is missing."
        )
    tokenizer = _bert_tokenizer_from_vocab_file(vocab_file)
    if not _vocab_size_matches(tokenizer, expected_vocab_size):
        raise RuntimeError(
            f"Tokenizer for {model_name_or_path} has vocab size {len(tokenizer)}, "
            f"but the checkpoint expects {expected_vocab_size}. "
            "Refusing to train with a tokenizer that does not match the embedding matrix."
        )
    return tokenizer


def _requires_finbert_probe(model_name_or_path: str, vocab_size: int) -> bool:
    """The hub id, or any checkpoint whose embedding matrix is the FinBERT vocabulary."""
    name = str(model_name_or_path).replace("\\", "/").rstrip("/")
    if name == "yiyanghkust/finbert-pretrain" or name.endswith("/yiyanghkust/finbert-pretrain"):
        return True
    return int(vocab_size) == FINBERT_VOCAB_SIZE


def _term_ids(tokenizer, term: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(term, add_special_tokens=False)]


def _term_is_unk(tokenizer, token_ids: list[int]) -> bool:
    if not token_ids:
        return True
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is None:
        return False
    return all(int(token_id) == int(unk) for token_id in token_ids)


def validate_finbert_tokenizer(tokenizer) -> dict:
    """Refuse a FinBERT tokenizer whose vocab is not 30,873 or whose probe terms are all [UNK]."""
    size = len(tokenizer)
    if size != FINBERT_VOCAB_SIZE:
        raise RuntimeError(
            f"FinBERT tokenizer vocabulary size is {size}, expected {FINBERT_VOCAB_SIZE}."
        )
    terms = {term: _term_ids(tokenizer, term) for term in FINBERT_PROBE_TERMS}
    if all(_term_is_unk(tokenizer, terms[term]) for term in FINBERT_PROBE_TERMS):
        raise RuntimeError(
            "FinBERT tokenizer maps profit, ebitda, and depreciation to [UNK]. "
            "Refusing to train with the broken vocabulary."
        )
    unk = getattr(tokenizer, "unk_token_id", None)
    return {
        "vocab_size": size,
        "unk_token_id": None if unk is None else int(unk),
        "terms": terms,
        "valid": True,
    }


def tokenizer_check_payload_is_valid(payload: object) -> bool:
    """True when a saved receipt records the real vocabulary and not an all-[UNK] probe."""
    if not isinstance(payload, dict) or payload.get("valid") is not True:
        return False
    if payload.get("vocab_size") != FINBERT_VOCAB_SIZE:
        return False
    terms = payload.get("terms")
    if not isinstance(terms, dict):
        return False
    unk = payload.get("unk_token_id")
    broken = 0
    for term in FINBERT_PROBE_TERMS:
        token_ids = terms.get(term)
        if not isinstance(token_ids, list) or not token_ids:
            broken += 1
            continue
        if unk is not None and all(int(token_id) == int(unk) for token_id in token_ids):
            broken += 1
    return broken < len(FINBERT_PROBE_TERMS)


def prepare_finbert_tokenizer_check(model_name_or_path: str, revision: Optional[str] = None) -> dict:
    """Load the FinBERT tokenizer and return a receipt. Raises before any training step."""
    tokenizer = _build_tokenizer(
        model_name_or_path, revision, expected_vocab_size=FINBERT_VOCAB_SIZE)
    return validate_finbert_tokenizer(tokenizer)


def injected_finbert_tokenizer_check() -> dict:
    """Receipt for a test double that never loads yiyanghkust/finbert-pretrain."""
    return {
        "vocab_size": FINBERT_VOCAB_SIZE,
        "unk_token_id": 1,
        "terms": {"profit": [10], "ebitda": [11], "depreciation": [12]},
        "valid": True,
    }


def _is_legacy_bert_checkpoint(config_dict: dict) -> bool:
    """True for a BERT checkpoint whose config.json omits model_type.

    yiyanghkust/finbert-pretrain is one of these: architectures is BertForMaskedLM,
    but Transformers 5 refuses AutoModel without model_type.
    """
    if config_dict.get("model_type"):
        return False
    architectures = config_dict.get("architectures") or []
    return any(str(name).startswith("Bert") for name in architectures)


def _load_legacy_bert_for_classification(model_name_or_path: str, config_dict: dict,
                                         num_labels: int, revision: Optional[str]):
    """Load a legacy BERT checkpoint into the three-class classification model.

    BertConfig.from_dict only records the architecture. BertForSequenceClassification.from_pretrained
    then copies the checkpoint tensors onto that module. Constructing the class from the config
    alone would leave the encoder randomly initialized, so this function never does that.
    The masked-LM head is not part of the classifier; the three-class head stays newly initialized.
    """
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig.from_dict(config_dict)
    if not config.model_type:
        config.model_type = "bert"
    config.num_labels = num_labels
    return BertForSequenceClassification.from_pretrained(
        model_name_or_path, config=config, revision=revision)


def load_model_and_tokenizer(model_name_or_path: str, num_labels: int = 3,
                             revision: Optional[str] = None):
    from transformers import AutoModelForSequenceClassification, PretrainedConfig

    config_dict, _unused = PretrainedConfig.get_config_dict(model_name_or_path, revision=revision)
    if _is_legacy_bert_checkpoint(config_dict):
        model = _load_legacy_bert_for_classification(
            model_name_or_path, config_dict, num_labels, revision)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels, revision=revision)
    tokenizer = _build_tokenizer(
        model_name_or_path, revision, expected_vocab_size=model.config.vocab_size)
    if _requires_finbert_probe(model_name_or_path, model.config.vocab_size):
        validate_finbert_tokenizer(tokenizer)
    return model, tokenizer


def _load_zero_shot(model_name_or_path: str, revision: Optional[str]):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, revision=revision)
    tokenizer = _build_tokenizer(
        model_name_or_path, revision, expected_vocab_size=model.config.vocab_size)
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
            if not torch.isfinite(loss).item():
                # Seen on Apple MPS under memory pressure ("command buffer exited with error status /
                # Insufficient Memory"): kernels are skipped silently and the loss turns NaN. Fail the run
                # loudly instead of returning a degenerate model as "completed".
                raise RuntimeError(f"non-finite training loss at epoch {epoch} (device {device})")
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
