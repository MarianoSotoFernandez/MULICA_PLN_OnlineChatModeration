

"""
XLM_RoBERTa.py
==============
XLM-RoBERTa-base fine-tuned for multi-label toxic comment classification.

Public API
----------
    train(X_train, Y_train, X_val, Y_val)  --> (model, tokenizer)
    predict(model, tokenizer, X)           --> np.ndarray (N, 6) binary predictions
    load(model_dir)                        --> (model, tokenizer)

Standalone usage
----------------
    from XLM_RoBERTa import train, predict, load
"""

import os
import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.backends.cuda.matmul.allow_tf32 = True    # faster matmuls on Ampere+
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True    # auto-tunes kernels for your input shape


# ─────────────────────────────────────────────────────────────────────────────
#  Defaults & device
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME  = "xlm-roberta-base"
DEFAULT_DIR = "./models/xlmroberta_models"
NUM_LABELS  = 6
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5
WARMUP_RATIO = 0.1
THRESHOLD   = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ToxicDataset(Dataset):
    """
    PyTorch Dataset that tokenizes text on-the-fly.

    Args:
        texts:     list of cleaned strings
        labels:    float32 array (N, 6); pass zeros for inference
        tokenizer: HuggingFace tokenizer
        max_len:   maximum token length (longer inputs are truncated)
    """

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer,
        max_len: int = MAX_LEN,
    ):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Model architecture
# ─────────────────────────────────────────────────────────────────────────────

class XLMRobertaToxicClassifier(nn.Module):
    """
    XLM-RoBERTa-base backbone with a two-layer classification head.

    Architecture:
        XLM-R encoder
            - [CLS] token  (hidden size 768)
            - Dropout(0.1)
            - Linear(768 --> 768) + GELU
            - Dropout(0.1)
            - Linear(768 --> 6)   (raw logits, one per label)

    Loss: BCEWithLogitsLoss (internal sigmoid - Not here).
    """

    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(model_name)
        hidden          = self.backbone.config.hidden_size   # 768 for base
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out     = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_tok = out.last_hidden_state[:, 0, :]   # [CLS] representation
        return self.classifier(cls_tok)             # raw logits, shape (B, 6)


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(
    texts: list[str],
    labels: np.ndarray,
    tokenizer,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ToxicDataset(texts, labels, tokenizer),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )


def _val_f1(
    model: XLMRobertaToxicClassifier,
    loader: DataLoader,
    threshold: float = THRESHOLD,
) -> float:
    """Compute macro F1 on a validation DataLoader without updating weights."""
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            probs = torch.sigmoid(model(ids, mask)).cpu().numpy()
            preds_list.append((probs >= threshold).astype(int))
            labels_list.append(batch["labels"].numpy())

    return f1_score(
        np.vstack(labels_list),
        np.vstack(preds_list),
        average="macro",
        zero_division=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────────────────

def train(
    X_train: list[str],
    Y_train: np.ndarray,
    X_val:   list[str],
    Y_val:   np.ndarray,
    model_dir: str = DEFAULT_DIR,
) -> tuple[XLMRobertaToxicClassifier, AutoTokenizer]:
    """
    Fine-tune XLM-RoBERTa-base. Saves the best checkpoint (by val macro F1).

    Args:
        X_train / Y_train: training texts and labels (N, 6)
        X_val   / Y_val:   validation texts and labels - used only for
                           checkpoint selection, never for gradient updates
        model_dir:         directory to save weights + tokenizer

    Returns:
        (model, tokenizer) - best checkpoint already loaded into model.
    """
    print(f"\n  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = _make_loader(X_train, Y_train, tokenizer, BATCH_SIZE, shuffle=True)
    val_loader   = _make_loader(X_val,   Y_val,   tokenizer, BATCH_SIZE * 2, shuffle=False)

    model = XLMRobertaToxicClassifier().to(DEVICE)

    # Differential learning rates: lower LR for backbone, higher for fresh head
    optimizer = AdamW([
        {"params": model.backbone.parameters(),   "lr": LR},
        {"params": model.classifier.parameters(), "lr": LR * 10},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Weighted BCE: upweights positive labels (rare in the Jigsaw dataset)
    pos_weight = torch.tensor(
        [(Y_train[:, i] == 0).sum() / max((Y_train[:, i] == 1).sum(), 1)
         for i in range(NUM_LABELS)],
        dtype=torch.float,
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    os.makedirs(model_dir, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(ids, mask), lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            if step % 100 == 0:
                print(f"    Epoch {epoch}/{EPOCHS} | step {step}/{len(train_loader)} "
                      f"| loss {running_loss/step:.4f} "
                      f"| lr {scheduler.get_last_lr()[0]:.2e}")

        val_f1 = _val_f1(model, val_loader)
        print(f"  --> Epoch {epoch} val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            tokenizer.save_pretrained(model_dir)
            print(f"    Best model saved! (F1={best_f1:.4f})")

    # Reload best weights before returning
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "best_model.pt"), map_location=DEVICE)
    )
    print(f"\n  Training complete. Best val Macro F1: {best_f1:.4f}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
#  Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model:     XLMRobertaToxicClassifier,
    tokenizer,
    X:         list[str],
    threshold: float = THRESHOLD,
) -> np.ndarray:
    """
    Generate binary multi-label predictions for a list of texts.

    Args:
        model:     fine-tuned XLMRobertaToxicClassifier (eval mode)
        tokenizer: HuggingFace tokenizer (saved alongside model)
        X:         list of cleaned text strings
        threshold: sigmoid probability cutoff (default 0.3)

    Returns:
        int array of shape (N, 6) - one binary column per label.
    """
    dummy_labels = np.zeros((len(X), NUM_LABELS), dtype=np.float32)
    loader       = _make_loader(X, dummy_labels, tokenizer, BATCH_SIZE * 2, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            probs = torch.sigmoid(model(ids, mask)).cpu().numpy()
            all_preds.append((probs >= threshold).astype(int))

    return np.vstack(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
#  Persistence
# ─────────────────────────────────────────────────────────────────────────────

def load(
    model_dir: str = DEFAULT_DIR,
) -> tuple[XLMRobertaToxicClassifier, AutoTokenizer]:
    """
    Load a fine-tuned checkpoint from disk.

    Args:
        model_dir: directory containing best_model.pt and tokenizer files

    Returns:
        (model, tokenizer) ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = XLMRobertaToxicClassifier()
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "best_model.pt"), map_location=DEVICE)
    )
    model.to(DEVICE).eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
#  Inference wrapper (optional standalone use)
# ─────────────────────────────────────────────────────────────────────────────

class XLMRClassifier:
    """
    Thin inference wrapper for production use.
    Loads a fine-tuned checkpoint from disk on instantiation.
    """

    LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def __init__(
        self,
        model_path: str = DEFAULT_DIR,
        threshold:  float = THRESHOLD,
    ):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"XLM-R model directory not found at '{model_path}'.\n"
                f"Train it first with: python train_and_evaluate.py"
            )
        self.threshold        = threshold
        self.model, self.tokenizer = load(model_path)

    def predict_single(self, message: str) -> dict:
        from utils import clean_text
        cleaned = clean_text(message)
        probs   = predict(self.model, self.tokenizer, [cleaned], self.threshold)[0]
        scores  = dict(zip(self.LABELS, probs.astype(float)))
        labels  = {l: bool(v) for l, v in scores.items()}
        return {
            "message":  message,
            "is_toxic": any(labels.values()),
            "labels":   labels,
        }

    def predict_batch(self, messages: list[str]) -> list[dict]:
        from utils import clean_text
        cleaned = [clean_text(m) for m in messages]
        preds   = predict(self.model, self.tokenizer, cleaned, self.threshold)
        results = []
        for message, pred in zip(messages, preds):
            labels = dict(zip(self.LABELS, pred.astype(bool).tolist()))
            results.append({
                "message":  message,
                "is_toxic": any(labels.values()),
                "labels":   labels,
            })
        return results
