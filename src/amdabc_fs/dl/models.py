"""DL baselines for binary opcode-presence vectors (PyTorch).

The data are presence/absence tables, not opcode sequences, so all four
models are tabular models:
  DL-MLP        two hidden layers (128, 64), dropout 0.2
  CNN1D         1-D convolutions over the feature vector in column order
                (opcode names sorted, so related opcodes such as invoke-* are adjacent)
  BiLSTM        bidirectional LSTM over the features as a sequence of tokens
  PatchTransformer  Transformer encoder over patches of 8 consecutive features
                (one token per patch, [CLS] token); default Transformer baseline
  FTTransformer feature tokenizer + Transformer encoder with a [CLS] token
                (Gorishniy et al., 2021), every feature is one token; ~30x slower
Training: Adam, binary cross-entropy, early stopping on a stratified validation
split taken from the training fold only.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn


class _Tok(nn.Module):
    """value_j * W_j + b_j : one d-dim token per feature."""
    def __init__(self, n, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n, d))

    def forward(self, x):                      # (B, n) -> (B, n, d)
        return x.unsqueeze(-1) * self.w + self.b


class MLPNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


class CNN1DNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.f(x.unsqueeze(1))).squeeze(-1)


class BiLSTMNet(nn.Module):
    def __init__(self, n, d=16, h=32):
        super().__init__()
        self.tok = _Tok(n, d)
        self.rnn = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * h, 1))

    def forward(self, x):
        out, _ = self.rnn(self.tok(x))
        return self.head(out.mean(1)).squeeze(-1)


class FTTransformerNet(nn.Module):
    def __init__(self, n, d=32, heads=4, layers=2):
        super().__init__()
        self.tok = _Tok(n, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):
        t = self.tok(x)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        return self.head(self.enc(t)[:, 0]).squeeze(-1)


class PatchTransformerNet(nn.Module):
    """Transformer over groups of features: the vector is cut into patches of
    `patch` consecutive features (columns are sorted by opcode name, so a patch
    holds related opcodes), each patch is one token (linear projection +
    position embedding), plus a [CLS] token. Attention cost grows with
    (D / patch)^2 instead of D^2 as in FT-Transformer."""
    def __init__(self, n, patch=8, d=64, heads=4, layers=2):
        super().__init__()
        self.patch = patch
        self.n_tok = -(-n // patch)                      # ceil
        self.pad = self.n_tok * patch - n
        self.proj = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.randn(1, self.n_tok + 1, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        if self.pad:
            x = nn.functional.pad(x, (0, self.pad))
        t = self.proj(x.view(len(x), self.n_tok, self.patch))
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.head(self.enc(t)[:, 0]).squeeze(-1)


NETS = {"DL-MLP": MLPNet, "CNN1D": CNN1DNet, "BiLSTM": BiLSTMNet, "FTTransformer": FTTransformerNet,
        "PatchTransformer": PatchTransformerNet}


class DLClassifier:
    """sklearn-like wrapper: fit / predict / predict_proba."""

    def __init__(self, name, epochs=100, patience=10, batch_size=128, lr=1e-3,
                 val_fraction=0.1, device="auto", seed=42):
        self.name, self.epochs, self.patience = name, epochs, patience
        self.bs, self.lr, self.val_fraction, self.seed = batch_size, lr, val_fraction, seed
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    def _t(self, X):
        return torch.as_tensor(np.asarray(X, np.float32), device=self.device)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        tr, va = train_test_split(np.arange(len(y)), test_size=self.val_fraction,
                                  stratify=y, random_state=self.seed)
        Xt, yt = self._t(X[tr]), torch.as_tensor(y[tr], dtype=torch.float32, device=self.device)
        Xv, yv = self._t(X[va]), torch.as_tensor(y[va], dtype=torch.float32, device=self.device)
        self.net = NETS[self.name](X.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=g).to(self.device)
            for i in range(0, len(perm), self.bs):
                b = perm[i:i + self.bs]
                opt.zero_grad()
                loss_fn(self.net(Xt[b]), yt[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = sum(loss_fn(self.net(Xv[i:i + 256]), yv[i:i + 256]).item() * len(yv[i:i + 256])
                        for i in range(0, len(yv), 256)) / len(yv)
            if v < best - 1e-5:
                best, best_state, wait = v, copy.deepcopy(self.net.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                out.append(torch.sigmoid(self.net(self._t(X[i:i + 256]))).cpu().numpy())
        p = np.concatenate(out)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
