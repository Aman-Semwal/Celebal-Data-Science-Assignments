"""Attention-LSTM demand forecaster (PyTorch).

Design (consistent with the horizon-safe design used by the tree models):
    - Input: the last ``sequence_length`` days of sales ending at the
      as-of date (``target_date - horizon_days``), per (store, item)
      series, plus static store/item embeddings.
    - Architecture: an embedding layer for store and item ids, a
      multi-layer LSTM over the sales sequence, an additive attention
      layer that learns to weight which past time steps matter most,
      and a small feed-forward head that combines the attended context
      vector with the store/item embeddings to output a single
      predicted value (the sales on ``target_date``).
    - This is trained as a direct 90-day-ahead point forecaster, exactly
      like the tree models, so all models are compared on the same task.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import FORECAST, MODEL_DEFAULTS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

torch.manual_seed(FORECAST.random_seed)


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention pooling over an LSTM's output sequence."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, lstm_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: Tensor of shape (batch, seq_len, hidden_size).

        Returns:
            context: Tensor of shape (batch, hidden_size), the attention-weighted sum.
            weights: Tensor of shape (batch, seq_len), the attention weights (sum to 1).
        """
        scores = self.score_layer(lstm_outputs).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)  # (batch, hidden)
        return context, weights


class AttentionLSTMNet(nn.Module):
    """LSTM + additive attention + store/item embeddings -> single sales prediction."""

    def __init__(
        self,
        n_stores: int,
        n_items: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.store_embedding = nn.Embedding(n_stores, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size + 2 * embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, sequence: torch.Tensor, store_idx: torch.Tensor, item_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sequence: (batch, seq_len, 1) past sales values.
            store_idx: (batch,) integer store codes.
            item_idx: (batch,) integer item codes.

        Returns:
            (batch,) predicted sales for the target day.
        """
        lstm_out, _ = self.lstm(sequence)  # (batch, seq_len, hidden)
        context, _ = self.attention(lstm_out)

        store_emb = self.store_embedding(store_idx)
        item_emb = self.item_embedding(item_idx)
        combined = torch.cat([context, store_emb, item_emb], dim=1)

        out = self.head(combined).squeeze(-1)
        return torch.relu(out)  # sales are non-negative


def _build_sequences(
    history_df: pd.DataFrame, target_df: pd.DataFrame, sequence_length: int, horizon_days: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized construction of (sequence, store_idx, item_idx, target) arrays.

    For each row of ``target_df`` (with columns date/store/item/sales), build
    the ``sequence_length``-day window of sales ending at
    ``target_date - horizon_days`` (the as-of date), pulled from
    ``history_df`` (must cover a wider date range than ``target_df``).
    """
    date_col, store_col, item_col, target_col = (
        FORECAST.date_col,
        FORECAST.store_col,
        FORECAST.item_col,
        FORECAST.target_col,
    )

    # Pivot history into a (store, item) -> dense daily sales array + date index for O(1) slicing.
    history_df = history_df.sort_values([store_col, item_col, date_col])
    series_map: dict[tuple, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    for (store, item), group in history_df.groupby([store_col, item_col]):
        series_map[(store, item)] = (
            group[date_col].to_numpy(),
            group[target_col].to_numpy(dtype="float32"),
        )

    sequences, store_idxs, item_idxs, targets = [], [], [], []
    for row in target_df.itertuples(index=False):
        store, item = getattr(row, store_col), getattr(row, item_col)
        target_date = getattr(row, date_col)
        as_of_date = target_date - pd.Timedelta(days=horizon_days)

        dates, sales = series_map.get((store, item), (None, None))
        if dates is None:
            continue
        end_pos = np.searchsorted(dates, np.datetime64(as_of_date), side="right")
        start_pos = end_pos - sequence_length
        if start_pos < 0:
            continue

        seq = sales[start_pos:end_pos]
        sequences.append(seq)
        store_idxs.append(store)
        item_idxs.append(item)
        targets.append(getattr(row, target_col))

    return (
        np.stack(sequences).astype("float32"),
        np.array(store_idxs, dtype="int64"),
        np.array(item_idxs, dtype="int64"),
        np.array(targets, dtype="float32"),
    )


class AttentionLSTMModel:
    """Sklearn-style wrapper around ``AttentionLSTMNet`` for training/inference."""

    name = "attention_lstm"

    def __init__(self, **override_params) -> None:
        self.params = {**MODEL_DEFAULTS.attention_lstm, **override_params}
        self.net: AttentionLSTMNet | None = None
        self.store_codes: dict = {}
        self.item_codes: dict = {}
        self.sales_mean: float = 0.0
        self.sales_std: float = 1.0

    def _encode_ids(
        self, stores: np.ndarray, items: np.ndarray, fit: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        if fit:
            self.store_codes = {s: i for i, s in enumerate(sorted(set(stores.tolist())))}
            self.item_codes = {s: i for i, s in enumerate(sorted(set(items.tolist())))}
        store_idx = np.array([self.store_codes.get(s, 0) for s in stores], dtype="int64")
        item_idx = np.array([self.item_codes.get(s, 0) for s in items], dtype="int64")
        return store_idx, item_idx

    def fit(
        self,
        history_df: pd.DataFrame,
        train_target_df: pd.DataFrame,
        valid_target_df: pd.DataFrame | None = None,
    ):
        """Fit the Attention-LSTM.

        Args:
            history_df: Full daily (date, store, item, sales) history, used
                as the lookup table to build input sequences.
            train_target_df: Rows to train on (date, store, item, sales) --
                each row's `sales` is the direct-forecast label.
            valid_target_df: Optional validation rows for monitoring loss.
        """
        seq_len = self.params["sequence_length"]
        horizon = FORECAST.horizon_days

        X_seq, store_raw, item_raw, y = _build_sequences(
            history_df, train_target_df, seq_len, horizon
        )
        store_idx, item_idx = self._encode_ids(store_raw, item_raw, fit=True)

        self.sales_mean, self.sales_std = float(y.mean()), float(y.std() + 1e-6)
        X_seq_norm = (X_seq - self.sales_mean) / self.sales_std
        y_norm = (y - self.sales_mean) / self.sales_std

        self.net = AttentionLSTMNet(
            n_stores=len(self.store_codes),
            n_items=len(self.item_codes),
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
        )

        train_ds = TensorDataset(
            torch.tensor(X_seq_norm).unsqueeze(-1),
            torch.tensor(store_idx),
            torch.tensor(item_idx),
            torch.tensor(y_norm),
        )
        train_loader = DataLoader(train_ds, batch_size=self.params["batch_size"], shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params["learning_rate"])
        loss_fn = nn.MSELoss()

        self.net.train()
        for epoch in range(self.params["epochs"]):
            epoch_loss = 0.0
            n_batches = 0
            for seq_batch, store_batch, item_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self.net(seq_batch, store_batch, item_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            logger.info(
                "Attention-LSTM epoch %d/%d — train MSE (normalized): %.4f",
                epoch + 1,
                self.params["epochs"],
                epoch_loss / max(n_batches, 1),
            )

        return self

    def predict(self, history_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
        """Predict sales for each row of ``target_df`` using ``history_df`` as the sequence source."""
        if self.net is None:
            raise RuntimeError("Call fit() before predict().")

        seq_len = self.params["sequence_length"]
        horizon = FORECAST.horizon_days
        X_seq, store_raw, item_raw, _ = _build_sequences(history_df, target_df, seq_len, horizon)
        store_idx, item_idx = self._encode_ids(store_raw, item_raw, fit=False)

        X_seq_norm = (X_seq - self.sales_mean) / self.sales_std

        self.net.eval()
        with torch.no_grad():
            preds_norm = self.net(
                torch.tensor(X_seq_norm).unsqueeze(-1),
                torch.tensor(store_idx),
                torch.tensor(item_idx),
            ).numpy()

        preds = preds_norm * self.sales_std + self.sales_mean
        return np.clip(preds, 0, None)

    def save(self, path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "state_dict": self.net.state_dict(),
                "params": self.params,
                "store_codes": self.store_codes,
                "item_codes": self.item_codes,
                "sales_mean": self.sales_mean,
                "sales_std": self.sales_std,
            },
            path,
        )

    def load(self, path) -> AttentionLSTMModel:
        import joblib

        payload = joblib.load(path)
        self.params = payload["params"]
        self.store_codes = payload["store_codes"]
        self.item_codes = payload["item_codes"]
        self.sales_mean = payload["sales_mean"]
        self.sales_std = payload["sales_std"]
        self.net = AttentionLSTMNet(
            n_stores=len(self.store_codes),
            n_items=len(self.item_codes),
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
        )
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()
        return self
