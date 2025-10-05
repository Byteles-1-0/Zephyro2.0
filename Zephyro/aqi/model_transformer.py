"""Transformer model for AQI forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_size: int
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    forecast_horizon: int = 7


class AQITransformer(nn.Module):
    """A lightweight Transformer encoder-decoder for time series forecasting."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.dropout)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder_input = nn.Parameter(torch.zeros(config.forecast_horizon, config.d_model))
        self.output_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        src : torch.Tensor
            Tensor of shape (batch, seq_len, input_size).
        Returns
        -------
        torch.Tensor
            Forecast tensor of shape (batch, forecast_horizon).
        """

        batch_size, seq_len, _ = src.shape
        src_proj = self.input_projection(src)
        src_pe = self.positional_encoding(src_proj)
        tgt_tokens = self.decoder_input.unsqueeze(0).expand(batch_size, -1, -1)
        tgt_pe = self.positional_encoding(tgt_tokens)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.config.forecast_horizon).to(src.device)
        memory = self.transformer.encoder(src_pe)
        decoded = self.transformer.decoder(tgt_pe, memory, tgt_mask=tgt_mask)
        out = self.output_head(decoded)
        return out.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def train_model(
    model: AQITransformer,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    X_train, y_train = train_data
    X_val, y_val = val_data

    history = {"train_loss": [], "val_loss": []}

    best_state = None
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train.to(device))
        loss = criterion(preds, y_train.to(device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(device))
            val_loss = criterion(val_preds, y_val.to(device))

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_val_loss": best_val_loss, "state_dict": model.state_dict()}


def predict(model: AQITransformer, X: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(X.to(device))


__all__ = ["AQITransformer", "ModelConfig", "train_model", "predict"]
