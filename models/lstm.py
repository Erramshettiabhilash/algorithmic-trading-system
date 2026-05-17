"""PyTorch LSTM time-series forecasting model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from models.research import ModelPrediction
from models.sequences import SequenceDataset, create_sequence_dataset

LSTMObjective = Literal["regression", "classification"]


@dataclass(frozen=True)
class LSTMModelConfig:
    """Configuration for an LSTM forecasting model."""

    sequence_length: int = 20
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 0.001
    epochs: int = 20
    batch_size: int = 32
    objective_type: LSTMObjective = "regression"
    random_state: int = 42
    device: str = "cpu"


@dataclass(frozen=True)
class LSTMTrainingHistory:
    """Loss history produced by LSTM training."""

    losses: list[float]


class LSTMForecaster:
    """Sequence model for return or direction forecasting."""

    def __init__(self, config: LSTMModelConfig | None = None) -> None:
        self.config = config or LSTMModelConfig()
        self.network: Any | None = None
        self.feature_names: list[str] = []
        self.target_name = "target"
        self.history = LSTMTrainingHistory(losses=[])

    def fit(self, features: pd.DataFrame, target: pd.Series) -> LSTMForecaster:
        """Train the LSTM on sliding feature windows."""

        torch, nn, optim, data = _torch_modules()
        _set_torch_seed(torch, self.config.random_state)

        dataset = create_sequence_dataset(
            features,
            target,
            sequence_length=self.config.sequence_length,
        )
        self.feature_names = dataset.feature_names
        self.target_name = dataset.target_name
        self.network = _LSTMNetwork(
            input_size=dataset.x.shape[2],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            objective_type=self.config.objective_type,
        ).to(self.config.device)

        x_tensor = torch.tensor(dataset.x, dtype=torch.float32)
        y_tensor = torch.tensor(dataset.y, dtype=torch.float32).reshape(-1, 1)
        tensor_dataset = data.TensorDataset(x_tensor, y_tensor)
        loader = data.DataLoader(
            tensor_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        criterion: Any
        if self.config.objective_type == "classification":
            criterion = nn.BCELoss()
        elif self.config.objective_type == "regression":
            criterion = nn.MSELoss()
        else:
            raise ValueError("objective_type must be 'regression' or 'classification'")

        optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        losses: list[float] = []

        self.network.train()
        for _ in range(self.config.epochs):
            epoch_losses: list[float] = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                optimizer.zero_grad()
                prediction = self.network(batch_x)
                loss = criterion(prediction, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            losses.append(float(np.mean(epoch_losses)))

        self.history = LSTMTrainingHistory(losses=losses)
        return self

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Predict using the fitted LSTM and return timestamp-aligned output."""

        if self.network is None:
            raise ValueError("model must be fit before calling predict")

        empty_target = pd.Series(
            np.zeros(len(features), dtype=np.float32),
            index=features.index,
            name=self.target_name,
        )
        dataset = create_sequence_dataset(
            features.loc[:, self.feature_names],
            empty_target,
            sequence_length=self.config.sequence_length,
        )
        predictions = self.predict_dataset(dataset)
        return ModelPrediction(
            predictions=predictions,
            target_name=self.target_name,
            model_name="lstm_forecaster",
        )

    def predict_dataset(self, dataset: SequenceDataset) -> pd.Series:
        """Predict from a pre-windowed sequence dataset."""

        if self.network is None:
            raise ValueError("model must be fit before calling predict")

        torch, _, _, _ = _torch_modules()
        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(dataset.x, dtype=torch.float32).to(self.config.device)
            raw = self.network(x_tensor).detach().cpu().numpy().reshape(-1)

        return pd.Series(raw, index=dataset.index, name=f"{self.config.objective_type}_prediction")


class _LSTMNetwork:
    """Internal torch module, created dynamically to keep torch import lazy."""

    def __new__(
        cls,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        objective_type: LSTMObjective,
    ) -> Any:
        torch, nn, _, _ = _torch_modules()

        class TorchLSTMNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                effective_dropout = dropout if num_layers > 1 else 0.0
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=effective_dropout,
                    batch_first=True,
                )
                self.output = nn.Linear(hidden_size, 1)
                if objective_type == "classification":
                    self.activation = nn.Sigmoid()
                else:
                    self.activation = nn.Identity()

            def forward(self, x: Any) -> Any:
                sequence_output, _ = self.lstm(x)
                last_hidden_state = sequence_output[:, -1, :]
                return self.activation(self.output(last_hidden_state))

        return TorchLSTMNetwork()


def _torch_modules() -> tuple[Any, Any, Any, Any]:
    """Import torch modules lazily with a clear install message."""

    try:
        import torch
        from torch import nn, optim
        from torch.utils import data
    except ImportError as exc:
        raise ImportError(
            "Install torch to train LSTMForecaster: python -m pip install torch"
        ) from exc

    return torch, nn, optim, data


def _set_torch_seed(torch: Any, random_state: int) -> None:
    """Set deterministic seed for small reproducible research experiments."""

    torch.manual_seed(random_state)
