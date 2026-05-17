# Step 6 - LSTM Time-Series Model

## Goal

Build a deep learning model that learns from sequences of past feature rows instead of treating
each timestamp as an independent observation.

## Why Sequential Learning Matters

Markets are path-dependent. A volatility spike after a calm period is not the same as a volatility
spike after weeks of stress. LSTMs can learn from the order of observations inside a window, which
can help when the pattern is not fully captured by one row of engineered features.

## Implemented Components

- `models.sequences.create_sequence_dataset`: converts feature rows into sliding windows.
- `models.lstm.LSTMModelConfig`: stores architecture and training settings.
- `models.lstm.LSTMForecaster`: PyTorch LSTM wrapper for regression and classification.
- `models.lstm.LSTMTrainingHistory`: stores training losses.

## Sliding Window Design

For a sequence length of 20, each sample looks like this:

```text
features[t-19], features[t-18], ..., features[t] -> target[t]
```

That means the prediction at timestamp `t` uses only information available through `t`.

## Regression LSTM

```python
from models import LSTMForecaster, LSTMModelConfig

model = LSTMForecaster(
    LSTMModelConfig(
        sequence_length=20,
        hidden_size=32,
        epochs=20,
        objective_type="regression",
    )
)

model.fit(x_train, y_train)
predictions = model.predict(x_test).predictions
```

Regression is used when we want to forecast return magnitude.

## Classification LSTM

```python
model = LSTMForecaster(
    LSTMModelConfig(
        sequence_length=20,
        hidden_size=32,
        epochs=20,
        objective_type="classification",
    )
)
```

Classification outputs probabilities between 0 and 1.

## XGBoost vs LSTM

XGBoost is usually stronger for tabular factor datasets with limited history. It is easier to tune,
faster to train, and more interpretable.

LSTMs are useful when the sequence itself matters: volatility clustering, trend transitions,
drawdown recovery, regime shifts, or multi-step temporal patterns.

## Overfitting Risks

Deep learning models can memorize noisy market history. Risk controls include:

- small architectures first
- chronological validation
- early stopping in later versions
- limited feature sets
- walk-forward testing
- comparing against simple baselines

## Interview-Ready Explanation

I built the LSTM as a sequence model that consumes rolling windows of engineered features. Unlike
XGBoost, which sees one feature row at a time, the LSTM can learn temporal memory inside the window.
That can help with path-dependent patterns such as volatility clustering and trend transitions, but
it also increases overfitting risk, so I evaluate it chronologically and compare it directly against
the XGBoost factor model.
