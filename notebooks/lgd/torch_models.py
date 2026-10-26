"""PyTorch models and training loop, running on the Apple Silicon GPU (MPS) when it is available.

Every model defines its own `loss(output, y)` and `expected_value(output)`; for most models the output is a logit
and the prediction is `sigmoid(logit)`, so predictions always lie in [0, 1] (a cure probability or a recovery rate).
"""

import copy
import os
import time

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn

# PyTorch's multi-threaded CPU kernels deadlock on macOS once LightGBM / XGBoost have started their
# own OpenMP thread pools. The networks train on the GPU, so one CPU thread costs nothing.
torch.set_num_threads(1)


def get_device() -> torch.device:
    """MPS (Apple GPU) when available, otherwise CPU. Override with LGD_DEVICE=cpu or LGD_DEVICE=mps."""
    requested = os.environ.get("LGD_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


DEVICE = get_device()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)  # seeds the CPU and the MPS generators


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Fully connected network. `loss='bce'` for probabilities / fractional targets, `'mse'` for regression.

    With `hidden=()` and `loss='bce'` this is a logistic regression, or a fractional logit when the
    target is a rate in [0, 1].
    """

    def __init__(self, n_features: int, hidden=(64, 32, 16), dropout: float = 0.0, loss: str = "bce"):
        super().__init__()
        layers, width = [], n_features
        for units in hidden:
            layers += [nn.Linear(width, units), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = units
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)
        self.loss_name = loss

    def forward(self, x):
        return self.layers(x).squeeze(-1)

    def expected_value(self, logits):
        return torch.sigmoid(logits)

    def loss(self, logits, y):
        if self.loss_name == "bce":
            return nn.functional.binary_cross_entropy_with_logits(logits, y)
        if self.loss_name == "mse":
            return nn.functional.mse_loss(torch.sigmoid(logits), y)
        raise ValueError(f"unknown loss {self.loss_name!r}")


class BetaRegression(nn.Module):
    """Beta regression: logit(mean) is linear in the features, with one shared precision parameter phi.

    Targets of exactly 0 or 1 are moved inside (0, 1) with the Smithson & Verkuilen (2006) transform
    y' = (y (n - 1) + 0.5) / n, where n is the number of training observations.
    """

    def __init__(self, n_features: int, n_observations: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.log_phi = nn.Parameter(torch.tensor(2.0))
        self.n_observations = n_observations

    def forward(self, x):
        return self.linear(x).squeeze(-1)

    def expected_value(self, logits):
        return torch.sigmoid(logits)

    def loss(self, logits, y):
        n = self.n_observations
        y = (y * (n - 1) + 0.5) / n
        mean, phi = torch.sigmoid(logits), torch.exp(self.log_phi)
        beta = torch.distributions.Beta(mean * phi + 1e-6, (1 - mean) * phi + 1e-6)
        return -beta.log_prob(y).mean()


class InflatedBetaRegression(nn.Module):
    """Zero-one inflated beta regression (Ospina & Ferrari, 2010) for rates with point masses at 0 and 1.

    Three outcomes are modelled: y = 0, y = 1, or 0 < y < 1 (multinomial logit), and for the last one a Beta
    distribution with logit(mean) = w·x + b and a shared precision phi. Prediction: E[y] = P(1) + P(0<y<1) · mean.
    With `hidden` layers the linear predictors are replaced by a small network.
    """

    BOUNDARY = 1e-4   # |y - 0| or |y - 1| below this counts as exactly 0 or 1

    def __init__(self, n_features: int, hidden=(), dropout: float = 0.0):
        super().__init__()
        layers, width = [], n_features
        for units in hidden:
            layers += [nn.Linear(width, units), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout > 0 else [])
            width = units
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 4)            # logits of (zero, one, interior) + logit of the Beta mean
        self.log_phi = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return self.head(self.body(x))

    def expected_value(self, output):
        probabilities = torch.softmax(output[:, :3], dim=1)
        return probabilities[:, 1] + probabilities[:, 2] * torch.sigmoid(output[:, 3])

    def loss(self, output, y):
        log_probabilities = torch.log_softmax(output[:, :3], dim=1)
        is_zero, is_one = y <= self.BOUNDARY, y >= 1 - self.BOUNDARY
        mean, phi = torch.sigmoid(output[:, 3]), torch.exp(self.log_phi)
        y_inside = y.clamp(self.BOUNDARY, 1 - self.BOUNDARY)
        beta = torch.distributions.Beta(mean * phi + 1e-6, (1 - mean) * phi + 1e-6)
        log_likelihood = torch.where(is_zero, log_probabilities[:, 0],
                         torch.where(is_one, log_probabilities[:, 1], log_probabilities[:, 2] + beta.log_prob(y_inside)))
        return -log_likelihood.mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def to_tensor(array) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array, dtype=np.float32), device=DEVICE)


def train_network(model: nn.Module, X_train, y_train, X_valid, y_valid, *, learning_rate: float = 1e-3,
                  batch_size: int = 64, max_epochs: int = 200, patience: int = 10, weight_decay: float = 0.0,
                  trial=None, verbose: bool = False) -> dict:
    """Train with Adam and early stopping on the validation loss; the best weights are restored.

    The whole dataset is copied to the GPU once and mini-batches are sliced there, which avoids the
    per-batch CPU -> GPU copies of a DataLoader. Passing an Optuna `trial` reports the validation loss
    every epoch so that unpromising trials are pruned early.
    """
    model.to(DEVICE)
    X_train, y_train, X_valid, y_valid = map(to_tensor, (X_train, y_train, X_valid, y_valid))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = {"loss": [], "val_loss": []}
    best_loss, best_state, epochs_without_improvement = float("inf"), None, 0
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(X_train), device=DEVICE)
        epoch_loss = torch.zeros((), device=DEVICE)
        for i in range(0, len(X_train), batch_size):
            batch = permutation[i:i + batch_size]
            loss = model.loss(model(X_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * len(batch)

        model.eval()
        with torch.no_grad():
            val_loss = model.loss(model(X_valid), y_valid).item()
        history["loss"].append(epoch_loss.item() / len(X_train))
        history["val_loss"].append(val_loss)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"epoch {epoch:>3}  loss={history['loss'][-1]:.5f}  val_loss={val_loss:.5f}")

        if trial is not None:
            import optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_loss < best_loss - 1e-6:
            best_loss, best_state, epochs_without_improvement = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    history.update(best_epoch=int(np.argmin(history["val_loss"])) + 1, best_val_loss=best_loss,
                   train_seconds=time.perf_counter() - start)
    if verbose:
        print(f"stopped after {len(history['loss'])} epochs, best epoch {history['best_epoch']} "
              f"(val_loss={best_loss:.5f}), {history['train_seconds']:.1f}s on {DEVICE}")
    return history


@torch.no_grad()
def predict(model: nn.Module, X, batch_size: int = 8192) -> np.ndarray:
    """The model's expected value as a NumPy array: a probability (cure) or a rate in [0, 1] (recovery)."""
    model.eval()
    X = to_tensor(X)
    outputs = [model.expected_value(model(X[i:i + batch_size])) for i in range(0, len(X), batch_size)]
    return torch.cat(outputs).cpu().numpy()


class TorchRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn style wrapper (fit / predict) so that PyTorch models can be cross-validated.

    `build_model(n_features, n_observations)` returns a fresh network. `fit` Min-Max scales the features,
    keeps `validation_share` of the rows for early stopping and trains with `fit_options`.
    """

    def __init__(self, build_model, fit_options: dict, validation_share: float = 0.15, max_epochs: int = 200,
                 patience: int = 10, seed: int = 42):
        self.build_model = build_model
        self.fit_options = fit_options
        self.validation_share = validation_share
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def fit(self, X, y):
        from sklearn.preprocessing import MinMaxScaler

        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        self.scaler_ = MinMaxScaler().fit(X)
        X = self.scaler_.transform(X)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(X))
        n_valid = max(1, int(self.validation_share * len(X)))
        valid, train = order[:n_valid], order[n_valid:]
        set_seed(self.seed)
        self.model_ = self.build_model(X.shape[1], len(train))
        train_network(self.model_, X[train], y[train], X[valid], y[valid], max_epochs=self.max_epochs,
                      patience=self.patience, **self.fit_options)
        return self

    def predict(self, X):
        return predict(self.model_, self.scaler_.transform(np.asarray(X, dtype=np.float32)))
