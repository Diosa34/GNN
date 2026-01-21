from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .graph import GraphData
from .model import TMDNet


def batch_iter(dataset: Iterable[GraphData], batch_size: int, shuffle: bool, seed: int):
    indices = np.arange(len(dataset))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        yield [dataset[i] for i in batch_ids]


def _batch_loss(model: TMDNet, batch: List[GraphData]) -> Tuple[float, float, List[float]]:
    preds = []
    targets = []
    for data in batch:
        pred = model.forward(data)
        preds.append(float(pred[0]))
        if data.y is None:
            raise ValueError("Target is missing in batch data.")
        targets.append(float(data.y.reshape(-1)[0]))
    preds_arr = np.asarray(preds, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.float32)
    mse = float(np.mean((preds_arr - targets_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    return mse, mae, preds


def _target_stats(dataset: Iterable[GraphData]) -> Tuple[float, float]:
    ys = [float(d.y.reshape(-1)[0]) for d in dataset if d.y is not None]
    mean = float(np.mean(ys))
    std = float(np.std(ys))
    std = std if std > 1e-6 else 1.0
    return mean, std


def train_model(
    train_dataset: Iterable[GraphData],
    val_dataset: Iterable[GraphData],
    node_feat_dim: int,
    edge_feat_dim: int,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 16,
    seed: int = 42,
    grad_clip: float = 1.0,
    hidden_dim: int = 128,
    num_layers: int = 4,
    normalize_y: bool = True,
    verbose: bool = True,
    ewt_epsilon: float = 0.5,
) -> Tuple[TMDNet, Dict[str, list]]:
    model = TMDNet(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seed=seed,
    )
    history = {
        "train_loss": [],
        "train_mae": [],
        "train_ewt": [],
        "val_loss": [],
        "val_mae": [],
        "val_ewt": [],
    }
    best_val = float("inf")
    best_state = None

    if normalize_y:
        y_mean, y_std = _target_stats(train_dataset)
        model.y_mean = y_mean
        model.y_std = y_std
    else:
        model.y_mean = 0.0
        model.y_std = 1.0

    for epoch in range(1, epochs + 1):
        losses, maes = [], []
        ewt_hits = 0
        ewt_total = 0
        for batch in batch_iter(train_dataset, batch_size, shuffle=True, seed=seed + epoch):
            model.zero_grad()
            batch_loss = 0.0
            for data in batch:
                pred = model.forward(data)
                target = data.y.reshape(-1)
                target = (target - model.y_mean) / model.y_std
                grad = (2.0 * (pred - target)) / len(batch)
                model.backward(data, grad)
                batch_loss += float(np.mean((pred - target) ** 2))
            model.clip_gradients(max_norm=grad_clip)
            model.step(lr, weight_decay)
            losses.append(batch_loss / len(batch))
            batch_preds = [(model.forward(d)[0] * model.y_std + model.y_mean) for d in batch]
            batch_targets = [d.y[0] for d in batch]
            maes.append(float(np.mean([abs(p - t) for p, t in zip(batch_preds, batch_targets)])))
            ewt_hits += sum(1 for p, t in zip(batch_preds, batch_targets) if abs(p - t) <= ewt_epsilon)
            ewt_total += len(batch)
        history["train_loss"].append(sum(losses) / len(losses))
        history["train_mae"].append(sum(maes) / len(maes))
        history["train_ewt"].append(ewt_hits / max(1, ewt_total))

        val_loss, val_mae, val_ewt = evaluate_model(model, val_dataset, batch_size, ewt_epsilon)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_ewt"].append(val_ewt)

        if verbose:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_mse={history['train_loss'][-1]:.4f} "
                f"val_mse={val_loss:.4f} "
                f"train_mae={history['train_mae'][-1]:.4f} "
                f"val_mae={val_mae:.4f} "
                f"train_ewt={history['train_ewt'][-1]:.3f} "
                f"val_ewt={val_ewt:.3f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = model_state(model)

    if best_state is not None:
        load_model_state(model, best_state)
    return model, history


def evaluate_model(
    model: TMDNet,
    dataset: Iterable[GraphData],
    batch_size: int = 32,
    ewt_epsilon: float = 0.5,
) -> Tuple[float, float, float]:
    losses, maes = [], []
    ewt_hits = 0
    ewt_total = 0
    y_mean = getattr(model, "y_mean", 0.0)
    y_std = getattr(model, "y_std", 1.0)
    for batch in batch_iter(dataset, batch_size, shuffle=False, seed=0):
        preds = []
        targets = []
        for data in batch:
            pred = model.forward(data)
            preds.append(float(pred[0] * y_std + y_mean))
            targets.append(float(data.y.reshape(-1)[0]))
        preds_arr = np.asarray(preds, dtype=np.float32)
        targets_arr = np.asarray(targets, dtype=np.float32)
        mse = float(np.mean((preds_arr - targets_arr) ** 2))
        mae = float(np.mean(np.abs(preds_arr - targets_arr)))
        losses.append(mse)
        maes.append(mae)
        ewt_hits += int(np.sum(np.abs(preds_arr - targets_arr) <= ewt_epsilon))
        ewt_total += len(batch)
    return float(np.mean(losses)), float(np.mean(maes)), ewt_hits / max(1, ewt_total)


def model_state(model: TMDNet) -> Dict[str, np.ndarray]:
    state = {
        "emb": model.emb.w.copy(),
        "node_proj_w": model.node_proj.w.copy(),
        "node_proj_b": model.node_proj.b.copy(),
        "read1_w": model.read1.w.copy(),
        "read1_b": model.read1.b.copy(),
        "read2_w": model.read2.w.copy(),
        "read2_b": model.read2.b.copy(),
    }
    for i, layer in enumerate(model.layers):
        state[f"layer_{i}_w_msg"] = layer.w_msg.w.copy()
        state[f"layer_{i}_b_msg"] = layer.w_msg.b.copy()
        state[f"layer_{i}_w_edge"] = layer.w_edge.w.copy()
        state[f"layer_{i}_b_edge"] = layer.w_edge.b.copy()
        state[f"layer_{i}_w_self"] = layer.w_self.w.copy()
        state[f"layer_{i}_b_self"] = layer.w_self.b.copy()
    return state


def load_model_state(model: TMDNet, state: Dict[str, np.ndarray]) -> None:
    model.emb.w = state["emb"]
    model.node_proj.w = state["node_proj_w"]
    model.node_proj.b = state["node_proj_b"]
    model.read1.w = state["read1_w"]
    model.read1.b = state["read1_b"]
    model.read2.w = state["read2_w"]
    model.read2.b = state["read2_b"]
    for i, layer in enumerate(model.layers):
        layer.w_msg.w = state[f"layer_{i}_w_msg"]
        layer.w_msg.b = state[f"layer_{i}_b_msg"]
        layer.w_edge.w = state[f"layer_{i}_w_edge"]
        layer.w_edge.b = state[f"layer_{i}_b_edge"]
        layer.w_self.w = state[f"layer_{i}_w_self"]
        layer.w_self.b = state[f"layer_{i}_b_self"]

