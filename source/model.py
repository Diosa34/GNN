from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .graph import GraphData


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


@dataclass
class Linear:
    w: np.ndarray
    b: np.ndarray
    x: np.ndarray | None = None
    dw: np.ndarray | None = None
    db: np.ndarray | None = None

    @staticmethod
    def init(in_dim: int, out_dim: int, rng: np.random.Generator) -> "Linear":
        scale = np.sqrt(2.0 / in_dim)
        w = rng.normal(0.0, scale, size=(in_dim, out_dim)).astype(np.float32)
        b = np.zeros((out_dim,), dtype=np.float32)
        return Linear(w=w, b=b, dw=np.zeros_like(w), db=np.zeros_like(b))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.dw += self.x.T @ grad
        self.db += grad.sum(axis=0)
        return grad @ self.w.T

    def zero_grad(self) -> None:
        self.dw.fill(0.0)
        self.db.fill(0.0)

    def step(self, lr: float, weight_decay: float = 0.0) -> None:
        if weight_decay:
            self.dw += weight_decay * self.w
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def clip_grad(self, max_norm: float) -> None:
        grad_norm = np.sqrt(np.sum(self.dw**2) + np.sum(self.db**2))
        if grad_norm > max_norm:
            scale = max_norm / (grad_norm + 1e-12)
            self.dw *= scale
            self.db *= scale


@dataclass
class Embedding:
    w: np.ndarray
    dw: np.ndarray
    idx: np.ndarray | None = None

    @staticmethod
    def init(num_embeddings: int, dim: int, rng: np.random.Generator) -> "Embedding":
        w = rng.normal(0.0, 0.1, size=(num_embeddings, dim)).astype(np.float32)
        return Embedding(w=w, dw=np.zeros_like(w))

    def forward(self, idx: np.ndarray) -> np.ndarray:
        self.idx = idx
        return self.w[idx]

    def backward(self, grad: np.ndarray) -> None:
        np.add.at(self.dw, self.idx, grad)

    def zero_grad(self) -> None:
        self.dw.fill(0.0)

    def step(self, lr: float, weight_decay: float = 0.0) -> None:
        if weight_decay:
            self.dw += weight_decay * self.w
        self.w -= lr * self.dw

    def clip_grad(self, max_norm: float) -> None:
        grad_norm = np.sqrt(np.sum(self.dw**2))
        if grad_norm > max_norm:
            scale = max_norm / (grad_norm + 1e-12)
            self.dw *= scale


class MPNNLayer:
    def __init__(self, hidden_dim: int, edge_dim: int, rng: np.random.Generator):
        self.w_msg = Linear.init(hidden_dim, hidden_dim, rng)
        self.w_edge = Linear.init(edge_dim, hidden_dim, rng)
        self.w_self = Linear.init(hidden_dim, hidden_dim, rng)
        self._cache = {}

    def forward(self, x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> np.ndarray:
        senders = edge_index[0]
        receivers = edge_index[1]
        x_s = x[senders]
        pre_msg = self.w_msg.forward(x_s) + self.w_edge.forward(edge_attr)
        msg = relu(pre_msg)
        agg = np.zeros_like(x)
        np.add.at(agg, receivers, msg)
        deg = np.bincount(receivers, minlength=x.shape[0]).astype(np.float32)
        deg = np.maximum(deg, 1.0)
        agg = agg / deg[:, None]
        pre_out = self.w_self.forward(x) + agg
        out = relu(pre_out) + x
        self._cache = {
            "x": x,
            "x_s": x_s,
            "senders": senders,
            "receivers": receivers,
            "edge_attr": edge_attr,
            "pre_msg": pre_msg,
            "msg": msg,
            "agg": agg,
            "deg": deg,
            "pre_out": pre_out,
        }
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        pre_out = self._cache["pre_out"]
        receivers = self._cache["receivers"]
        senders = self._cache["senders"]
        x = self._cache["x"]
        x_s = self._cache["x_s"]
        deg = self._cache["deg"]

        grad_pre_out = grad_out * (pre_out > 0)
        grad_x = self.w_self.backward(grad_pre_out)
        grad_x += grad_out
        grad_agg = grad_pre_out
        grad_msg = grad_agg[receivers] / deg[receivers, None]

        pre_msg = self._cache["pre_msg"]
        grad_pre_msg = grad_msg * (pre_msg > 0)

        grad_x_s = self.w_msg.backward(grad_pre_msg)
        _ = self.w_edge.backward(grad_pre_msg)

        np.add.at(grad_x, senders, grad_x_s)
        return grad_x

    def zero_grad(self) -> None:
        self.w_msg.zero_grad()
        self.w_edge.zero_grad()
        self.w_self.zero_grad()

    def step(self, lr: float, weight_decay: float = 0.0) -> None:
        self.w_msg.step(lr, weight_decay)
        self.w_edge.step(lr, weight_decay)
        self.w_self.step(lr, weight_decay)

    def clip_grad(self, max_norm: float) -> None:
        self.w_msg.clip_grad(max_norm)
        self.w_edge.clip_grad(max_norm)
        self.w_self.clip_grad(max_norm)


class TMDNet:
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        max_atomic_num: int = 100,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.emb = Embedding.init(max_atomic_num, hidden_dim, self.rng)
        self.node_proj = Linear.init(node_feat_dim + hidden_dim, hidden_dim, self.rng)
        self.layers = [MPNNLayer(hidden_dim, edge_feat_dim, self.rng) for _ in range(num_layers)]
        self.read1 = Linear.init(hidden_dim * 2, hidden_dim, self.rng)
        self.read2 = Linear.init(hidden_dim, 1, self.rng)
        self._cache = {}

    def forward(self, data: GraphData) -> np.ndarray:
        z = np.clip(data.z, 0, self.emb.w.shape[0] - 1)
        emb = self.emb.forward(z)
        h0 = np.concatenate([data.x, emb], axis=-1)
        pre_h = self.node_proj.forward(h0)
        h = relu(pre_h)
        for layer in self.layers:
            h = layer.forward(h, data.edge_index, data.edge_attr)
        mean_pool = h.mean(axis=0, keepdims=True)
        max_pool = h.max(axis=0, keepdims=True)
        max_idx = np.argmax(h, axis=0)
        pooled = np.concatenate([mean_pool, max_pool], axis=-1)
        pre_read = self.read1.forward(pooled)
        h_read = relu(pre_read)
        out = self.read2.forward(h_read)
        self._cache = {
            "h0": h0,
            "pre_h": pre_h,
            "h": h,
            "pre_read": pre_read,
            "pooled": pooled,
            "mean_pool": mean_pool,
            "max_pool": max_pool,
            "max_idx": max_idx,
            "h_read": h_read,
            "z": z,
        }
        return out.reshape(-1)

    def backward(self, data: GraphData, grad_out: np.ndarray) -> None:
        grad = self.read2.backward(grad_out.reshape(1, -1))
        pre_read = self._cache["pre_read"]
        grad = grad * (pre_read > 0)
        grad = self.read1.backward(grad)
        num_nodes = data.x.shape[0]
        hidden_dim = grad.shape[1] // 2
        grad_mean = grad[:, :hidden_dim]
        grad_max = grad[:, hidden_dim:]
        grad_h = np.repeat(grad_mean / num_nodes, num_nodes, axis=0)
        max_idx = self._cache["max_idx"]
        np.add.at(grad_h, max_idx, grad_max)
        for layer in reversed(self.layers):
            grad_h = layer.backward(grad_h)

        pre_h = self._cache["pre_h"]
        grad_pre_h = grad_h * (pre_h > 0)
        grad_h0 = self.node_proj.backward(grad_pre_h)

        node_feat_dim = data.x.shape[1]
        grad_emb = grad_h0[:, node_feat_dim:]
        self.emb.backward(grad_emb)

    def zero_grad(self) -> None:
        self.emb.zero_grad()
        self.node_proj.zero_grad()
        for layer in self.layers:
            layer.zero_grad()
        self.read1.zero_grad()
        self.read2.zero_grad()

    def step(self, lr: float, weight_decay: float = 0.0) -> None:
        self.emb.step(lr, weight_decay)
        self.node_proj.step(lr, weight_decay)
        for layer in self.layers:
            layer.step(lr, weight_decay)
        self.read1.step(lr, weight_decay)
        self.read2.step(lr, weight_decay)

    def clip_gradients(self, max_norm: float = 1.0) -> None:
        self.emb.clip_grad(max_norm)
        self.node_proj.clip_grad(max_norm)
        for layer in self.layers:
            layer.clip_grad(max_norm)
        self.read1.clip_grad(max_norm)
        self.read2.clip_grad(max_norm)

