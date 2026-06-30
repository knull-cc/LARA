import numpy as np
import torch
import torch.nn.functional as F


def unwrap_dataset(dataset):
    return getattr(dataset, "dataset", dataset)


def _sample_to_parts(sample):
    if len(sample) == 5:
        sample = sample[:4]
    seq_x, seq_y, seq_x_mark, seq_y_mark = sample
    return seq_x, seq_y, seq_x_mark, seq_y_mark


def _as_float_tensor(items):
    return torch.tensor(np.stack(items, axis=0), dtype=torch.float32)


def choose_key_mode(pasts, requested_mode, max_key_dim):
    if requested_mode != "auto":
        return requested_mode
    flat_dim = pasts.shape[1] * pasts.shape[2]
    return "flatten" if flat_dim <= max_key_dim else "mean"


def build_keys(pasts, key_mode):
    if key_mode == "mean":
        x = pasts.mean(dim=2, keepdim=True)
    elif key_mode == "flatten":
        x = pasts
    else:
        raise ValueError(f"Unknown LARA key_mode: {key_mode}")

    x = x - x[:, -1:, :]
    x = x / (x.std(dim=1, keepdim=True).clamp_min(1e-5))
    keys = x.flatten(start_dim=1)
    return F.normalize(keys, dim=1)


class TimeSeriesMemory:
    """Train-only memory bank used by the LARA MVP retriever."""

    def __init__(self, pasts, futures, marks, indices, key_mode):
        self.pasts = pasts
        self.futures = futures
        self.marks = marks
        self.indices = indices
        self.key_mode = key_mode
        self.keys = build_keys(pasts, key_mode)

        self.seq_len = pasts.shape[1]
        self.pred_len = futures.shape[1]
        self.channels = pasts.shape[2]
        self.size = pasts.shape[0]

    @classmethod
    def from_dataset(cls, dataset, pred_len, key_mode="auto", max_key_dim=8192):
        base_dataset = unwrap_dataset(dataset)
        pasts = []
        futures = []
        marks = []
        indices = []

        for index in range(len(base_dataset)):
            seq_x, seq_y, seq_x_mark, _ = _sample_to_parts(base_dataset[index])
            pasts.append(seq_x)
            futures.append(seq_y[-pred_len:])
            if seq_x_mark is None:
                marks.append(np.zeros((1,), dtype=np.float32))
            else:
                marks.append(np.asarray(seq_x_mark[-1], dtype=np.float32))
            indices.append(index)

        past_tensor = _as_float_tensor(pasts)
        future_tensor = _as_float_tensor(futures)
        mark_tensor = _as_float_tensor(marks)
        index_tensor = torch.tensor(indices, dtype=torch.long)
        resolved_key_mode = choose_key_mode(past_tensor, key_mode, max_key_dim)

        return cls(
            pasts=past_tensor,
            futures=future_tensor,
            marks=mark_tensor,
            indices=index_tensor,
            key_mode=resolved_key_mode,
        )

    def query_keys(self, query):
        return build_keys(query.detach().float().cpu(), self.key_mode)
