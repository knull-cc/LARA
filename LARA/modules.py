import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UtilityLabeler(nn.Module):
    """Oracle utility labels from counterfactual interpolation with host forecast."""

    def __init__(self, alpha_step=0.1):
        super().__init__()
        if alpha_step <= 0 or alpha_step > 1:
            raise ValueError("alpha_step must be in (0, 1].")
        alpha_grid = torch.arange(0.0, 1.0 + 1e-6, alpha_step)
        self.register_buffer("alpha_grid", alpha_grid)

    def forward(self, y_host, y_true, y_cand):
        with torch.no_grad():
            host = y_host.detach()
            true = y_true.detach()
            cand = y_cand.detach()
            base_loss = (host - true).pow(2).mean(dim=(1, 2), keepdim=False).unsqueeze(1)

            best_loss = None
            best_alpha = torch.zeros(cand.shape[:2], device=cand.device, dtype=cand.dtype)
            for alpha in self.alpha_grid.to(cand.device, cand.dtype):
                mixed = (1.0 - alpha) * host.unsqueeze(1) + alpha * cand
                loss = (mixed - true.unsqueeze(1)).pow(2).mean(dim=(2, 3))
                if best_loss is None:
                    best_loss = loss
                    best_alpha.fill_(float(alpha))
                else:
                    improved = loss < best_loss
                    best_loss = torch.where(improved, loss, best_loss)
                    best_alpha = torch.where(improved, alpha.expand_as(best_alpha), best_alpha)

            delta = (base_loss - best_loss).clamp_min(0.0)
            return delta, best_alpha


class UtilityReranker(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, candidate_features):
        return self.net(candidate_features).squeeze(-1)


def sparsemax(logits, dim=-1):
    logits = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(logits, descending=True, dim=dim).values
    range_shape = [1] * logits.dim()
    range_shape[dim] = logits.size(dim)
    rhos = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype).view(range_shape)
    support = 1 + rhos * zs > zs.cumsum(dim)
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (zs.cumsum(dim).gather(dim, k.long() - 1) - 1) / k
    return torch.clamp(logits - tau, min=0.0)


def aggregate_candidates(scores, candidate_futures, temperature=0.1, mode="softmax"):
    scaled = scores / max(temperature, 1e-6)
    if mode == "sparsemax":
        weights = sparsemax(scaled, dim=1)
    elif mode == "softmax":
        weights = F.softmax(scaled, dim=1)
    else:
        raise ValueError(f"Unknown LARA sparse mode: {mode}")

    y_ret = torch.einsum("bm,bmhc->bhc", weights, candidate_futures)
    return y_ret, weights


class FusionGate(nn.Module):
    def __init__(self, pred_len, gate_type="scalar", stat_dim=6):
        super().__init__()
        self.pred_len = pred_len
        self.gate_type = gate_type
        output_dim = pred_len if gate_type == "horizon" else 1
        self.net = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )
        nn.init.constant_(self.net[-1].bias, -1.5)

    def forward(self, stats):
        gate = torch.sigmoid(self.net(stats))
        if self.gate_type == "horizon":
            return gate.unsqueeze(-1)
        return gate.view(gate.shape[0], 1, 1)


def listwise_kl_loss(scores, utility, temperature=0.1):
    target = F.softmax(utility / max(temperature, 1e-6), dim=1)
    log_prob = F.log_softmax(scores / max(temperature, 1e-6), dim=1)
    return F.kl_div(log_prob, target, reduction="batchmean")


def pairwise_utility_margin_loss(scores, utility, margin=0.2):
    if scores.shape[1] < 2:
        return scores.new_tensor(0.0)

    with torch.no_grad():
        best_idx = utility.argmax(dim=1)
        worst_idx = utility.argmin(dim=1)
        utility_gap = utility.gather(1, best_idx[:, None]) - utility.gather(1, worst_idx[:, None])
        valid = utility_gap.squeeze(1) > 1e-8

    if not valid.any():
        return scores.new_tensor(0.0)

    pos_score = scores.gather(1, best_idx[:, None]).squeeze(1)
    neg_score = scores.gather(1, worst_idx[:, None]).squeeze(1)
    return F.relu(float(margin) - (pos_score - neg_score))[valid].mean()


def utility_score_alignment_loss(scores, utility, mode="mse"):
    if scores.shape[1] < 2:
        return scores.new_tensor(0.0)

    utility = utility.detach()
    score_centered = scores - scores.mean(dim=1, keepdim=True)
    utility_centered = utility - utility.mean(dim=1, keepdim=True)
    score_std = score_centered.std(dim=1, keepdim=True)
    utility_std = utility_centered.std(dim=1, keepdim=True)
    valid = (utility_std.squeeze(1) > 1e-8) & (score_std.squeeze(1) > 1e-8)

    if not valid.any():
        return scores.new_tensor(0.0)

    score_z = score_centered[valid] / score_std[valid].clamp_min(1e-6)
    utility_z = utility_centered[valid] / utility_std[valid].clamp_min(1e-6)
    if mode == "mse":
        return F.mse_loss(score_z, utility_z)
    if mode == "corr":
        corr = (score_z * utility_z).mean(dim=1)
        return (1.0 - corr).mean()
    raise ValueError(f"Unknown LARA score alignment loss: {mode}")


def normalized_entropy(weights):
    entropy = -(weights * (weights.clamp_min(1e-8)).log()).sum(dim=1)
    return entropy.mean() / math.log(max(weights.shape[1], 2))


def rank_correlation(a, b):
    if a.shape[1] < 2:
        return torch.zeros((), device=a.device)
    ar = torch.argsort(torch.argsort(a, dim=1), dim=1).float()
    br = torch.argsort(torch.argsort(b, dim=1), dim=1).float()
    ar = ar - ar.mean(dim=1, keepdim=True)
    br = br - br.mean(dim=1, keepdim=True)
    denom = ar.std(dim=1).clamp_min(1e-6) * br.std(dim=1).clamp_min(1e-6)
    return ((ar * br).mean(dim=1) / denom).mean()
