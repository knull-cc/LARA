import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from LARA.memory import TimeSeriesMemory
from LARA.modules import (
    FusionGate,
    UtilityLabeler,
    UtilityReranker,
    aggregate_candidates,
    listwise_kl_loss,
    normalized_entropy,
    pairwise_utility_margin_loss,
    rank_correlation,
    utility_score_alignment_loss,
)
from LARA.retrieval import CandidateRetriever
from models.DLinear import Model as DLinearModel


def _strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


class Model(nn.Module):
    """LARA MVP adapter plugged into DLinear.

    The host DLinear produces y_host. LARA retrieves train-only candidate futures,
    learns a candidate utility reranker from counterfactual forecast-loss labels,
    and gates y_host with the retrieval aggregate.
    """

    supports_lara_context = True

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len if configs.task_name != "classification" else configs.seq_len
        self.channels = configs.enc_in

        self.host = DLinearModel(configs, individual=getattr(configs, "individual", False))
        self.freeze_host = bool(getattr(configs, "lara_freeze_host", False))
        self._load_host_checkpoint(getattr(configs, "lara_host_ckpt", ""))
        if self.freeze_host:
            for param in self.host.parameters():
                param.requires_grad = False

        self.top_m = int(getattr(configs, "lara_top_m", 32))
        self.temperature = float(getattr(configs, "lara_temperature", 0.1))
        self.rank_temperature = float(getattr(configs, "lara_rank_temperature", 0.1))
        self.lambda_rank = float(getattr(configs, "lara_lambda_rank", 0.3))
        self.lambda_pair = float(getattr(configs, "lara_lambda_pair", 0.0))
        self.pair_margin = float(getattr(configs, "lara_pair_margin", 0.2))
        self.lambda_score = float(getattr(configs, "lara_lambda_score", 0.0))
        self.score_loss_mode = getattr(configs, "lara_score_loss", "mse")
        self.lambda_sparse = float(getattr(configs, "lara_lambda_sparse", 0.01))
        self.lambda_gate = float(getattr(configs, "lara_lambda_gate", 0.0))
        self.sparse_mode = getattr(configs, "lara_sparse_mode", "softmax")
        self.offset_align = bool(getattr(configs, "lara_offset_align", True))

        self.labeler = UtilityLabeler(alpha_step=float(getattr(configs, "lara_alpha_step", 0.1)))
        self.reranker = UtilityReranker(feature_dim=6)
        self.gate = FusionGate(self.pred_len, gate_type=getattr(configs, "lara_gate", "scalar"))

        self.memory = None
        self.retriever = None
        self.memory_prepared = False
        self.warned_no_memory = False
        self.aux_loss = None
        self.last_diagnostics = {}
        self.diagnostic_sums = {}
        self.diagnostic_count = 0
        self.oracle_sums = defaultdict(float)
        self.oracle_count = 0

    @staticmethod
    def _parse_int_list(value, default):
        if value is None:
            return list(default)
        if isinstance(value, (list, tuple)):
            return [int(item) for item in value]
        parsed = []
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
        return parsed or list(default)

    def _load_host_checkpoint(self, checkpoint_path):
        if not checkpoint_path:
            return
        if not os.path.exists(checkpoint_path):
            print(f"[LARA] host checkpoint not found, training host jointly: {checkpoint_path}")
            return
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _strip_module_prefix(state)
        missing, unexpected = self.host.load_state_dict(state, strict=False)
        print(
            f"[LARA] loaded host checkpoint from {checkpoint_path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def prepare_memory(self, train_data, device=None):
        self.memory = TimeSeriesMemory.from_dataset(
            train_data,
            pred_len=self.pred_len,
            key_mode=getattr(self.configs, "lara_key_mode", "auto"),
            max_key_dim=int(getattr(self.configs, "lara_max_key_dim", 8192)),
        )
        self.retriever = CandidateRetriever(
            self.memory,
            top_m=self.top_m,
            chunk_size=int(getattr(self.configs, "lara_retrieval_chunk", 4096)),
            overlap_margin=int(getattr(self.configs, "lara_overlap_margin", 0)),
        )
        self.memory_prepared = True
        print(
            "[LARA] train-only memory prepared: "
            f"N={self.memory.size}, seq_len={self.memory.seq_len}, "
            f"pred_len={self.memory.pred_len}, channels={self.memory.channels}, "
            f"key_mode={self.memory.key_mode}, top_m={self.top_m}"
        )

    def _host_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, sample_index=None):
        def call_host():
            if getattr(self.host, "supports_sample_index", False):
                return self.host(
                    x_enc,
                    x_mark_enc,
                    x_dec,
                    x_mark_dec,
                    sample_index=sample_index,
                )
            return self.host(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.freeze_host:
            self.host.eval()
            with torch.no_grad():
                return call_host()
        return call_host()

    def _build_gate_stats(self, scores, weights, y_host, y_ret):
        sorted_scores = torch.sort(scores, dim=1, descending=True).values
        top1 = sorted_scores[:, 0]
        margin = top1 - sorted_scores[:, 1] if scores.shape[1] > 1 else torch.zeros_like(top1)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / torch.log(torch.tensor(float(max(weights.shape[1], 2)), device=weights.device))
        active_k = (weights > 1e-3).float().sum(dim=1) / float(max(weights.shape[1], 1))
        host_scale = y_host.detach().std(dim=(1, 2))
        ret_scale = y_ret.detach().std(dim=(1, 2))
        return torch.stack([top1, margin, entropy, active_k, host_scale, ret_scale], dim=1)

    def _align_candidate_futures(self, x_enc, cand_pasts, cand_futures):
        if not self.offset_align:
            return cand_futures
        query_last = x_enc[:, -1:, :].unsqueeze(1)
        candidate_last = cand_pasts[:, :, -1:, :]
        candidate_delta = cand_futures - candidate_last
        return query_last + candidate_delta

    def _set_diagnostics(self, diagnostics):
        self.last_diagnostics = {
            key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            for key, value in diagnostics.items()
        }
        for key, value in self.last_diagnostics.items():
            self.diagnostic_sums[key] = self.diagnostic_sums.get(key, 0.0) + value
        self.diagnostic_count += 1

    def get_lara_diagnostics(self, reset=False):
        diagnostics = dict(self.last_diagnostics)
        if reset:
            self.last_diagnostics = {}
        return diagnostics

    def get_lara_diagnostic_average(self, reset=False):
        if self.diagnostic_count == 0:
            return {}
        diagnostics = {
            key: value / self.diagnostic_count
            for key, value in self.diagnostic_sums.items()
        }
        if reset:
            self.diagnostic_sums = {}
            self.diagnostic_count = 0
        return diagnostics

    def _add_oracle_sum(self, key, values):
        if torch.is_tensor(values):
            values = values.detach()
            self.oracle_sums[key] += float(values.sum().cpu())
            return
        self.oracle_sums[key] += float(values)

    def _accumulate_oracle_diagnostics(self, y_host, y_ret, y_final, y_true, cand_futures):
        with torch.no_grad():
            batch_size = int(y_true.shape[0])
            self.oracle_count += batch_size

            host_err = (y_host - y_true).pow(2).mean(dim=(1, 2))
            ret_err = (y_ret - y_true).pow(2).mean(dim=(1, 2))
            final_err = (y_final - y_true).pow(2).mean(dim=(1, 2))
            cand_err = (cand_futures - y_true.unsqueeze(1)).pow(2).mean(dim=(2, 3))
            max_candidates = cand_err.shape[1]

            self._add_oracle_sum("candidate_pool_size", batch_size * max_candidates)
            self._add_oracle_sum("host_mse", host_err)
            self._add_oracle_sum("retrieval_mse", ret_err)
            self._add_oracle_sum("lara_final_mse", final_err)

            oracle_gate_err = torch.minimum(host_err, ret_err)
            self._add_oracle_sum("oracle_gate_mse", oracle_gate_err)
            self._add_oracle_sum("oracle_gate_beneficial_rate", (ret_err < host_err).float())
            self._add_oracle_sum("oracle_gate_gain", host_err - oracle_gate_err)

            host_horizon_err = (y_host - y_true).pow(2).mean(dim=2)
            ret_horizon_err = (y_ret - y_true).pow(2).mean(dim=2)
            horizon_gate_err = torch.minimum(host_horizon_err, ret_horizon_err).mean(dim=1)
            self._add_oracle_sum("oracle_horizon_gate_mse", horizon_gate_err)
            self._add_oracle_sum(
                "oracle_horizon_gate_beneficial_rate",
                (ret_horizon_err < host_horizon_err).float().mean(dim=1),
            )
            self._add_oracle_sum("oracle_horizon_gate_gain", host_err - horizon_gate_err)

            top_m_values = self._parse_int_list(
                getattr(self.configs, "lara_oracle_ms", None),
                default=[1, 3, 5, 10, 20, 50],
            )
            for requested_m in top_m_values:
                m = int(requested_m)
                if m <= 0 or m > max_candidates:
                    continue
                best_err = cand_err[:, :m].min(dim=1).values
                prefix = f"oracle_candidate_m{m}"
                self._add_oracle_sum(f"{prefix}_mse", best_err)
                self._add_oracle_sum(f"{prefix}_beneficial_rate", (best_err < host_err).float())
                self._add_oracle_sum(f"{prefix}_gain", host_err - best_err)

                cand_horizon_err = (cand_futures[:, :m] - y_true.unsqueeze(1)).pow(2).mean(dim=3)
                best_horizon_err = cand_horizon_err.min(dim=1).values.mean(dim=1)
                horizon_prefix = f"oracle_candidate_horizon_m{requested_m}"
                self._add_oracle_sum(f"{horizon_prefix}_mse", best_horizon_err)
                self._add_oracle_sum(f"{horizon_prefix}_gain", host_err - best_horizon_err)

            topk_values = self._parse_int_list(
                getattr(self.configs, "lara_oracle_topk", None),
                default=[1, 3, 5],
            )
            sorted_idx = cand_err.argsort(dim=1)
            for requested_k in topk_values:
                k = int(requested_k)
                if k <= 0 or k > max_candidates:
                    continue
                gather_idx = sorted_idx[:, :k, None, None].expand(-1, -1, cand_futures.shape[2], cand_futures.shape[3])
                topk_futures = cand_futures.gather(1, gather_idx)
                topk_avg = topk_futures.mean(dim=1)
                topk_err = (topk_avg - y_true).pow(2).mean(dim=(1, 2))
                prefix = f"oracle_aggregation_top{k}_mse"
                self._add_oracle_sum(prefix, topk_err)
                self._add_oracle_sum(f"oracle_aggregation_top{k}_gain", host_err - topk_err)

    def get_lara_oracle_average(self, reset=False):
        if self.oracle_count == 0:
            return {}
        diagnostics = {
            key: value / self.oracle_count
            for key, value in sorted(self.oracle_sums.items())
        }
        if reset:
            self.oracle_sums = defaultdict(float)
            self.oracle_count = 0
        return diagnostics

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
        sample_index=None,
        mode="train",
        y_true=None,
    ):
        y_host = self._host_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, sample_index=sample_index)
        self.aux_loss = y_host.new_tensor(0.0)

        if not self.memory_prepared:
            if not self.warned_no_memory:
                print("[LARA] memory is not prepared; falling back to the plain host forecaster.")
                self.warned_no_memory = True
            return y_host

        retrieval = self.retriever.retrieve(
            x_enc,
            query_mark=x_mark_enc,
            sample_index=sample_index,
            train=(mode == "train"),
        )
        cand_pasts = retrieval["pasts"].to(y_host.device)
        cand_futures = retrieval["futures"].to(y_host.device)
        cand_futures = self._align_candidate_futures(x_enc, cand_pasts, cand_futures)
        scores = self.reranker(retrieval["features"].to(y_host.device))
        y_ret, weights = aggregate_candidates(
            scores,
            cand_futures,
            temperature=self.temperature,
            mode=self.sparse_mode,
        )

        gate_stats = self._build_gate_stats(scores, weights, y_host, y_ret)
        gate = self.gate(gate_stats)
        y_final = (1.0 - gate) * y_host + gate * y_ret
        if y_true is not None and mode == "test":
            self._accumulate_oracle_diagnostics(y_host.detach(), y_ret.detach(), y_final.detach(), y_true.detach(), cand_futures.detach())

        diagnostics = {
            "gate": gate.mean(),
            "active_k": (weights > 1e-3).float().sum(dim=1).mean(),
            "weight_entropy": normalized_entropy(weights),
            "offset_align": y_host.new_tensor(1.0 if self.offset_align else 0.0),
        }

        if y_true is not None:
            utility, alpha_star = self.labeler(y_host, y_true, cand_futures)
            oracle_mse_per_query = (
                (y_host.detach().unsqueeze(1) * (1.0 - alpha_star[:, :, None, None])
                 + cand_futures.detach() * alpha_star[:, :, None, None]
                 - y_true.detach().unsqueeze(1))
                .pow(2)
                .mean(dim=(2, 3))
                .min(dim=1)
                .values
            )
            host_mse_per_query = (y_host.detach() - y_true.detach()).pow(2).mean(dim=(1, 2))
            final_mse_per_query = (y_final.detach() - y_true.detach()).pow(2).mean(dim=(1, 2))
            best_idx = scores.detach().argmax(dim=1)
            top_alpha = alpha_star.gather(1, best_idx[:, None]).squeeze(1)
            weighted_alpha = (weights.detach() * alpha_star).sum(dim=1)
            gate_target = weighted_alpha[:, None, None].expand_as(gate)
            gate_loss = F.mse_loss(gate, gate_target)

            diagnostics.update(
                {
                    "host_mse": host_mse_per_query.mean(),
                    "final_mse": final_mse_per_query.mean(),
                    "oracle_mse": oracle_mse_per_query.mean(),
                    "oracle_gain": ((host_mse_per_query - oracle_mse_per_query)
                                    / host_mse_per_query.clamp_min(1e-8)).mean(),
                    "positive_utility": (utility > 1e-8).float().mean(),
                    "helpful_query": (utility.max(dim=1).values > 1e-8).float().mean(),
                    "alpha_star": alpha_star.mean(),
                    "top_alpha": top_alpha.mean(),
                    "weighted_alpha": weighted_alpha.mean(),
                    "gate_loss": gate_loss,
                    "score_std": scores.detach().std(dim=1).mean(),
                    "utility_gap": (utility.max(dim=1).values - utility.min(dim=1).values).mean(),
                }
            )

        if self.training and y_true is not None:
            rank_loss = listwise_kl_loss(scores, utility, temperature=self.rank_temperature)
            pair_loss = pairwise_utility_margin_loss(scores, utility, margin=self.pair_margin)
            score_loss = utility_score_alignment_loss(scores, utility, mode=self.score_loss_mode)
            sparse_loss = normalized_entropy(weights)
            self.aux_loss = (
                self.lambda_rank * rank_loss
                + self.lambda_pair * pair_loss
                + self.lambda_score * score_loss
                + self.lambda_sparse * sparse_loss
                + self.lambda_gate * gate_loss
            )
            diagnostics.update(
                {
                    "rank_loss": rank_loss,
                    "pair_loss": pair_loss,
                    "score_loss": score_loss,
                    "sparse_loss": sparse_loss,
                    "rank_corr": rank_correlation(utility, scores.detach()),
                    "lambda_pair": y_host.new_tensor(self.lambda_pair),
                    "lambda_score": y_host.new_tensor(self.lambda_score),
                    "lambda_gate": y_host.new_tensor(self.lambda_gate),
                }
            )

        self._set_diagnostics(diagnostics)
        return y_final[:, -self.pred_len:, :]
