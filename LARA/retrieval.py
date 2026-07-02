import torch


class CandidateRetriever:
    """Top-M retrieval over a train-only TimeSeriesMemory."""

    def __init__(
        self,
        memory,
        top_m=32,
        chunk_size=4096,
        overlap_margin=0,
        phase_top_k=0,
        phase_weight=0.2,
        pibr_period=24,
        pibr_weight=0.0,
        pibr_delta_weight=0.5,
        phase_rerank_mode="add",
    ):
        self.memory = memory
        self.top_m = top_m
        self.chunk_size = chunk_size
        self.overlap_margin = overlap_margin
        self.phase_top_k = int(phase_top_k)
        self.phase_weight = float(phase_weight)
        self.pibr_period = int(pibr_period)
        self.pibr_weight = float(pibr_weight)
        self.pibr_delta_weight = float(pibr_delta_weight)
        self.phase_rerank_mode = phase_rerank_mode

    def _similarity(self, query):
        q_keys = self.memory.query_keys(query).to(query.device)
        sims = []
        for start in range(0, self.memory.size, self.chunk_size):
            end = min(start + self.chunk_size, self.memory.size)
            keys = self.memory.keys[start:end].to(query.device)
            sims.append(torch.matmul(q_keys, keys.transpose(0, 1)))
        return torch.cat(sims, dim=1)

    def _apply_temporal_mask(self, sim, sample_index):
        if sample_index is None:
            return sim
        sample_index = sample_index.to(sim.device).long()
        memory_index = self.memory.indices.to(sim.device)
        radius = self.memory.seq_len + self.memory.pred_len + self.overlap_margin
        mask = (memory_index.unsqueeze(0) - sample_index.unsqueeze(1)).abs() <= radius
        masked = sim.masked_fill(mask, float("-inf"))

        all_masked = torch.isinf(masked).all(dim=1)
        if all_masked.any():
            masked[all_masked] = sim[all_masked]
        return masked

    def _time_bonus(self, query_mark, cand_indices, device):
        if query_mark is None:
            return torch.zeros_like(cand_indices, dtype=torch.float32, device=device)

        q_mark = query_mark[:, -1, :].detach().float().to(device)
        mem_marks = self.memory.marks[cand_indices.cpu()].to(device)
        if q_mark.shape[-1] != mem_marks.shape[-1]:
            return torch.zeros_like(cand_indices, dtype=torch.float32, device=device)
        return torch.exp(-(q_mark.unsqueeze(1) - mem_marks).abs().mean(dim=-1))

    def _phase_bonus(self, query_mark, cand_indices, device):
        if query_mark is None:
            return torch.zeros_like(cand_indices, dtype=torch.float32, device=device)

        q_mark = query_mark[:, -1, :].detach().float().to(device)
        mem_marks = self.memory.marks[cand_indices.cpu()].to(device)
        if q_mark.shape[-1] != mem_marks.shape[-1]:
            return torch.zeros_like(cand_indices, dtype=torch.float32, device=device)

        diff = (q_mark.unsqueeze(1) - mem_marks).abs()
        max_abs = torch.maximum(q_mark.abs().max(), mem_marks.abs().max())
        if max_abs <= 0.51:
            diff = torch.minimum(diff, (1.0 - diff).clamp_min(0.0))
            distance = diff.mean(dim=-1)
        elif diff.shape[-1] in (4, 5):
            periods = torch.tensor([12.0, 31.0, 7.0, 24.0, 4.0], device=device, dtype=diff.dtype)
            periods = periods[:diff.shape[-1]].view(1, 1, -1)
            wrapped = torch.minimum(diff.remainder(periods), periods - diff.remainder(periods))
            distance = (wrapped / periods).mean(dim=-1)
        else:
            distance = diff.mean(dim=-1)
        return torch.exp(-4.0 * distance)

    @staticmethod
    def _cosine01(a, b):
        a = a.flatten(start_dim=-2)
        b = b.flatten(start_dim=-2)
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        sim = torch.nn.functional.cosine_similarity(a.unsqueeze(1), b, dim=-1)
        return ((sim + 1.0) * 0.5).clamp(0.0, 1.0)

    def _pibr_bonus(self, query, cand_pasts):
        period = min(self.pibr_period, query.shape[1], cand_pasts.shape[2])
        if period <= 1:
            return cand_pasts.new_zeros(cand_pasts.shape[:2])

        query_cycle = query[:, -period:, :].detach().float()
        cand_cycle = cand_pasts[:, :, -period:, :].detach().float()
        profile_bonus = self._cosine01(query_cycle, cand_cycle)

        if query.shape[1] < 2 * period or cand_pasts.shape[2] < 2 * period:
            return profile_bonus

        query_delta = query[:, -period:, :] - query[:, -2 * period:-period, :]
        cand_delta = cand_pasts[:, :, -period:, :] - cand_pasts[:, :, -2 * period:-period, :]
        delta_bonus = self._cosine01(query_delta.detach().float(), cand_delta.detach().float())
        delta_weight = min(max(self.pibr_delta_weight, 0.0), 1.0)
        return (1.0 - delta_weight) * profile_bonus + delta_weight * delta_bonus

    def _phase_rerank(self, values, indices, phase_bonus, pibr_bonus):
        if self.phase_rerank_mode == "phase":
            rerank_score = phase_bonus
        elif self.phase_rerank_mode == "pibr":
            rerank_score = pibr_bonus
        else:
            rerank_score = values + self.phase_weight * phase_bonus + self.pibr_weight * pibr_bonus

        k = min(self.top_m, values.shape[1])
        _, select_pos = torch.topk(rerank_score, k=k, dim=1)
        return (
            values.gather(1, select_pos),
            indices.gather(1, select_pos),
            phase_bonus.gather(1, select_pos),
            pibr_bonus.gather(1, select_pos),
            select_pos,
        )

    def retrieve(self, query, query_mark=None, sample_index=None, train=False):
        sim = self._similarity(query)
        if train:
            sim = self._apply_temporal_mask(sim, sample_index)

        device = query.device
        pool_k = self.top_m
        use_phase_rerank = self.phase_top_k > self.top_m
        if use_phase_rerank:
            pool_k = self.phase_top_k

        pool_k = min(pool_k, self.memory.size)
        values, indices = torch.topk(sim, k=pool_k, dim=1)
        phase_bonus = self._phase_bonus(query_mark, indices, device)

        cand_futures = self.memory.futures[indices.cpu()].to(device)
        cand_pasts = self.memory.pasts[indices.cpu()].to(device)
        pibr_bonus = self._pibr_bonus(query, cand_pasts)

        if use_phase_rerank:
            values, indices, phase_bonus, pibr_bonus, select_pos = self._phase_rerank(
                values,
                indices,
                phase_bonus,
                pibr_bonus,
            )
            gather_idx = select_pos[:, :, None, None].expand(-1, -1, cand_futures.shape[2], cand_futures.shape[3])
            cand_futures = cand_futures.gather(1, gather_idx)
            cand_pasts = cand_pasts.gather(1, gather_idx)

        k = values.shape[1]

        rank = torch.arange(k, device=device, dtype=query.dtype).unsqueeze(0).expand_as(values)
        rank_feature = 1.0 - rank / max(k - 1, 1)
        score_margin = values[:, :1] - values
        time_bonus = self._time_bonus(query_mark, indices, device)

        future_center = cand_futures.mean(dim=1, keepdim=True)
        future_dispersion = (cand_futures - future_center).pow(2).mean(dim=(2, 3)).sqrt()
        past_scale = cand_pasts.std(dim=(2, 3))

        features = torch.stack(
            [
                values,
                time_bonus,
                phase_bonus,
                pibr_bonus,
                rank_feature,
                score_margin,
                future_dispersion,
                past_scale,
            ],
            dim=-1,
        )

        return {
            "indices": indices.to(device),
            "similarity": values,
            "phase_bonus": phase_bonus,
            "pibr_bonus": pibr_bonus,
            "phase_rerank": torch.full((query.shape[0],), 1.0 if use_phase_rerank else 0.0, device=device),
            "phase_pool_k": torch.full((query.shape[0],), float(pool_k), device=device),
            "pasts": cand_pasts,
            "futures": cand_futures,
            "features": features,
        }
