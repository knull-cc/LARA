import torch


class CandidateRetriever:
    """Top-M retrieval over a train-only TimeSeriesMemory."""

    def __init__(self, memory, top_m=32, chunk_size=4096, overlap_margin=0):
        self.memory = memory
        self.top_m = top_m
        self.chunk_size = chunk_size
        self.overlap_margin = overlap_margin

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

    def retrieve(self, query, query_mark=None, sample_index=None, train=False):
        sim = self._similarity(query)
        if train:
            sim = self._apply_temporal_mask(sim, sample_index)

        k = min(self.top_m, self.memory.size)
        values, indices = torch.topk(sim, k=k, dim=1)
        device = query.device

        cand_futures = self.memory.futures[indices.cpu()].to(device)
        cand_pasts = self.memory.pasts[indices.cpu()].to(device)

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
            "pasts": cand_pasts,
            "futures": cand_futures,
            "features": features,
        }
