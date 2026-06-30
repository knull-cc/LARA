import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalTemporalRetrieval(nn.Module):
    def __init__(self, d_series, c, ci=False, period_len=24):
        super().__init__()
        self.agg = False
        self.period_len = period_len
        self.c = c
        self.linear = nn.Linear(d_series, d_series)
        self.ci = ci
        if self.ci:
            self.ds_convs = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=(2, 1 + 2 * (self.period_len // 2)),
                        stride=1,
                        padding=(0, self.period_len // 2),
                        padding_mode="zeros",
                        bias=False,
                    )
                    for _ in range(self.c)
                ]
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(2, 1 + 2 * (self.period_len // 2)),
                stride=1,
                padding=(0, self.period_len // 2),
                padding_mode="zeros",
                bias=False,
            )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, q):
        _, channels, steps = x.shape
        global_query = self.linear(q)

        if self.agg:
            weight = F.softmax(global_query, dim=1)
            global_query = torch.sum(global_query * weight, dim=1, keepdim=True)
            global_query = global_query.repeat(1, channels, 1)

        out = torch.stack([x, global_query], dim=2)

        if self.ci:
            conv_outs = [
                self.ds_convs[i](out[:, i, :, :].unsqueeze(1))
                for i in range(self.c)
            ]
            conv_out = torch.cat(conv_outs, dim=1).squeeze(2)
        else:
            out = out.reshape(-1, 1, 2, steps)
            conv_out = self.conv2d(out).reshape(-1, channels, steps)

        return self.dropout(conv_out)


class Model(nn.Module):
    """GTR adapted to the TSF-Lib four-argument forecasting interface."""

    supports_sample_index = True

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = int(getattr(configs, "cycle", 24))
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = bool(int(getattr(configs, "use_revin", 1)))
        self.individual = bool(getattr(configs, "individual", False))

        self.Q = nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)
        self.GTR = GlobalTemporalRetrieval(
            d_series=self.seq_len,
            c=self.enc_in,
            ci=self.individual,
        )
        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )

    def _cycle_index(self, x, sample_index=None):
        batch_size = x.shape[0]
        if sample_index is None:
            return torch.full(
                (batch_size,),
                self.seq_len % self.cycle_len,
                dtype=torch.long,
                device=x.device,
            )
        if not torch.is_tensor(sample_index):
            sample_index = torch.as_tensor(sample_index)
        sample_index = sample_index.to(device=x.device, dtype=torch.long)
        return (sample_index + self.seq_len) % self.cycle_len

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
        sample_index=None,
    ):
        x = x_enc

        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = x.permute(0, 2, 1)
        cycle_index = self._cycle_index(x, sample_index=sample_index)
        gather_index = (
            cycle_index.view(-1, 1)
            + torch.arange(self.seq_len, device=x.device).view(1, -1)
        ) % self.cycle_len
        query_input = self.Q[gather_index].permute(0, 2, 1)
        global_information = self.GTR(x_input, query_input)

        input_proj = self.input_proj(x_input + global_information)
        hidden = self.model(input_proj)
        output = self.output_proj(hidden + input_proj).permute(0, 2, 1)

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output[:, -self.pred_len:, :]
