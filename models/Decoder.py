import torch
import torch.nn as nn


class ConditionalLoRADecoder(nn.Module):
    """
    trial-conditioned low-rank weight update

    y = (W0 + U diag(s(trial)) V) z + (b0 + db(trial))

    - W0,b0: shared base Linear
    - U,V: shared low-rank factors
    - s(trial), db(trial): generated from trial embedding
    """
    def __init__(
        self, z_dim, y_dim, pe_dim = 32, rank = 8, hidden = 64, device = "cuda:0",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.rank = rank
        self.base = nn.Linear(z_dim, y_dim, bias=True)
        self.output = nn.Linear(y_dim, y_dim, bias=True)
        # 由 embedding 生成 low-rank 系数 s(trial) 和 bias shift db(trial)
        self.to_s = nn.Sequential(
            nn.Linear(pe_dim, rank)
        )

        self.to_db = nn.Sequential(
            nn.Linear(pe_dim, y_dim)
        )

        # 关键：让模型一开始“退化回 base”，训练更稳（避免一上来就乱转）
        nn.init.zeros_(self.to_s[-1].weight)
        nn.init.zeros_(self.to_s[-1].bias)
        nn.init.zeros_(self.to_db[-1].weight)
        nn.init.zeros_(self.to_db[-1].bias)

        # 共享低秩因子 U,V
        # U: (C, r), V: (r, z)
        self.U = nn.Parameter(torch.randn(y_dim, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, z_dim) * 0.02)

    def forward(self, z, trial_pe = None):
        """
        z: (B, T, z_dim) 或 (B, z_dim)
        trial_id: (B,y_dim,pe_dim)  每个样本对应一个 trial index

        如果 trial_id=None，则退化为共享 base decoder（方便你逐步迁移代码）。
        """
        z = z.to(self.device)

        # 兼容模式：不传 trial_id 就等价于原 decoder
        if trial_pe is None:
            return self.base(z)

        y0 = self.base(z)                    # (B,T,C) or (B,C)
        s = self.to_s(trial_pe)              # (B, r)
        db = self.to_db(trial_pe)            # (B, C)

        # 处理 (B,T,*) 广播
        if z.dim() == 3:
            s = s[:, None, :]    # (B,1,r)
            db = db[:, None, :]  # (B,1,C)


        z_r = torch.matmul(z, self.V.t())
        z_r = z_r * s
        delta = torch.matmul(z_r, self.U.t())
        return y0+delta+db
