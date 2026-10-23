import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimePatchChannelMaskedMultiheadAttention(nn.Module):
    def __init__(
        self,Cmax,dim,patch_len=1,stride = None,num_heads = 2,pad_value = 0.0,attn_nonlinear='tanh',
        num_post_layers=0,masked_attention=True,use_pos_emb=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} 必须能被 num_heads={num_heads} 整除"
        assert patch_len >= 1
        self.Cmax = Cmax
        self.dim = dim
        self.patch_len = patch_len
        self.masked_attention = masked_attention
        self.use_pos_emb = use_pos_emb
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_nonlinear = attn_nonlinear


        self.pos_emb = nn.Embedding(Cmax, dim)
        self.miss_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.miss_token, mean=0.0, std=0.02)

        # patch -> token：把长度 P 的时间片映射到 dim
        # 你也可以换成 Conv1d/MLP，这里先给最直的 Linear
        self.patch_proj = nn.Linear(patch_len, dim)

        self.outmlp = nn.Sequential(*[
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        ])
        self.session_emb = nn.Sequential(*[
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh()
        ])
        
        # 多头投影
        self.WQ = nn.Linear(dim, dim, bias=True)
        self.WK = nn.Linear(dim, dim, bias=True)
        self.WV = nn.Linear(dim, dim, bias=True)

        # token -> patch：把 dim decode 回长度 P
        # self.out_proj = nn.Linear(dim, patch_len)

        # 后续层：unmasked attention + mlp
        self.postmlp = nn.Sequential(*[
            nn.Linear(dim, dim, bias=True),
            nn.Tanh(),
            nn.Linear(dim, dim, bias=True)
        ])
        self.num_post_layers = num_post_layers
        self.post_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                'WQ': nn.Linear(dim, dim, bias=True),
                'WK': nn.Linear(dim, dim, bias=True),
                'WV': nn.Linear(dim, dim, bias=True),
                'mlp': nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.Tanh(),
                    nn.Linear(dim, dim)
                )
            })
            for _ in range(num_post_layers)
        ])

    def _run_attention(self, x, WQ, WK, WV):
        # x: (B,N,C,dim)
        B, N, C, D = x.shape
        BN = B * N
        H, Hd = self.num_heads, self.head_dim

        x = x.view(BN, C, D)

        Q = WQ(x)
        K = WK(x)
        V = WV(x)

        Q = Q.view(BN, C, H, Hd).transpose(1, 2)
        K = K.view(BN, C, H, Hd).transpose(1, 2)
        V = V.view(BN, C, H, Hd).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Hd)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, N, C, D)
        return out

    def forward(self, y, channel_mask,session_variability=True):
        """
        y: (B,T,C)
        """
        B, T, C = y.shape
        y_patch = y.unfold(dimension=1, size=self.patch_len, step=1) # (B,N,C,P) # N = T - P +1
        N = y_patch.shape[1]
        C = self.Cmax
        valid = channel_mask.view(B, 1, C).expand(B, N, C)

        # ---- 1) patch -> token + pos；missing 用 miss_token + pos ----
        # (B,N,C,P) -> (BN,C,P) -> embed (BN,C,dim)
        BN = B * N
        y_flat = y_patch.contiguous().view(BN, C, self.patch_len)                 # (BN,C,P)
        y_embed = torch.tanh(self.patch_proj(y_flat))                         # (BN,C,dim)

        miss = self.miss_token.view(1, 1, -1).expand(BN, C, -1)      # (BN,C,dim)
        v_flat = valid.contiguous().view(BN, C)                                   # (BN,C)
        h_obs = y_embed
        h_miss = miss
        if self.use_pos_emb:
            idx = torch.arange(C, device=y.device)
            pos = self.pos_emb(idx).unsqueeze(0).expand(BN, -1, -1)
            h_obs = h_obs + pos
            h_miss = h_miss + pos
        h_all = torch.where(v_flat.unsqueeze(-1)==1, h_obs, h_miss)     # (BN,C,dim)

        # 加上 session embedding
        if session_variability:
            h_all_session_ = h_all.view(B,N,C,self.dim)[:,0,:,:]
            h_all_session_emb = self.session_emb(h_all_session_)
            sess_mean = h_all_session_emb.mean(dim=2)
            h_all = h_all + h_all_session_emb[:, None, :, :].expand(B, N, C, self.dim).reshape(BN, C, self.dim)
        else:
            sess_mean = None

        # ---- 2) 多头 attention over channels ----
        Q = self.WQ(h_all)
        K = self.WK(h_all)
        V = self.WV(h_all)

        H, Hd = self.num_heads, self.head_dim
        Q = Q.view(BN, C, H, Hd).transpose(1, 2)  # (BN,H,C,Hd)
        K = K.view(BN, C, H, Hd).transpose(1, 2)
        V = V.view(BN, C, H, Hd).transpose(1, 2)
        # V = h_all.view(BN,C,H,Hd).transpose(1,2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Hd)           # (BN,H,C,C)

        # key mask：invalid key -> -inf
        valid_key = v_flat.view(BN, 1, 1, C)                         # (BN,1,1,C)
        if self.masked_attention:
            scores = scores.masked_fill(valid_key==0, float("-inf"))

        attn = F.softmax(scores, dim=-1)                             # (BN,H,C,C)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = attn @ V                                               # (BN,H,C,Hd)
        out = out.transpose(1, 2).contiguous().view(B, N, C, self.dim) # (BN,C,dim)
        
        # out = self.outmlp(out)
        if self.attn_nonlinear == 'tanh':
            y_hat = torch.tanh(out)
        elif self.attn_nonlinear == 'relu':
            y_hat = F.relu(out)
        elif self.attn_nonlinear == 'none':
            y_hat = out
        if self.num_post_layers > 0:
            y_hat = self.postmlp(y_hat)
            for layer in self.post_attn_layers:
                attn_out = self._run_attention(y_hat, layer['WQ'], layer['WK'], layer['WV'])
                y_hat = y_hat + attn_out
                y_hat = y_hat + layer['mlp'](y_hat)

        return y_hat, sess_mean
    
