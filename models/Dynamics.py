import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, sampling_rate = 500,hid=128, depth=1):
        super().__init__()
        layers = []
        d = in_dim
        self.out_dim = out_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hid), nn.Tanh()]
            d = hid
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.sampling_rate = sampling_rate
    def forward(self, x): 
        return x[:, :self.out_dim]+1/self.sampling_rate*self.net(torch.relu(x))

class LatentDynamics(nn.Module):
    """
    x: (B, T, Xdim) or any encoder-accepted shape
    Encode observations and predict subsequent latent states.
    """
    def __init__(self, encoder, z_dim, num_conditions=0,sampling_rate=500,input_time=2):
        super().__init__()
        self.encoder = encoder                      # f_theta (train)
        self.z_dim = z_dim
        self.predictor = MLP(in_dim=z_dim+num_conditions, out_dim=z_dim, sampling_rate=sampling_rate, hid=z_dim, depth=3)
        self.input_time=input_time
    def forward(self, x, TF, condition_idx=None):

        z_enc = self.encoder(x)
        B, T = z_enc.shape[0], z_enc.shape[1]
        
        if TF < 0:
            return z_enc[:,1:,:], z_enc
        
        pred_list = [] # 存每一步的预测 z_hat，对应时刻 t = 0..T-1
        if condition_idx is not None:
            c = condition_idx.to(z_enc.device)
            zero_c = torch.zeros_like(c)

        for t in range(T):
            if TF is not None and TF > 0 and (t % TF) == 0:
                z = z_enc[:, t, :]

            if condition_idx is None:
                z = self.predictor(z)
            else:
                c_t = c if (t < self.input_time) else zero_c
                z = self.predictor(torch.cat([z, c_t], dim=-1))
            pred_list.append(z)

        # 组装输出
        z_pred_all = torch.stack(pred_list, dim=1)      # (B, T, z_dim) # 纯预测（每一步都有）


        return z_pred_all[:,:-1,:], z_enc
