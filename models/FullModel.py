from models.Dynamics import *
from models.Encoder import *
from models.Decoder import *
from models.masked_attention import *
def apply_output_nonlinear(x, name):
    name = (name or "").lower()
    if name in ["none", "linear", "identity", ""]:
        return x
    elif name == "sigmoid":
        return torch.sigmoid(x)
    elif name == "tanh":
        return torch.tanh(x)
    elif name == "softplus":
        return torch.nn.Softplus()(x)
    elif name == "relu":
        return F.relu(x)
    else:
        raise ValueError(f"Unknown output_nonlinear: {name}")
    
class FullModel(nn.Module):
    def __init__(self, z_dim, Cmax, attn_dim,behavior_dim=2,hidden_dim = 32, num_post_layers=0,
                 num_conditions=0,tf=1,warmup_steps=0,sampling_rate =500,LoRA_rank=8,input_time=2,beh_delay=5,
                 attn_nonlinear = 'tanh',y_nonlinear='none',beh_nonlinear='tanh',session_variability=True,
                 dyn_loss_weight=1.0, recon_loss_weight=1.0, pred_loss_weight=1.0, beh_weight =0,
                  device="cuda:0",criterion_mode='MSE',masked_attention=True,use_pos_emb=True):
        super().__init__()

        self.device = torch.device(device)
        if criterion_mode == 'MSE':
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            self.criterion = nn.SmoothL1Loss(reduction='mean', beta=0.001)
        self.dyn_loss_weight = dyn_loss_weight
        self.recon_loss_weight = recon_loss_weight  
        self.pred_loss_weight = pred_loss_weight
        self.session_variability = session_variability
        self.y_nonlinear = y_nonlinear
        self.beh_nonlinear = beh_nonlinear
        self.beh_weight = beh_weight
        self.beh_delay = beh_delay  # 行为预测相对于神经预测的延迟（单位时间步）
        self.tf = tf
        self.warmup = warmup_steps
        
        self.attention = TimePatchChannelMaskedMultiheadAttention(Cmax,dim=attn_dim,patch_len=warmup_steps,attn_nonlinear = attn_nonlinear,num_post_layers=num_post_layers,masked_attention=masked_attention,use_pos_emb=use_pos_emb)
        self.encoder = MLPEncoder(channel=Cmax, emb = attn_dim, hidden_dim=z_dim,mlp_hidden_dim=hidden_dim,dropout=0.0)
        self.decoder = ConditionalLoRADecoder(z_dim=z_dim,y_dim=Cmax,rank=LoRA_rank,pe_dim=Cmax,hidden=hidden_dim,device=device)
        self.dynamics = LatentDynamics(self.encoder, z_dim,num_conditions=num_conditions, sampling_rate=sampling_rate,input_time=input_time)

        self.decoder_behavior = ConditionalLoRADecoder(z_dim=z_dim,y_dim=behavior_dim,rank=LoRA_rank,pe_dim=Cmax,hidden=hidden_dim,device=device)
        
            

    def forward(self, y, channel_mask,training_mask,trial_id=None,condition_idx=None, behavior = None):

        y = y.to(self.device, dtype=torch.float32)
        channel_mask = channel_mask.to(self.device)
        training_mask = training_mask.to(self.device)
        
        y_new, session_emb = self.attention(y, training_mask,self.session_variability)  # (B,T-P+1,Cmax,D), (B,Cmax)
        z_pred, z_enc = self.dynamics(y_new, TF=self.tf,condition_idx=condition_idx) # (B, T-P, z_dim)
        if self.session_variability:
            y_pred = self.decoder(z_pred, session_emb)  # [B,T-P, y_dim]
            y_recon = self.decoder(z_enc, session_emb)  # [B,T-P+1, y_dim]
        else:
            y_pred = self.decoder(z_pred, None)  # [B,T-P, y_dim]
            y_recon = self.decoder(z_enc, None)  # [B,T-P+1, y_dim]

        y_pred = apply_output_nonlinear(y_pred,self.y_nonlinear)
        y_recon = apply_output_nonlinear(y_recon,self.y_nonlinear)

        if behavior is not None:
            if session_emb is not None:
                # y_behavior = self.decoder_behavior(z_pred,session_emb)
                y_behavior = self.decoder_behavior(z_pred,session_emb)
            else:
                y_behavior = self.decoder_behavior(z_pred,None)
            y_behavior = apply_output_nonlinear(y_behavior,self.beh_nonlinear)
        
        mask = channel_mask[:, None, :].to(y_pred.dtype)  # (B, 1, C)  1有效0无效
        mask = mask.expand_as(y_pred)                     # (B, T-1, C)
        y_pred_m = y_pred * mask
        
        mask = channel_mask[:, None, :].to(y_pred.dtype)  # (B, 1, C)  1有效0无效
        mask = mask.expand_as(y[:, self.warmup:, :])
        y_tgt_m  = y[:, self.warmup:, :] * mask

        mask = channel_mask[:, None, :].to(y.dtype)  # (B, 1, C)  1有效0无效
        mask = mask.expand_as(y_recon)  
        y_recon_m = y_recon *mask

        mask = channel_mask[:, None, :].to(y.dtype)  # (B, 1, C)  1有效0无效
        mask = mask.expand_as(y[:, self.warmup-1:, :])  
        y_true_m = y[:, self.warmup-1:, :] * mask
        
        recon_loss = self.criterion(y_recon_m,y_true_m) # self.criterion(y_true_m, y_recon_m)
        pred_loss = self.criterion(y_pred_m, y_tgt_m)

        dyn_loss = self.criterion(z_pred, z_enc[:, 1:, :])
        loss = self.recon_loss_weight*recon_loss + self.dyn_loss_weight*dyn_loss + self.pred_loss_weight*pred_loss

        if behavior is not None:
            # print(y_behavior.shape,behavior.shape)
            behavior = behavior.to(self.device, dtype=torch.float32)
            if self.beh_delay > 0:
                behavior_pred = y_behavior[:, :-self.beh_delay, :]
                behavior_target = behavior[:, self.warmup + self.beh_delay:, :]
            else:
                behavior_pred = y_behavior
                behavior_target = behavior[:, self.warmup:, :]

            behavior_loss = self.criterion(behavior_pred,behavior_target)
            loss += self.beh_weight*behavior_loss
            return y_pred, y_recon, z_pred, z_enc, y_behavior, loss, recon_loss, dyn_loss, behavior_loss
        else:
            return y_pred, y_recon, z_pred, z_enc,loss, recon_loss, dyn_loss
    
