
import torch
import torch.nn as nn
from models.FullModel import *
import datetime
import os
from utils.DataSample import Input_Data
from torch.utils.data import DataLoader 
from utils.pytorchtools import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
import random
import numpy as np
today = datetime.date.today().strftime("%Y%m%d")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data._utils.collate import default_collate
from utils.metrics_calculation import evaluate_model_training_real





def split_dataset(data, trial_id,condition_idx, mask, training_mask, gt_mask, beh_data, test_size=0.2, random_state=42):

    n_samples = data.shape[0]
    indices = np.arange(n_samples)

    # 随机分割索引
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state,shuffle=False)
    # 分割数据
    train_data = data[train_idx]
    test_data = data[test_idx]

    train_beh = beh_data[train_idx]
    test_beh = beh_data[test_idx]

    train_trial_id = trial_id[train_idx]
    test_trial_id = trial_id[test_idx]

    train_condition_idx = condition_idx[train_idx]
    test_condition_idx = condition_idx[test_idx]

    train_mask = mask[train_idx] if mask is not None else None
    test_mask = mask[test_idx] if mask is not None else None

    train_training_mask = training_mask[train_idx] if training_mask is not None else None
    test_training_mask = training_mask[test_idx] if training_mask is not None else None

    train_gt_mask = gt_mask[train_idx] if gt_mask is not None else None
    test_gt_mask = gt_mask[test_idx] if gt_mask is not None else None
    return (train_data, test_data, train_beh, test_beh, train_trial_id, test_trial_id, train_condition_idx, test_condition_idx, train_mask, test_mask, train_training_mask, test_training_mask, train_gt_mask, test_gt_mask, train_idx, test_idx)

def random_augment_channel_masked_data_ratio(train_data, train_trial_id, train_beh_data, train_condition_idx, training_mask, channel_mask, gt_mask, n_aug=1,
    mask_ratio=0.2, mask_ratio_range=None,rounding="round", generator=None):
    """
    增强（按比例mask多个通道）：
    - 只要该 trial 可见通道数 > 1（channel_mask.sum > 1）
    - 在候选通道（可见且training_mask原本为1）里，随机mask掉一定比例
    - 保证至少保留1个候选通道为1（避免全0训练mask）

    参数:
      train_data:      (B, T, C)
      training_mask:   (B, C)  1=参与训练/可用, 0=mask掉
      channel_mask:    (B, C)  1=该trial该通道可见, 0=不可见
      n_aug:           每个符合条件的trial生成多少条增强样本
      mask_ratio:      固定mask比例（0~1）
      mask_ratio_range:可选，(low, high)，每条增强样本随机采样mask比例
      rounding:        比例->数量的取整方式
      generator:       torch.Generator
    返回:
      augmented_train_data, augmented_training_mask, augmented_channel_mask
    """

    B, T, C = train_data.shape

    augmented_data_list = [train_data]
    augmented_beh_data_list = [train_beh_data]
    augmented_trial_id_list = [train_trial_id]
    augmented_training_mask_list = [training_mask]
    augmented_channel_mask_list = [channel_mask]
    augmented_gt_mask_list = [gt_mask]
    augmented_condition_idx_list = [train_condition_idx]

    visible_counts = channel_mask.sum(dim=1)  # (B,)

    def ratio_to_m(k, r):
        r = float(r)
        r = max(0.0, min(1.0, r))
        x = k * r
        if rounding == "floor":
            m = int(torch.floor(torch.tensor(x)).item())
        elif rounding == "ceil":
            m = int(torch.ceil(torch.tensor(x)).item())
        else:
            m = int(torch.round(torch.tensor(x)).item())

        # 至少mask 1个，但最多mask到只剩1个（避免全0）
        m = max(1, m)
        m = min(k - 1, m)
        return m

    for b in range(B):
        if visible_counts[b].item() <= 1:
            continue

        candidate = (channel_mask[b] > 0) & (training_mask[b] > 0)
        cand_idx = torch.nonzero(candidate, as_tuple=False).squeeze(-1)  # (K,)
        K = cand_idx.numel()

        # 候选通道<=1：没法按比例mask且仍保留至少1个
        if K <= 1:
            continue

        for _ in range(n_aug):
            # 采样一个比例
            if mask_ratio_range is not None:
                low, high = mask_ratio_range
                if high < low:
                    low, high = high, low
                r = torch.empty(1).uniform_(low, high, generator=generator).item()
            else:
                r = mask_ratio

            m = ratio_to_m(K, r)  # 要mask掉的通道数

            # 随机选m个候选通道置0
            perm = torch.randperm(K, generator=generator)
            drop_idx = cand_idx[perm[:m]]

            trial_data = train_data[b:b+1]                    # (1, T, C)
            trial_beh_data = train_beh_data[b:b+1]
            trial_id = train_trial_id[b:b+1]
            trial_condition_idx = train_condition_idx[b:b+1]
            trial_channel_mask = channel_mask[b:b+1]          # (1, C)
            trial_gt_mask = gt_mask[b:b+1]
            new_training_mask = training_mask[b:b+1].clone()  # (1, C)
            new_training_mask[:, drop_idx] = 0

            augmented_data_list.append(trial_data)
            augmented_beh_data_list.append(trial_beh_data)
            augmented_trial_id_list.append(trial_id)
            augmented_condition_idx_list.append(trial_condition_idx)
            augmented_training_mask_list.append(new_training_mask)
            augmented_channel_mask_list.append(trial_channel_mask)
            augmented_gt_mask_list.append(trial_gt_mask)

    augmented_train_data = torch.cat(augmented_data_list, dim=0)
    augmented_beh_data = torch.cat(augmented_beh_data_list, dim=0)
    augmented_trial_id = torch.cat(augmented_trial_id_list, dim=0)
    augmented_condition_idx = torch.cat(augmented_condition_idx_list, dim=0)
    augmented_training_mask = torch.cat(augmented_training_mask_list, dim=0)
    augmented_channel_mask = torch.cat(augmented_channel_mask_list, dim=0)
    augmented_gt_mask = torch.cat(augmented_gt_mask_list, dim=0)
    return augmented_train_data, augmented_trial_id, augmented_beh_data, augmented_condition_idx, augmented_training_mask, augmented_channel_mask, augmented_gt_mask

def train_test_model(model, optimizer, data, behavior_data, trial_id,condition_idx,mask, training_mask, gt_mask, ckpts, epochs, augment = True,n_aug=1, earlystop_epochs=100, test_size=0.2, val_size=0.1,random_state=42, num_workers=24, batch_size=50, shuffle_train=True, shuffle_eval=True, earlystop_patience=50, earlystop_delta=0, earlystop_verbose=True, grad_clip_norm=1.0, eval_interval=50, mask_ratio=0.2, mask_ratio_range=(0.3,0.7), augmentation_rounding="round"):
    def seed_worker(worker_id):
        worker_seed = random_state + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    # 1. 分割数据集
    print("Splitting dataset...")
    (train_data, test_data, train_beh_data, test_beh_data, train_trial_id,test_trial_id, train_condition_idx, test_condition_idx, train_mask, test_mask, 
        train_training_mask, test_training_mask, train_gt_mask, test_gt_mask, train_idx, test_idx) = split_dataset(data, trial_id,condition_idx, mask, training_mask, gt_mask, beh_data=behavior_data, test_size=test_size,random_state=random_state)

    # 2. 从训练集中再分割出验证集
    if val_size > 0:
        (train_data, val_data, train_beh_data, val_beh_data, train_trial_id, val_trial_id, train_condition_idx, val_condition_idx, train_mask, val_mask, 
        train_training_mask, val_training_mask, train_gt_mask, val_gt_mask, train_idx, val_idx) = split_dataset(train_data,train_trial_id, train_condition_idx,  train_mask, train_training_mask, train_gt_mask, beh_data=train_beh_data, test_size=val_size/(1-test_size),random_state=random_state)
    else:
        val_data, val_mask, val_training_mask, val_gt_mask = None, None, None


    if augment:
        g = torch.Generator()
        g.manual_seed(random_state)
        train_data, train_trial_id, train_beh_data, train_condition_idx, train_training_mask, train_mask, train_gt_mask = random_augment_channel_masked_data_ratio(train_data, train_trial_id, train_beh_data, train_condition_idx, train_training_mask, train_mask,train_gt_mask, n_aug=n_aug,mask_ratio=mask_ratio,mask_ratio_range=mask_ratio_range,rounding=augmentation_rounding,generator=g)


    # g_loader = torch.Generator()
    # g_loader.manual_seed(random_state)

    # 3. 创建数据加载器
    train_dataset = Input_Data(train_data, None, train_trial_id, train_mask, train_training_mask, condition_idx=train_condition_idx, beh_data=train_beh_data, gt_mask=train_gt_mask)
    # train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=24,worker_init_fn=seed_worker,generator=g_loader)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    if val_data is not None:
        val_dataset = Input_Data(val_data, None, val_trial_id, val_mask, val_training_mask, condition_idx=val_condition_idx, beh_data=val_beh_data, gt_mask=val_gt_mask)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_eval, num_workers=num_workers)

    test_dataset = Input_Data(test_data, None, test_trial_id, test_mask, test_training_mask, condition_idx=test_condition_idx, beh_data=test_beh_data, gt_mask=test_gt_mask)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_eval, num_workers=num_workers)

    # 4. 初始化早停
    early_stopping = EarlyStopping(patience=earlystop_patience, delta=earlystop_delta, path=ckpts, verbose=earlystop_verbose, trace_func=tqdm.write)

    # 5. 训练循环
    pbar = tqdm(range(epochs), desc="Training initialization ... ", unit="epoch", dynamic_ncols=True)

    # 存储训练历史
    history = {
        'train_loss': [],'val_loss': [],'train_r2': [],'val_r2': [],
        'test_r2': [],'train_ev': [],'val_ev': [],'test_ev': []
    }

    for epoch in pbar:
        # 训练阶段
        model.train()
        train_loss_epoch = 0
        train_recon_loss_epoch = 0
        train_dyn_loss_epoch = 0
        train_beh_loss_epoch = 0
        for i, (y_batch, trial_id_batch, condition_idx_batch, behavior_batch, mask_batch, training_mask_batch, gt_mask_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred, y_recon, z_pred, z, y_behavior, loss, recon_loss, dyn_loss, beh_loss = model(y_batch, mask_batch, training_mask_batch,trial_id_batch, condition_idx_batch, behavior_batch)

            # backward
            # loss+= sum(p.pow(2).sum() for p in model.parameters()) * 0.0001  
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            train_loss_epoch += loss.item()
            train_recon_loss_epoch += recon_loss.item()
            train_dyn_loss_epoch += dyn_loss.item()
            train_beh_loss_epoch += beh_loss.item()
        # 计算平均训练损失
        avg_train_loss = train_loss_epoch / len(train_loader)
        avg_train_recon_loss = train_recon_loss_epoch / len(train_loader)
        avg_train_dyn_loss = train_dyn_loss_epoch / len(train_loader)
        avg_train_beh_loss = train_beh_loss_epoch / len(train_loader)
        # 验证阶段（每50个epoch或在验证集上）
        if epoch % eval_interval == 0:
            # 训练集评估
            train_r2, train_ev, _, _, train_beh_r2, train_beh_ev, avg_train_loss = evaluate_model_training_real(model, train_loader)

            # 验证集评估（如果有）
            if val_data is not None:
                val_r2, val_ev,  _, _, val_beh_r2, val_beh_ev, avg_val_loss = evaluate_model_training_real(model, val_loader)
                history['val_loss'].append(avg_val_loss)  # 使用训练损失作为参考
                history['val_r2'].append(val_r2)
                history['val_ev'].append(val_ev)

            # 测试集评估（仅用于监控，不用于早停）
            test_r2, test_ev,  _, _, test_beh_r2, test_beh_ev, avg_test_loss = evaluate_model_training_real(model, test_loader)

            # 更新历史记录
            history['train_loss'].append(avg_train_loss)
            history['train_r2'].append(train_r2)
            history['train_ev'].append(train_ev)
            history['test_r2'].append(test_r2)
            history['test_ev'].append(test_ev)

            # 打印进度
            log_message = (
                f"[{epoch+1}/{epochs}] "
                f"Loss={avg_train_loss:.3f}, "
                f"recon_loss={avg_train_recon_loss:.6f}, "
                f"dyn_loss={avg_train_dyn_loss:.6f}, "
                f"beh_loss={avg_train_beh_loss:.3f}, "
                f"Train_R2={train_r2:.3f}, "
                f"Train_EV={train_ev:.3f}, "
                f"Train_Beh_R2={train_beh_r2:.3f}, "
                f"Train_Beh_EV={train_beh_ev:.3f}"
            )

            if val_data is not None:
                log_message += f", Val_R2={val_r2:.3f}, Val_EV={val_ev:.3f}"


            log_message += f", Test_R2={test_r2:.3f}, Test_EV={test_ev:.3f}"
            log_message += f"  Test_Beh_R2={test_beh_r2:.3f}, Test_Beh_EV={test_beh_ev:.3f}"
            tqdm.write(log_message)

            # 更新进度条描述
            pbar.set_description(f"Train Loss: {avg_train_loss:.3f}, "f"Train R2: {train_r2:.3f}")

        # 早停检查（使用验证集损失或训练集损失）
        if epoch > earlystop_epochs:

            if val_data is not None:
                val_r2, val_ev,  _, _, val_beh_r2, val_beh_ev, avg_val_loss = evaluate_model_training_real(model, val_loader)
                # 使用验证集损失进行早停
                early_stopping(avg_val_loss, model)  # 这里可以使用val_loss
            else:
                train_r2, train_ev,  _, _, train_beh_r2, train_beh_ev, avg_train_loss = evaluate_model_training_real(model, train_loader)
                # 没有验证集时使用训练集损失
                early_stopping(avg_train_loss, model)

            if early_stopping.early_stop:
                pbar.set_description("Early stopping triggered")
                break

    # 6. 最终测试
    # print("\n" + "="*50)
    # print("Final Evaluation:")
    # print("="*50)

    model.load_state_dict(torch.load(ckpts))
    print('Training_set evaluation:')
    final_train_r2, final_train_ev,  _, _, final_train_beh_r2, final_train_beh_ev, _ = evaluate_model_training_real(model, train_loader, final_eval=True)

    print('\n')
    print('Test set evaluation:')
    final_test_r2, final_test_ev ,  _, _,  final_test_beh_r2, final_test_beh_ev,_= evaluate_model_training_real(model, test_loader, final_eval=True)

    if val_data is not None:
        print('\n')
        print('Evaluation set evaluation:')
        final_val_r2, final_val_ev, _, _,   final_val_beh_r2, final_val_beh_ev,_ = evaluate_model_training_real(model, val_loader, final_eval=True)

    return model, history, (train_idx, test_idx,val_idx if val_size>0 else None), train_loader, test_loader, val_loader


