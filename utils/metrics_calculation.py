from json import decoder

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score
import random
from sklearn.linear_model import Ridge
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def evaluate_zpred_gt_dyn_linear_all(model, data_loader):
    model.eval()

    z_list = []
    gt_dyn_list = []

    with torch.no_grad():
        for y, gt_dyn, trial_id, mask, training_mask in data_loader:
            y_pred, y_recon, z_pred, z, loss, recon_loss, dyn_loss = model(y, mask, training_mask, trial_id)
            z_list.append(z_pred.detach().cpu())
            gt_dyn_list.append(gt_dyn.detach().cpu())

    z_pred_all = torch.cat(z_list, dim=0)
    gt_dyn_all = torch.cat(gt_dyn_list, dim=0)

    # 如果 gt_dyn 需要和 z_pred 对齐，改成：
    gt_dyn_all = torch.cat([g[:, model.warmup:, :].detach().cpu() for g in gt_dyn_list], dim=0)

    X = z_pred_all.reshape(-1, z_pred_all.shape[-1])
    Y = gt_dyn_all.reshape(-1, gt_dyn_all.shape[-1])

    X = to_numpy(X)
    Y = to_numpy(Y)

    reg = LinearRegression()
    reg.fit(X, Y)

    Y_pred = reg.predict(X)

    r2_all = r2_score(Y, Y_pred, multioutput="uniform_average")
    r2_raw = r2_score(Y, Y_pred, multioutput="raw_values")

    print("Linear decoding z_pred -> gt_dyn")
    print(f"Overall R2: {r2_all:.4f}")
    print(f"R2 per gt_dyn dim: {r2_raw}")

    model.train()

    return {
        "r2_all": r2_all,
        "r2_raw": r2_raw,
    }

def evaluate_model_training(model, data_loader, final_eval=False):
    model.eval()
    r2_list, ev_list, obser_r2_list, unobser_r2_list, obser_ev_list, unobser_ev_list = ([] for _ in range(6))
    loss_epoch = recon_loss_epoch = dyn_loss_epoch = 0
    z_list, m_list, yp_list, yt_list = [], [], [], []
    with torch.no_grad():
        for i, (y, gt_dyn, trial_id, mask, training_mask) in enumerate(data_loader):
            y_pred, y_recon, z_pred, z, loss, recon_loss, dyn_loss = model(y, mask, training_mask,trial_id)
            loss_epoch += loss.item()
            recon_loss_epoch += recon_loss.item()
            dyn_loss_epoch += dyn_loss.item()
            z_list.append(z_pred.detach().cpu())
            m_list.append(mask.detach().cpu())
            yp_list.append(y_pred.detach().cpu())
            yt_list.append(y[:, model.warmup:].detach().cpu())
            # 计算每个样本的R²和EV
            for j in range(y.shape[0]):
                y_system = y[j, model.warmup:, :]
                y_hat_system = y_pred[j, :, :]
                mask_ = mask[j]
                y_true = y_system.detach().cpu().numpy().flatten()
                y_pred_flat = y_hat_system.detach().cpu().numpy().flatten()
                r2 = r2_score(y_true, y_pred_flat)
                ev = explained_variance_score(y_true, y_pred_flat)
                r2_list.append(r2)
                ev_list.append(ev)

                if final_eval:
                    obser_list = np.where(mask_.cpu().numpy()==1)[0]
                    unobser_list = np.where(mask_.cpu().numpy()==0)[0]
                    y_true = y_system[:,obser_list].detach().cpu().numpy().flatten()
                    y_pred_flat = y_hat_system[:,obser_list].detach().cpu().numpy().flatten()
                    r2 = r2_score(y_true, y_pred_flat)
                    ev = explained_variance_score(y_true, y_pred_flat)
                
                    obser_r2_list.append(r2)
                    obser_ev_list.append(ev)
                    if unobser_list.shape[0]==0:
                        continue
                    y_true_unobser = y_system[:,unobser_list].detach().cpu().numpy().flatten()
                    y_pred_flat_unobser = y_hat_system[:,unobser_list].detach().cpu().numpy().flatten()

                    r2_unobser = r2_score(y_true_unobser, y_pred_flat_unobser)
                    ev_unobser = explained_variance_score(y_true_unobser, y_pred_flat_unobser)
                    unobser_r2_list.append(r2_unobser)
                    unobser_ev_list.append(ev_unobser)

                # 计算平均训练损失
            z_preds = torch.cat(z_list, dim=0)
            masks   = torch.cat(m_list, dim=0)
            y_preds = torch.cat(yp_list, dim=0)
            y_gt    = torch.cat(yt_list, dim=0)
        avg_loss =loss_epoch / len(data_loader)
        avg_recon_loss = recon_loss_epoch / len(data_loader)
        avg_dyn_loss = dyn_loss_epoch / len(data_loader)
        # 计算平均值
    if len(r2_list) > 0 and len(ev_list) > 0:
        avg_r2 = np.mean(r2_list)
        avg_ev = np.mean(ev_list)
        std_r2 = np.std(r2_list)
        std_ev = np.std(ev_list)
    else:
        avg_r2 = avg_ev = std_ev = std_r2 = 0.0

    
    if final_eval:
        avg_obser_r2 = np.mean(obser_r2_list) if len(obser_r2_list)>0 else 0.0
        avg_unobser_r2 = np.mean(unobser_r2_list) if len(unobser_r2_list)>0 else 0.0
        avg_obser_ev = np.mean(obser_ev_list) if len(obser_ev_list)>0 else 0.0
        avg_unobser_ev = np.mean(unobser_ev_list) if len(unobser_ev_list)>0 else 0.0

        std_obser_r2 = np.std(obser_r2_list) if len(obser_r2_list)>0 else 0.0
        std_unobser_r2 = np.std(unobser_r2_list) if len(unobser_r2_list)>0 else 0.0
        std_obser_ev = np.std(obser_ev_list) if len(obser_ev_list)>0 else 0.0
        std_unobser_ev = np.std(unobser_ev_list) if len(unobser_ev_list)>0 else 0.0

        print(f"Observed Channels - R²: {avg_obser_r2:.4f} ± {std_obser_r2:.4f}, EV: {avg_obser_ev:.4f} ± {std_obser_ev:.4f}")
        print(f"Unobserved Channels - R²: {avg_unobser_r2:.4f} ± {std_unobser_r2:.4f}, EV: {avg_unobser_ev:.4f} ± {std_unobser_ev:.4f}")
        print(f"Total Channels - R²: {avg_r2:.4f} ± {std_r2:.4f}, EV: {avg_ev:.4f} ± {std_ev:.4f}")
        return avg_obser_r2, avg_obser_ev, avg_loss , z_preds, masks,y_preds, y_gt
    
    else:
        model.train()  # 恢复训练模式
        return avg_r2, avg_ev, avg_loss
    
def evaluate_model_final(model, data_loader):
    model.eval()

    r2_list, ev_list = [], []
    obser_r2_list, unobser_r2_list = [], []
    obser_ev_list, unobser_ev_list = [], []


    z_list, m_list, yp_list, yt_list = [], [], [], []

    with torch.no_grad():
        for i, (y, gt_dyn, trial_id, mask, training_mask) in enumerate(data_loader):
            y_pred, y_recon, z_pred, z, loss, recon_loss, dyn_loss = model(y, mask, training_mask, trial_id)

            z_list.append(z_pred.detach().cpu())
            m_list.append(mask.detach().cpu())
            yp_list.append(y_pred.detach().cpu())
            yt_list.append(y[:, model.warmup:].detach().cpu())

            for j in range(y.shape[0]):
                y_system = y[j, model.warmup:, :]
                y_hat_system = y_pred[j, :, :]
                mask_ = mask[j]

                y_true = y_system.detach().cpu().numpy().flatten()
                y_pred_flat = y_hat_system.detach().cpu().numpy().flatten()

                r2 = r2_score(y_true, y_pred_flat)
                ev = explained_variance_score(y_true, y_pred_flat)

                r2_list.append(r2)
                ev_list.append(ev)

                obser_list = np.where(mask_.cpu().numpy() == 1)[0]
                unobser_list = np.where(mask_.cpu().numpy() == 0)[0]

                if len(obser_list) > 0:
                    y_true_obs = y_system[:, obser_list].detach().cpu().numpy().flatten()
                    y_pred_obs = y_hat_system[:, obser_list].detach().cpu().numpy().flatten()

                    obser_r2_list.append(r2_score(y_true_obs, y_pred_obs))
                    obser_ev_list.append(explained_variance_score(y_true_obs, y_pred_obs))

                if len(unobser_list) > 0:
                    y_true_unobs = y_system[:, unobser_list].detach().cpu().numpy().flatten()
                    y_pred_unobs = y_hat_system[:, unobser_list].detach().cpu().numpy().flatten()

                    unobser_r2_list.append(r2_score(y_true_unobs, y_pred_unobs))
                    unobser_ev_list.append(explained_variance_score(y_true_unobs, y_pred_unobs))

    # dataloader 跑完之后再 concat
    z_preds = torch.cat(z_list, dim=0) if len(z_list) > 0 else None
    masks   = torch.cat(m_list, dim=0) if len(m_list) > 0 else None
    y_preds = torch.cat(yp_list, dim=0) if len(yp_list) > 0 else None
    y_gt    = torch.cat(yt_list, dim=0) if len(yt_list) > 0 else None

    metrics = {
        "avg_r2": np.mean(r2_list) if len(r2_list) > 0 else 0.0,
        "avg_ev": np.mean(ev_list) if len(ev_list) > 0 else 0.0,
        "avg_obser_r2": np.mean(obser_r2_list) if len(obser_r2_list) > 0 else 0.0,
        "avg_unobser_r2": np.mean(unobser_r2_list) if len(unobser_r2_list) > 0 else 0.0,
        "avg_obser_ev": np.mean(obser_ev_list) if len(obser_ev_list) > 0 else 0.0,
        "avg_unobser_ev": np.mean(unobser_ev_list) if len(unobser_ev_list) > 0 else 0.0,
    }

    model.train()

    return metrics, z_preds, masks, y_preds, y_gt

def evaluate_model_training_real(model, data_loader,final_eval=False):
    model.eval()
    r2_list, ev_list, beh_r2_list, beh_ev_list, obser_r2_list, unobser_r2_list, obser_ev_list, unobser_ev_list = ([] for _ in range(8))
    recon_loss_epoch, dyn_loss_epoch, beh_loss_epoch, loss_epoch = 0, 0, 0, 0

    with torch.no_grad():
        for i, (y,trial_id,condition_idx, behavior, mask, training_mask, gt_mask) in enumerate(data_loader):
            y_pred, y_recon, z_pred, z, y_behavior, loss, recon_loss, dyn_loss, beh_loss = model(y, mask, training_mask,trial_id, condition_idx,behavior=behavior)
            loss_epoch += loss.item()
            recon_loss_epoch += recon_loss.item()
            dyn_loss_epoch += dyn_loss.item()
            beh_loss_epoch += beh_loss.item()

            # 计算每个样本的R²和EV
            for j in range(y.shape[0]):
                mask_ = mask[j]
                gt_mask_ = gt_mask[j]
                y_system = y[j, model.warmup:, :]
                y_hat_system = y_pred[j, :, :]
                y_true = y_system.detach().cpu().numpy().T #flatten()
                y_pred_flat = y_hat_system.detach().cpu().numpy().T #flatten()
                r2 = r2_score(y_true, y_pred_flat)
                ev = explained_variance_score(y_true, y_pred_flat)
                r2_list.append(r2)
                ev_list.append(ev)
                if model.beh_delay > 0:
                    target = y_behavior[j, :-model.beh_delay, :]
                    pred = behavior[j, model.warmup + model.beh_delay:, :]
                else:
                    target = y_behavior[j, :, :]
                    pred = behavior[j, model.warmup:, :]

                beh_ev_list.append(explained_variance_score(pred.detach().cpu().numpy().flatten(),target.detach().cpu().numpy().flatten()))
                beh_r2_list.append(r2_score(pred.detach().cpu().numpy().flatten(),target.detach().cpu().numpy().flatten()))

                if final_eval:
                    obser_list = np.where(mask_.cpu().numpy()==1)[0]
                    unobser_list = np.where((mask_.cpu().numpy()==0) & (gt_mask_.cpu().numpy()!=0))[0]
                     # Observed channels
                    if len(obser_list) > 0:
                        y_true = y_system[:,obser_list].detach().cpu().numpy().flatten()
                        y_pred_flat = y_hat_system[:,obser_list].detach().cpu().numpy().flatten()
                        r2 = r2_score(y_true, y_pred_flat)
                        ev = explained_variance_score(y_true, y_pred_flat)
                    
                        obser_r2_list.append(r2)
                        obser_ev_list.append(ev)
                    if unobser_list.shape[0]==0:
                        continue
                    y_true_unobser = y_system[:,unobser_list].detach().cpu().numpy().flatten()
                    y_pred_flat_unobser = y_hat_system[:,unobser_list].detach().cpu().numpy().flatten()

                    r2_unobser = r2_score(y_true_unobser, y_pred_flat_unobser)
                    ev_unobser = explained_variance_score(y_true_unobser, y_pred_flat_unobser)
                    unobser_r2_list.append(r2_unobser)
                    unobser_ev_list.append(ev_unobser)
                # 计算平均训练损失
        avg_loss =loss_epoch / len(data_loader)
        avg_recon_loss = recon_loss_epoch / len(data_loader)
        avg_dyn_loss = dyn_loss_epoch / len(data_loader)
        avg_beh_loss = beh_loss_epoch / len(data_loader)
        # 计算平均值
    if len(r2_list) > 0 and len(ev_list) > 0:
        avg_r2 = np.mean(r2_list)
        avg_ev = np.mean(ev_list)
        std_r2 = np.std(r2_list)
        std_ev = np.std(ev_list)
        avg_beh_r2 = np.mean(beh_r2_list)
        avg_beh_ev = np.mean(beh_ev_list)
    else:
        avg_r2 = avg_ev = avg_beh_ev = avg_beh_r2 = 0.0

    if final_eval:
        avg_obser_r2 = np.mean(obser_r2_list) if len(obser_r2_list)>0 else 0.0
        avg_unobser_r2 = np.mean(unobser_r2_list) if len(unobser_r2_list)>0 else 0.0
        avg_obser_ev = np.mean(obser_ev_list) if len(obser_ev_list)>0 else 0.0
        avg_unobser_ev = np.mean(unobser_ev_list) if len(unobser_ev_list)>0 else 0.0

        std_obser_r2 = np.std(obser_r2_list) if len(obser_r2_list)>0 else 0.0
        std_unobser_r2 = np.std(unobser_r2_list) if len(unobser_r2_list)>0 else 0.0
        std_obser_ev = np.std(obser_ev_list) if len(obser_ev_list)>0 else 0.0
        std_unobser_ev = np.std(unobser_ev_list) if len(unobser_ev_list)>0 else 0.0

        print(f"Observed Channels - R²: {avg_obser_r2:.4f} ± {std_obser_r2:.4f}, EV: {avg_obser_ev:.4f} ± {std_obser_ev:.4f}")
        print(f"Unobserved Channels - R²: {avg_unobser_r2:.4f} ± {std_unobser_r2:.4f}, EV: {avg_unobser_ev:.4f} ± {std_unobser_ev:.4f}")
        print(f"Total Channels - R²: {avg_r2:.4f} ± {std_r2:.4f}, EV: {avg_ev:.4f} ± {std_ev:.4f}")
        return avg_r2, avg_ev, std_r2, std_ev, avg_beh_r2, avg_beh_ev, avg_loss
    
    
    model.train()  # 恢复训练模式
    return avg_r2, avg_ev, std_r2, std_ev, avg_beh_r2, avg_beh_ev, avg_loss

def calculate_masked_decoding_accuracy_hidden_real(model, ckpts, train_loader, test_loader, seed):
    rng = np.random.default_rng(seed)
    model.load_state_dict(torch.load(ckpts))
    model.eval()
    all_train_z = []
    all_train_z_pred = []
    all_train_behavior = []
    for i, (y,trial_id,condition_idx, behavior, mask, training_mask, gt_mask) in enumerate(train_loader):
        y_pred, y_recon, z_pred, z, y_behavior, loss, recon_loss, dyn_loss, beh_loss = model(y, mask, training_mask,trial_id, condition_idx,behavior=behavior)
        all_train_z.append(z.detach().cpu().numpy())
        all_train_z_pred.append(z_pred.detach().cpu().numpy())
        all_train_behavior.append(behavior.detach().cpu().numpy())
    all_test_z = []
    all_test_z_pred = []
    all_test_behavior = []
    for i, (y,trial_id,condition_idx, behavior, mask, training_mask, gt_mask) in enumerate(test_loader):
        y_pred, y_recon, z_pred, z, y_behavior, loss, recon_loss, dyn_loss, beh_loss = model(y, mask, training_mask,trial_id, condition_idx,behavior=behavior)
        all_test_z.append(z.detach().cpu().numpy())
        all_test_z_pred.append(z_pred.detach().cpu().numpy())
        all_test_behavior.append(behavior.detach().cpu().numpy())
    all_train_z = np.concatenate(all_train_z, axis=0) # batch * time * channel
    all_train_z_pred = np.concatenate(all_train_z_pred, axis=0)
    all_train_behavior = np.concatenate(all_train_behavior, axis=0)
    all_test_z = np.concatenate(all_test_z, axis=0)
    all_test_z_pred = np.concatenate(all_test_z_pred, axis=0)
    all_test_behavior = np.concatenate(all_test_behavior, axis=0)

    decoder = Ridge(alpha=1e-2)

    if model.beh_delay > 0:
        train_z = all_train_z[:, 1:-model.beh_delay, :]
        test_z = all_test_z[:, 1:-model.beh_delay, :]

        train_z_pred = all_train_z_pred[:, :-model.beh_delay, :]
        test_z_pred = all_test_z_pred[:, :-model.beh_delay, :]

        train_behavior = all_train_behavior[:, model.warmup + model.beh_delay:, :]
        test_behavior = all_test_behavior[:, model.warmup + model.beh_delay:, :]

    else:
        train_z = all_train_z[:, 1:, :]
        test_z = all_test_z[:, 1:, :]

        train_z_pred = all_train_z_pred
        test_z_pred = all_test_z_pred

        train_behavior = all_train_behavior[:, model.warmup:, :]
        test_behavior = all_test_behavior[:, model.warmup:, :]


    decoder.fit(train_z.reshape(-1, train_z_pred.shape[-1]), train_behavior.reshape(-1, 2))

    test_accuracy = decoder.score(test_z.reshape(-1, test_z.shape[-1]),test_behavior.reshape(-1, 2))
    train_pred_accuracy = decoder.score(train_z_pred.reshape(-1, train_z_pred.shape[-1]),train_behavior.reshape(-1, 2))
    test_pred_accuracy = decoder.score(test_z_pred.reshape(-1, test_z_pred.shape[-1]),test_behavior.reshape(-1, 2))

    return {
        "decode_test_r2": test_accuracy,
        "decode_train_pred_r2": train_pred_accuracy,
        "decode_test_pred_r2": test_pred_accuracy,
    }
def calculate_masked_decoding_accuracy_real_new(model, ckpts, train_loader, test_loader, seed):
    model.load_state_dict(torch.load(ckpts))
    model.eval()
    all_train_y = []
    all_train_y_pred = []
    all_train_behavior = []
    all_train_mask = []
    all_test_y = []
    all_test_y_pred = []
    all_test_behavior = []
    all_test_mask = []

    # y_pred[:, 0] is trained against y[:, model.warmup], so channel mixing is
    # performed below only after the two signals have been time-aligned.
    with torch.no_grad():
        for y, trial_id, condition_idx, behavior, mask, training_mask, gt_mask in train_loader:
            y_pred, y_recon, z_pred, z, y_behavior, loss, recon_loss, dyn_loss, beh_loss = model(
                y, mask, training_mask, trial_id, condition_idx, behavior=behavior
            )
            all_train_y.append(y.detach().cpu().numpy())
            all_train_y_pred.append(y_pred.detach().cpu().numpy())
            all_train_behavior.append(behavior.detach().cpu().numpy())
            all_train_mask.append(mask.detach().cpu().numpy())

        for y, trial_id, condition_idx, behavior, mask, training_mask, gt_mask in test_loader:
            y_pred, y_recon, z_pred, z, y_behavior, loss, recon_loss, dyn_loss, beh_loss = model(
                y, mask, training_mask, trial_id, condition_idx, behavior=behavior
            )
            all_test_y.append(y.detach().cpu().numpy())
            all_test_y_pred.append(y_pred.detach().cpu().numpy())
            all_test_behavior.append(behavior.detach().cpu().numpy())
            all_test_mask.append(mask.detach().cpu().numpy())

    all_train_y = np.concatenate(all_train_y, axis=0)  # trial * time * channel
    all_train_y_pred = np.concatenate(all_train_y_pred, axis=0)
    all_train_behavior = np.concatenate(all_train_behavior, axis=0)
    all_train_mask = np.concatenate(all_train_mask, axis=0)
    all_test_y = np.concatenate(all_test_y, axis=0)
    all_test_y_pred = np.concatenate(all_test_y_pred, axis=0)
    all_test_behavior = np.concatenate(all_test_behavior, axis=0)
    all_test_mask = np.concatenate(all_test_mask, axis=0)

    if model.beh_delay > 0:
        train_y = all_train_y[:, model.warmup:-model.beh_delay, :]
        test_y = all_test_y[:, model.warmup:-model.beh_delay, :]

        train_y_pred = all_train_y_pred[:, :-model.beh_delay, :]
        test_y_pred = all_test_y_pred[:, :-model.beh_delay, :]

        train_behavior = all_train_behavior[:, model.warmup + model.beh_delay:, :]
        test_behavior = all_test_behavior[:, model.warmup + model.beh_delay:, :]

    else:
        train_y = all_train_y[:, model.warmup:, :]
        test_y = all_test_y[:, model.warmup:, :]

        train_y_pred = all_train_y_pred
        test_y_pred = all_test_y_pred

        train_behavior = all_train_behavior[:, model.warmup:, :]
        test_behavior = all_test_behavior[:, model.warmup:, :]


    # mask=1: retain the observed ground truth; mask=0: fill with y_pred.
    train_mask = all_train_mask[:, None, :].astype(train_y.dtype, copy=False)
    test_mask = all_test_mask[:, None, :].astype(test_y.dtype, copy=False)
    train_hybrid_y = train_y * train_mask + train_y_pred * (1.0 - train_mask)
    test_hybrid_y = test_y * test_mask + test_y_pred * (1.0 - test_mask)
    train_mask_y = train_y * train_mask
    test_mask_y = test_y * test_mask

    decoder = Ridge(alpha=1e-2)
    decoder.fit(train_y.reshape(-1, train_y.shape[-1]), train_behavior.reshape(-1, 2))

    test_accuracy = decoder.score(test_y.reshape(-1, test_y.shape[-1]), test_behavior.reshape(-1, 2))
    train_hybrid_accuracy = decoder.score(train_hybrid_y.reshape(-1, train_hybrid_y.shape[-1]), train_behavior.reshape(-1, 2))
    test_hybrid_accuracy = decoder.score(test_hybrid_y.reshape(-1, test_hybrid_y.shape[-1]), test_behavior.reshape(-1, 2))
    train_pred_accuracy = decoder.score(train_y_pred.reshape(-1, train_y_pred.shape[-1]),train_behavior.reshape(-1, 2))
    test_pred_accuracy = decoder.score(test_y_pred.reshape(-1, test_y_pred.shape[-1]),test_behavior.reshape(-1, 2))
    train_mask_accuracy = decoder.score(train_mask_y.reshape(-1, train_mask_y.shape[-1]), train_behavior.reshape(-1, 2))
    test_mask_accuracy = decoder.score(test_mask_y.reshape(-1, test_mask_y.shape[-1]), test_behavior.reshape(-1, 2))

    return {
        "decode_test_r2": test_accuracy,
        "decode_train_pred_r2": train_pred_accuracy,
        "decode_test_pred_r2": test_pred_accuracy,
        "decode_train_hybrid_r2": train_hybrid_accuracy,
        "decode_test_hybrid_r2": test_hybrid_accuracy,
        "decode_train_mask_r2": train_mask_accuracy,
        "decode_test_mask_r2": test_mask_accuracy,
    }


def evaluate_model_final_real(model, data_loader):
    model.eval()

    r2_list, ev_list = [], []
    obser_r2_list, unobser_r2_list = [], []
    obser_ev_list, unobser_ev_list = [], []


    z_list, m_list, yp_list, yt_list = [], [], [], []

    with torch.no_grad():
        for i, (y, trial_id, condition_idx, behavior, mask, training_mask, gt_mask) in enumerate(data_loader):
            y_pred, y_recon, z_pred, z, y_behavior, loss, recon_loss, dyn_loss, beh_loss = model(y, mask, training_mask,trial_id, condition_idx,behavior=behavior)
            
            z_list.append(z_pred.detach().cpu())
            m_list.append(mask.detach().cpu())
            yp_list.append(y_pred.detach().cpu())
            yt_list.append(y[:, model.warmup:].detach().cpu())

            for j in range(y.shape[0]):
                y_system = y[j, model.warmup:, :]
                y_hat_system = y_pred[j, :, :]
                mask_ = mask[j]
                gt_mask_ = gt_mask[j]
                
                y_true = y_system.detach().cpu().numpy().flatten()
                y_pred_flat = y_hat_system.detach().cpu().numpy().flatten()

                r2 = r2_score(y_true, y_pred_flat)
                ev = explained_variance_score(y_true, y_pred_flat)

                r2_list.append(r2)
                ev_list.append(ev)

                obser_list = np.where(mask_.cpu().numpy() == 1)[0]
                unobser_list = np.where((mask_.cpu().numpy() == 0) & (gt_mask_.cpu().numpy()!=0))[0]

                if len(obser_list) > 0:
                    y_true_obs = y_system[:, obser_list].detach().cpu().numpy().flatten()
                    y_pred_obs = y_hat_system[:, obser_list].detach().cpu().numpy().flatten()

                    obser_r2_list.append(r2_score(y_true_obs, y_pred_obs))
                    obser_ev_list.append(explained_variance_score(y_true_obs, y_pred_obs))

                if len(unobser_list) > 0:
                    y_true_unobs = y_system[:, unobser_list].detach().cpu().numpy().flatten()
                    y_pred_unobs = y_hat_system[:, unobser_list].detach().cpu().numpy().flatten()

                    unobser_r2_list.append(r2_score(y_true_unobs, y_pred_unobs))
                    unobser_ev_list.append(explained_variance_score(y_true_unobs, y_pred_unobs))

    # dataloader 跑完之后再 concat
    z_preds = torch.cat(z_list, dim=0) if len(z_list) > 0 else None
    masks   = torch.cat(m_list, dim=0) if len(m_list) > 0 else None
    y_preds = torch.cat(yp_list, dim=0) if len(yp_list) > 0 else None
    y_gt    = torch.cat(yt_list, dim=0) if len(yt_list) > 0 else None

    metrics = {
        "avg_r2": np.mean(r2_list) if len(r2_list) > 0 else 0.0,
        "avg_ev": np.mean(ev_list) if len(ev_list) > 0 else 0.0,
        "avg_obser_r2": np.mean(obser_r2_list) if len(obser_r2_list) > 0 else 0.0,
        "avg_unobser_r2": np.mean(unobser_r2_list) if len(unobser_r2_list) > 0 else 0.0,
        "avg_obser_ev": np.mean(obser_ev_list) if len(obser_ev_list) > 0 else 0.0,
        "avg_unobser_ev": np.mean(unobser_ev_list) if len(unobser_ev_list) > 0 else 0.0,
    }

    model.train()

    return metrics, z_preds, masks, y_preds, y_gt
