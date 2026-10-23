import numpy as np
import torch
import os
import random
from pathlib import Path
from scipy.io import loadmat,savemat
import torch.nn.functional as F
def set_channel_zero(start_idx, end_idx, channel_idx,channel_mask,training_mask):
    """将指定范围内的trial的指定通道掩码设置为0"""
    channel_mask[start_idx:end_idx, channel_idx] = 0
    training_mask[start_idx:end_idx, channel_idx] = 0
    return channel_mask,training_mask
def generate_ood_mask(num_trials,num_channels,train_ratio,val_ratio,test_ratio,full_obs_ratio=0.1):
    num_train = int(num_trials * train_ratio)
    num_val = int(num_trials * val_ratio)
    num_test = num_trials - num_train - num_val

    # 创建掩码张量
    channel_mask = torch.ones((num_trials, num_channels))
    training_mask = torch.ones((num_trials, num_channels))

    #设置训练集掩码
    full_n = int(num_train * full_obs_ratio)

    # 先把 [0, full_n) 留作全1（不做任何mask）
    start = full_n
    mid = (num_train + start)//2

    #全观测剩下的部分再分成两部分：缺z和缺x
    channel_mask, training_mask = set_channel_zero(start, mid, 2,channel_mask,training_mask)
    channel_mask, training_mask = set_channel_zero(mid, num_train, 0,channel_mask,training_mask)

    # 设置验证集掩码 (20%)--与训练集一致

    full_n_val = int(num_val * full_obs_ratio)
    val_start, val_mid, val_end = num_train + full_n_val,num_train + (num_val + full_n_val)//2, num_train + num_val
    channel_mask, training_mask = set_channel_zero(val_start, val_mid, 2,channel_mask,training_mask)
    channel_mask, training_mask = set_channel_zero(val_mid, val_end, 0,channel_mask,training_mask)

    # 设置测试集掩码 (20%)--始终mask掉y通道
    test_start, test_end = val_end, val_end + num_test
    channel_mask, training_mask = set_channel_zero(test_start, test_end, 1,channel_mask,training_mask)
    return channel_mask,training_mask

def generate_partial_mask(num_trials,num_channels,train_ratio,val_ratio,test_ratio,full_obs_ratio=0):
    num_train = int(num_trials * train_ratio)
    num_val = int(num_trials * val_ratio)
    num_test = num_trials - num_train - num_val
    channel_mask = torch.ones((num_trials, num_channels))
    training_mask = torch.ones((num_trials, num_channels))
    full_n = int(num_train * full_obs_ratio)
    # 先把 [0, full_n) 留作全1（不做任何mask）
    start = full_n
    remain = num_train - start

    num_mask_groups = 3  # 你原来是3个quarter分别mask不同通道
    group_size = remain // num_mask_groups

    for g in range(num_mask_groups):
        s = start + g * group_size
        e = start + (g + 1) * group_size if g < num_mask_groups - 1 else num_train
        channel_mask, training_mask = set_channel_zero(s, e, g,channel_mask,training_mask)   # g=0/1/2 -> mask 不同通道


    # 设置验证集掩码 (20%)
    val_start, val_end = num_train, num_train + num_val
    val_segment = num_val // 3
    for i in range(3):
        start_idx = val_start + i * val_segment
        end_idx = start_idx + val_segment if i < 2 else val_end
        channel_mask, training_mask = set_channel_zero(start_idx, end_idx, i,channel_mask,training_mask)

    # 设置测试集掩码 (20%)
    test_start, test_end = val_end, val_end + num_test
    test_segment = num_test // 3
    for i in range(3):
        start_idx = test_start + i * test_segment
        end_idx = start_idx + test_segment if i < 2 else test_end
        channel_mask, training_mask = set_channel_zero(start_idx, end_idx, i,channel_mask,training_mask)

    return channel_mask,training_mask

def session_channel_mask(trial_id,n_channels,ratio_range=(0.0, 0.30),per_session_fixed_ratio=None,seed=42,session_ids=None):
    rng = np.random.default_rng(seed)

    trial_id = np.asarray(trial_id)
    if trial_id.ndim != 1:
        raise ValueError("trial_id 必须是一维数组，shape = (n_trials_total,)")

    # session 集合
    if session_ids is None:
        session_ids = np.unique(trial_id)
    else:
        session_ids = np.asarray(session_ids)

    # ratio 设置
    if per_session_fixed_ratio is not None:
        if not (0.0 <= per_session_fixed_ratio <= 1.0):
            raise ValueError("per_session_fixed_ratio 必须在 [0, 1] 内。")
        min_r = max_r = float(per_session_fixed_ratio)
    else:
        if (not isinstance(ratio_range, (tuple, list))) or len(ratio_range) != 2:
            raise ValueError("ratio_range 必须是 (min_ratio, max_ratio)。")
        min_r, max_r = map(float, ratio_range)

    # 为每个 session 生成一个固定 mask（shape: (n_channels,)）
    mask_by_session = {}
    for sid in session_ids:
        r = rng.uniform(min_r, max_r) if min_r != max_r else min_r
        k = int(np.round(r * n_channels))
        k = int(np.clip(k, 0, n_channels))

        m = np.ones(n_channels, dtype=bool)
        if k > 0:
            idx = rng.choice(n_channels, size=k, replace=False)
            m[idx] = False

        mask_by_session[int(sid)] = m.astype(np.int8)  # 1=保留, 0=mask

    # 映射到每个 trial
    mask_trials = np.empty((trial_id.shape[0], n_channels), dtype=np.int8)
    for i, sid in enumerate(trial_id):
        mask_trials[i] = mask_by_session[int(sid)]

    return mask_trials

def random_channel_mask_real(n_trials = 500, n_channels = 50, ratio_range = (0.0, 0.30), per_trial_fixed_ratio = None,seed = 42,mask=None):

    rng = np.random.default_rng(seed)

    if per_trial_fixed_ratio is not None:
        if not (0.0 <= per_trial_fixed_ratio <= 1.0):
            raise ValueError("per_trial_fixed_ratio 必须在 [0, 1] 内。")
        min_r = max_r = float(per_trial_fixed_ratio)
    else:
        if (not isinstance(ratio_range, (tuple, list)) or len(ratio_range) != 2):
            raise ValueError("ratio_range 必须是 (min_ratio, max_ratio)。")
        min_r, max_r = map(float, ratio_range)
    if mask is None:
        mask = np.ones((n_trials, n_channels), dtype=bool)

    mask_before = mask.copy()
    for t in range(n_trials):
        r = rng.uniform(min_r, max_r) if min_r != max_r else min_r
        k = int(np.round(r * n_channels))
        k = np.clip(k, 0, n_channels)

        if k > 0:
            available = np.flatnonzero(mask[t])  # 当前为 True 的通道索引
            k_eff = min(k, available.size)       # 防止可用通道不够导致报错
            if k_eff > 0:
                idx = rng.choice(available, size=k_eff, replace=False)
                mask[t, idx] = False


    return mask.astype(np.int8), mask_before.astype(np.int8)  # 1=保留, 0=mask



def summarize_patterns(channel_mask: torch.Tensor):
    """
    channel_mask: (B, C), values in {0,1} or bool
    return: python list of lists, first row is all-ones
    """
    cm = channel_mask.detach().to("cpu")
    if cm.dtype != torch.bool:
        cm = cm.bool()

    B, C = cm.shape
    all_ones = tuple([1] * C)

    seen = set()
    patterns = [list(all_ones)]  # 第一行全1

    for i in range(B):
        key = tuple(cm[i].int().tolist())
        if key == all_ones:
            continue
        if key not in seen:
            seen.add(key)
            patterns.append(list(key))
    patterns = torch.tensor(patterns)
    return patterns

def set_all_seeds(seed=42, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 让 CUDA 尽量走确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        
        # 有些 CUDA 环境还需要这个
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def generate_valid_response_mask(tensor, threshold=0.0):
    """
    为[B, T, C]张量生成响应通道的mask
    
    参数:
    tensor: 形状为[B, T, C]的张量
    threshold: 响应阈值，默认为0.0（大于阈值表示有响应）
    
    返回:
    mask: 形状为[B, C]的布尔张量，有效通道为1，无效通道为0
    """
    mean_values = tensor.mean(dim=1)  # 形状[B, C]，沿时间维度求平均
    mask = (mean_values > threshold).float()
    
    return np.array(mask)

def load_data_as_function(folder, pattern="*trial_data_with_different_time_bins*.mat"):
    folder = Path(folder)
    mat_paths = sorted(folder.glob(pattern))

    minT_global = None
    cond_idx_list = []
    day_id_list = []

    for day_i, p in enumerate(mat_paths):
        d = loadmat(p, variable_names=["stop_time_20ms", "target_index"])
        stop = np.asarray(d["stop_time_20ms"]).squeeze()
        minT_day = int(np.min(stop))
        minT_global = minT_day if (minT_global is None) else min(minT_global, minT_day)

        cond_idx = torch.as_tensor(np.asarray(d["target_index"]).squeeze(), dtype=torch.long)
        cond_idx_list.append(cond_idx)
        day_id_list.append(torch.full((cond_idx.shape[0],), day_i, dtype=torch.long))

    print(f"Found {len(mat_paths)} days, global min T = {minT_global}")

    x_list = []
    y_list = []
    channel_num = None

    for p in mat_paths:
        d = loadmat(p, variable_names=["smooth_spike_data_20ms", "curs_v_data_20ms"])
        neuro = d["smooth_spike_data_20ms"]
        beh = d["curs_v_data_20ms"]

        neuro_cut = neuro[:, :, :minT_global]
        beh_cut = beh[:, :, :minT_global]

        x = torch.as_tensor(neuro_cut, dtype=torch.float32).permute(0, 2, 1)
        y = torch.as_tensor(beh_cut, dtype=torch.float32).permute(0, 2, 1)

        if channel_num is None:
            channel_num = x.shape[2]
        else:
            assert x.shape[2] == channel_num, f"通道数不一致：期望 {channel_num}，但 {p.name} 是 {x.shape[2]}"

        x_list.append(x)
        y_list.append(y)

    cut_neuro_data = torch.cat(x_list, dim=0) # (B_total, T_global, C)
    cut_beh_data = torch.cat(y_list, dim=0)
    condition_idx = torch.cat(cond_idx_list, dim=0) # (B_total,)
    trial_id = torch.cat(day_id_list, dim=0) # (B_total,) 这里表示第几天

    return cut_neuro_data, cut_beh_data, condition_idx, trial_id, minT_global, channel_num

def remove_speed_outlier(cut_beh_data, cut_neuro_data, condition_idx, trial_id, mat_path='data/speed_outlier.mat'):
    speed_outlier = loadmat(mat_path)['speed_outlier'][0].tolist()
    desired_list = [i for i in range(cut_beh_data.shape[0]) if i not in speed_outlier]

    print('speed_outlier.shape:', len(speed_outlier))

    cut_beh_data = cut_beh_data[desired_list]
    cut_neuro_data = cut_neuro_data[desired_list]
    condition_idx = condition_idx[desired_list]
    trial_id = trial_id[desired_list]

    return cut_beh_data, cut_neuro_data, condition_idx, trial_id

def prepare_beh_and_condition(cut_beh_data, condition_idx):
    """
    处理行为数据，并生成 one-hot condition
    """
    cut_beh_data = cut_beh_data - torch.mean(cut_beh_data.reshape((-1, 2)), dim=0)
    cut_beh_data = cut_beh_data / torch.max(torch.abs(cut_beh_data))

    num_conditions = int(condition_idx.max().item() + 1)
    condition = F.one_hot(condition_idx, num_classes=num_conditions)

    return cut_beh_data, condition

def prepare_beh_and_condition_maze(cut_beh_data, condition_idx):
    """
    处理行为数据，并生成 one-hot condition
    """
    cut_beh_data = cut_beh_data - torch.mean(cut_beh_data.reshape(-1, 2), dim=0)
    std = torch.std(cut_beh_data.reshape(-1, 2), dim=0, correction=0)
    cut_beh_data = cut_beh_data / (std + 1e-8)

    num_conditions = int(condition_idx.max().item() + 1)
    condition = F.one_hot(condition_idx, num_classes=num_conditions)

    return cut_beh_data, condition


def prepare_mask_and_patterns(cut_neuro_data, channel_num, mask_ratio=0.1, threshold=1e-3, seed=42):
    """
    生成 gt_mask、channel_mask、training_mask 和 patterns
    """
    batch_size = cut_neuro_data.shape[0]

    gt_mask_np = generate_valid_response_mask(cut_neuro_data, threshold=threshold)
    channel_mask, gt_mask = random_channel_mask_real(batch_size,channel_num,per_trial_fixed_ratio=mask_ratio,seed=seed,mask=gt_mask_np)
    training_mask = channel_mask.copy()

    channel_mask = torch.as_tensor(channel_mask)
    training_mask = torch.as_tensor(training_mask)
    gt_mask = torch.as_tensor(gt_mask)
    patterns = summarize_patterns(training_mask)

    return channel_mask, training_mask, gt_mask, patterns


def load_ibci_trials_simple(load_path: str, behavior_key: str = "cursor_pos"):
    data = torch.load(load_path, weights_only=False)

    spikes_list = [torch.as_tensor(x, dtype=torch.float32) for x in data["spikes"]]
    behavior_list = [torch.as_tensor(x, dtype=torch.float32) for x in data[behavior_key]]
    time_len_list = data["time_len"]
    condition_idx = torch.tensor(data["cond_id"], dtype=torch.long) - 1

    global_min_len = min(time_len_list)
    cut_neuro_data = torch.stack([x[:global_min_len] for x in spikes_list], dim=0)
    cut_beh_data = torch.stack([x[:global_min_len] for x in behavior_list], dim=0)

    trial_id = torch.zeros(cut_neuro_data.shape[0], dtype=torch.long)
    channel_num = cut_neuro_data.shape[2]

    return cut_neuro_data, cut_beh_data, trial_id, channel_num, condition_idx


from pathlib import Path

import numpy as np
import torch


def load_maze_data(data_path = "data/maze_data.npz", cut_start = 30, behavior_dims = 2,device = None,):

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{data_path}")

    with np.load(data_path, allow_pickle=True) as maze_data:
        cut_neuro_data = torch.as_tensor(maze_data["neural"][:, cut_start:, :], dtype=torch.float32, device=device,)
        cut_beh_data = torch.as_tensor(maze_data["behavior"][:, cut_start:, -behavior_dims:], dtype=torch.float32, device=device,)
        condition_idx = torch.as_tensor(maze_data["condition_id"], dtype=torch.long, device=device,)

    num_samples = cut_neuro_data.shape[0]

    trial_id = torch.zeros(num_samples, dtype=torch.long, device=device,)
    channel_num = cut_neuro_data.shape[2]

    return cut_neuro_data, cut_beh_data, condition_idx, trial_id, channel_num