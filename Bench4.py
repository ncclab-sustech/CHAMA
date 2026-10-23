
from utils.config import load_config_cli, project_path

# Parse the CLI before importing training dependencies so --dry-run stays lightweight.
if __name__ == '__main__':
    config = load_config_cli('bench4.yaml')

import datetime
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import torch
from models.FullModel import FullModel
from training.train_test_model_real import train_test_model
from utils.metrics_calculation import evaluate_model_final_real, calculate_masked_decoding_accuracy_real_new, calculate_masked_decoding_accuracy_hidden_real
from utils.pattern_mask_gen import (
    load_maze_data,
    prepare_beh_and_condition_maze,
    prepare_mask_and_patterns,
    set_all_seeds,
    prepare_beh_and_condition,
)

def load_data(config):
    settings = config['data']
    kwargs = dict(settings['loader'])
    kwargs['data_path'] = project_path(kwargs['data_path'])
    neural, behavior, condition_idx, trial_id, channel_num = load_maze_data(**kwargs)
    if settings['shuffle_trials']:
        generator = None
        if settings['shuffle_seed'] is not None:
            generator = torch.Generator().manual_seed(settings['shuffle_seed'])
        idx = torch.randperm(behavior.shape[0], generator=generator)
        behavior, neural, condition_idx, trial_id = behavior[idx], neural[idx], condition_idx[idx], trial_id[idx]
    if settings['normalization'] == 'zscore':
        behavior, condition = prepare_beh_and_condition_maze(behavior, condition_idx)
    elif settings['normalization'] == 'max_abs':
        behavior, condition = prepare_beh_and_condition(behavior, condition_idx)
    else:
        raise ValueError('Unsupported behavior normalization: ' + str(settings['normalization']))
    return dict(data=neural, behavior_data=behavior, trial_id=trial_id, condition_idx=condition, channel_num=channel_num)

def evaluate(config, model, train_loader, test_loader, checkpoint, seed):
    row = {}
    for setting in config['evaluation']['teacher_forcing']:
        prefix = setting['name']
        if setting['tf'] is not None:
            model.tf = setting['tf']
        metrics, *_ = evaluate_model_final_real(model, test_loader)
        row.update({f'{prefix}_{key}': value for key, value in metrics.items()})
        if config['evaluation']['decode_observations']:
            metrics = calculate_masked_decoding_accuracy_real_new(model=model, ckpts=checkpoint, train_loader=train_loader, test_loader=test_loader, seed=seed)
            row.update({f'{prefix}_{key}_decoding': value for key, value in metrics.items()})
        if config['evaluation']['decode_latents']:
            metrics = calculate_masked_decoding_accuracy_hidden_real(model=model, ckpts=checkpoint, train_loader=train_loader, test_loader=test_loader, seed=seed)
            row.update({f'{prefix}_{key}_decoding_hidden': value for key, value in metrics.items()})
    return row

def run_one_task(task):
    task_id, config, experiment, seed, mask_ratio, device, bundle = task
    runtime = config['runtime']
    name = experiment['name']
    if task_id < runtime['stagger_workers']:
        time.sleep(task_id * runtime['worker_start_delay_seconds'])
    if device.startswith('cuda'):
        torch.cuda.set_device(int(device.split(':', 1)[1]))
    model_kwargs = {**config['model'], **experiment['model'], 'device': device}
    if model_kwargs['Cmax'] == 'auto':
        model_kwargs['Cmax'] = bundle['channel_num']
    train_kwargs = {**config['training'], **experiment['training']}
    if train_kwargs['random_state'] == 'seed':
        train_kwargs['random_state'] = seed
    attempts = runtime['max_task_retries'] + 1
    for attempt in range(attempts):
        try:
            print(f'start: {name}, seed={seed}, mask_ratio={mask_ratio}, device={device}, attempt={attempt + 1}/{attempts}')
            set_all_seeds(seed)
            data_kwargs = {key: value for key, value in bundle.items() if key != 'channel_num'}
            channel_mask, training_mask, gt_mask, _ = prepare_mask_and_patterns(bundle['data'], bundle['channel_num'], mask_ratio=mask_ratio, threshold=config['masking']['threshold'], seed=seed)
            data_kwargs.update(mask=channel_mask, training_mask=training_mask, gt_mask=gt_mask)
            model = FullModel(**model_kwargs).to(device)
            optimizer_kwargs = dict(config['optimizer']['kwargs'])
            optimizer_kwargs['betas'] = tuple(optimizer_kwargs['betas'])
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            today = datetime.date.today().strftime('%Y%m%d')
            checkpoint = project_path(config['outputs']['checkpoint'], exp_name=name, seed=seed, mask_ratio=mask_ratio, today=today)
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            result = train_test_model(model=model, optimizer=optimizer, ckpts=checkpoint, **data_kwargs, **train_kwargs)
            _, _, _, train_loader, test_loader, _ = result
            row = {'experiment': name, 'mode': 'attention', 'warmup_steps': model_kwargs['warmup_steps'], 'seed': seed, 'mask_ratio': mask_ratio}
            row.update(evaluate(config, model, train_loader, test_loader, checkpoint, seed))
            print(f'done: {name}, seed={seed}, mask_ratio={mask_ratio}')
            return row
        except Exception:
            print(traceback.format_exc())
            if attempt + 1 < attempts:
                delay = runtime['retry_backoff_base'] ** attempt
                print(f'retrying in {delay}s')
                time.sleep(delay)
    print(f'task failed: {name}, seed={seed}, mask_ratio={mask_ratio}')
    return None

def main(config):
    if config['data']['kind'] != 'maze':
        raise ValueError('This entry point requires data.kind: maze')
    checkpoint = config['outputs']['checkpoint'].replace('{today}', datetime.date.today().strftime('%Y%m%d'))
    config = {**config, 'outputs': {**config['outputs'], 'checkpoint': checkpoint}}
    runtime = config['runtime']
    output_csv = project_path(config['outputs']['metrics_csv'])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    done = set()
    if runtime['skip_completed'] and os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        columns = ['experiment', 'seed', 'mask_ratio']
        done = set(zip(*(existing[column] for column in columns)))
    bundle = load_data(config)
    tasks = []
    ratios = runtime['mask_ratios']
    for seed in runtime['seeds']:
        for experiment in config['experiments']:
            for ratio in ratios:
                key = (experiment['name'], seed, ratio)
                if key in done:
                    print(f'skip: {key}')
                    continue
                device = runtime['devices'][len(tasks) % len(runtime['devices'])]
                tasks.append((len(tasks), config, experiment, seed, ratio, device, bundle))
    if not tasks:
        print('all tasks are already complete')
        return

    def save_result(row):
        if row is not None:
            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
    if not runtime['parallel']:
        for task in tasks:
            save_result(run_one_task(task))
        return
    context = mp.get_context(runtime['start_method'])
    with ProcessPoolExecutor(max_workers=runtime['max_workers'], mp_context=context) as executor:
        futures = [executor.submit(run_one_task, task) for task in tasks]
        for future in as_completed(futures):
            try:
                save_result(future.result())
            except Exception:
                print('worker process failed before returning a result')
                print(traceback.format_exc())

if __name__ == '__main__':
    main(config)
