"""Independent bench2 entry point: data preparation, training, and evaluation."""
from utils.config import load_config_cli, project_path

# Parse the CLI before importing training dependencies so --dry-run stays lightweight.
if __name__ == '__main__':
    config = load_config_cli('bench2.yaml')

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
from training.train_test_model_synthetic import train_test_model
from utils.metrics_calculation import set_seed, evaluate_model_final, evaluate_zpred_gt_dyn_linear_all
from utils.pattern_mask_gen import session_channel_mask

def load_data(config):
    settings = config['data']
    kwargs = dict(settings['loader'])
    path = project_path(kwargs.pop('path'))
    loaded = torch.load(path, **kwargs)
    y = torch.as_tensor(loaded[settings['neural_key']])
    gt_dyn = loaded[settings['latent_key']]
    trial_id = loaded[settings['trial_id_key']]
    channel_mask = torch.as_tensor(session_channel_mask(trial_id, y.shape[2], **config['masking']['kwargs']))
    training_mask = channel_mask.clone()
    return dict(data=y, gt_dyn=gt_dyn, trial_id=trial_id, mask=channel_mask, training_mask=training_mask, channel_num=y.shape[2])

def evaluate(config, model, train_loader, test_loader, checkpoint, seed):
    row = {}
    for setting in config['evaluation']['teacher_forcing']:
        prefix = setting['name']
        if setting['tf'] is not None:
            model.tf = setting['tf']
        metrics, *_ = evaluate_model_final(model, test_loader)
        row.update({f'{prefix}_{key}': value for key, value in metrics.items()})
        if config['evaluation']['latent_regression']:
            metrics = evaluate_zpred_gt_dyn_linear_all(model, test_loader)
            row.update({f'{prefix}_{key}': value for key, value in metrics.items()})
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
            set_seed(seed)
            data_kwargs = {key: value for key, value in bundle.items() if key != 'channel_num'}
            model = FullModel(**model_kwargs).to(device)
            optimizer_kwargs = dict(config['optimizer']['kwargs'])
            optimizer_kwargs['betas'] = tuple(optimizer_kwargs['betas'])
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            today = datetime.date.today().strftime('%Y%m%d')
            checkpoint = project_path(config['outputs']['checkpoint'], exp_name=name, seed=seed, mask_ratio=mask_ratio, today=today)
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            result = train_test_model(model=model, optimizer=optimizer, ckpts=checkpoint, **data_kwargs, **train_kwargs)
            _, _, (train_loader, test_loader, _) = result
            row = {'experiment': name, 'seed': seed}
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
    if config['data']['kind'] != 'lorenz_redundant':
        raise ValueError('This entry point requires data.kind: lorenz_redundant')
    checkpoint = config['outputs']['checkpoint'].replace('{today}', datetime.date.today().strftime('%Y%m%d'))
    config = {**config, 'outputs': {**config['outputs'], 'checkpoint': checkpoint}}
    runtime = config['runtime']
    output_csv = project_path(config['outputs']['metrics_csv'])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    done = set()
    if runtime['skip_completed'] and os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        columns = ['experiment', 'seed']
        done = set(zip(*(existing[column] for column in columns)))
    bundle = load_data(config)
    tasks = []
    ratios = [None]
    for seed in runtime['seeds']:
        for experiment in config['experiments']:
            for ratio in ratios:
                key = (experiment['name'], seed)
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
