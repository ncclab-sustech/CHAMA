"""Load standalone benchmark configurations without importing PyTorch."""
import argparse
import sys
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value, **variables):
    """Resolve data/output paths relative to the project, including templates."""
    path = Path(str(value).format(**variables))
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


@lru_cache(maxsize=None)
def _signature_options(relative_path, function):
    import ast
    tree = ast.parse((PROJECT_ROOT / relative_path).read_bytes())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    return {arg.arg for arg in node.args.args if arg.arg != 'self'}


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a YAML mapping.')
    sections = {'benchmark', 'data', 'masking', 'model', 'training', 'optimizer', 'runtime', 'evaluation', 'outputs', 'experiments'}
    if set(config) != sections:
        raise ValueError(f'Configuration sections mismatch: missing={sections - set(config)}, unknown={set(config) - sections}')
    for name in sections - {'benchmark', 'experiments'}:
        if not isinstance(config[name], dict):
            raise ValueError(f'{name} must be a mapping.')
    model_keys = _signature_options('models/FullModel.py', '__init__') - {'device'}
    training_file = 'real' if config['data']['kind'] in {'co', 'ibci', 'maze'} else 'synthetic'
    train_keys = _signature_options(f'training/train_test_model_{training_file}.py', 'train_test_model')
    train_keys -= {'model', 'optimizer', 'data', 'gt_dyn', 'trial_id', 'mask', 'training_mask', 'ckpts', 'behavior_data', 'condition_idx', 'gt_mask'}
    for name, allowed in [('model', model_keys), ('training', train_keys)]:
        unknown = set(config[name]) - allowed
        missing = allowed - set(config[name])
        if unknown or missing:
            raise ValueError(f'{name} parameters mismatch: missing={missing}, unknown={unknown}')
    experiments = config['experiments']
    if not isinstance(experiments, list) or not experiments:
        raise ValueError('experiments must be a nonempty list.')
    names = []
    for experiment in experiments:
        if set(experiment) != {'name', 'model', 'training'}:
            raise ValueError('Each experiment needs name, model, and training.')
        names.append(experiment['name'])
        for name, allowed in [('model', model_keys), ('training', train_keys)]:
            if not isinstance(experiment[name], dict) or set(experiment[name]) - allowed:
                raise ValueError(f'Invalid experiment {name} overrides.')
    if len(names) != len(set(names)):
        raise ValueError('Experiment names must be unique.')
    if config['data']['kind'] not in {'lorenz_partial', 'lorenz_ood', 'lorenz_redundant', 'co', 'ibci', 'maze'}:
        raise ValueError('Unknown data kind.')
    if config['optimizer']['name'] != 'AdamW':
        raise ValueError('Only AdamW is supported.')
    run = config['runtime']
    if not run['devices'] or not run['seeds'] or run['max_workers'] < 1:
        raise ValueError('devices, seeds, and max_workers must be nonempty/positive.')
    for ratio in run['mask_ratios']:
        if not 0 <= ratio <= 1:
            raise ValueError('mask_ratios must be between 0 and 1.')
    for experiment in experiments:
        train = {**config['training'], **experiment['training']}
        model = {**config['model'], **experiment['model']}
        if not isinstance(model['use_pos_emb'], bool):
            raise ValueError('use_pos_emb must be true or false.')
        if train['epochs'] < 1 or train['batch_size'] < 1 or train['eval_interval'] < 1 or train['num_workers'] < 0:
            raise ValueError('Invalid epoch, batch, evaluation interval, or worker count.')
        if not (0 < train['test_size'] < 1 and 0 <= train['val_size'] < 1 - train['test_size']):
            raise ValueError('Invalid train/validation/test split.')
        if model['warmup_steps'] < 1 or model['attn_dim'] % 2:
            raise ValueError('warmup_steps must be positive and attn_dim must be even.')


def load_config(path):
    with Path(path).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def load_config_cli(default_filename=None, transform=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Run a benchmark from a standalone YAML configuration.')
    default = PROJECT_ROOT / 'configs' / default_filename if default_filename else None
    parser.add_argument('--config', type=Path, default=default, required=default is None, help='Benchmark YAML file')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print configuration without loading data or starting training')
    args = parser.parse_args()
    config = load_config(args.config)
    if transform is not None:
        config = transform(config)
        validate_config(config)
    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        parser.exit()
    return config
