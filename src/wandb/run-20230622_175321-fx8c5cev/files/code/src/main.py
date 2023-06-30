import os
import json
import copy
import argparse
import traceback
import wandb
import torch
from torch import multiprocessing

import config
from config import printl as print
import resources
from train.classifier import ClassifierTrainer, generate_figs

def test():
    from models import test as models_test
    models_test()
    
def training_run(settings, device='cpu', generate_figs=False):
    if hasattr(device, '__getitem__'):
        device = device[0]
    save_dir = os.path.join(config.RESULTS_BASE_DIR, settings['save_dir'])
    os.makedirs(save_dir, exist_ok=True)
    if len(os.listdir(save_dir)) > 0:
        save_dir = os.path.join(save_dir, 'trial_%d'%(max(int(f.split('_')[-1]) for f in os.listdir(save_dir))+1))
    else:
        save_dir = os.path.join(save_dir, 'trial_0')
    results_save_dir = os.path.join(save_dir, 'results')
    model_save_dir = os.path.join(save_dir, 'models')
    figures_save_dir = os.path.join(save_dir, 'figures')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(figures_save_dir, exist_ok=True)
    config.specify_log_file(os.path.join(save_dir, 'log.txt'))
    with open(os.path.join(save_dir, 'settings.json'), 'w') as F:
        json.dump(settings, F, indent=2)
    trainer = ClassifierTrainer(using_wandb=False, **settings)
    trainer.train_model(settings['total_epochs'], results_save_dir=results_save_dir, model_save_dir=model_save_dir)
    if generate_figs:
        generate_figs(save_dir)

def spawn_agent_(sweep_id, project, device, classifier_settings):
    wandb.agent(sweep_id, project=project, function=lambda: run_wandb_trial_(device, classifier_settings))

def run_wandb_trial_(device, classifier_settings):
    try:
        classifier_settings = copy.deepcopy(classifier_settings)
        wandb.init()
        wandb_config = dict(wandb.config)
        wandb_config = config.nest_dict(wandb_config)
        for wc_key, wc_val in wandb_config.items():
            if (wc_key in classifier_settings.items()) and isinstance(wc_val, dict):
                classifier_settings[wc_key].update(wc_val)
            else:
                classifier_settings[wc_key] = wc_val
        save_dir = config.results_subdir(settings['save_dir'])
        if len(os.listdir(save_dir)) > 0:
            save_dir = os.path.join(save_dir, 'trial_%d'%(max(int(f.split('_')[-1]) for f in os.listdir(save_dir))+1))
        else:
            save_dir = os.path.join(save_dir, 'trial_0')
        results_save_dir = os.path.join(save_dir, 'results')
        model_save_dir = os.path.join(save_dir, 'models')
        figures_save_dir = os.path.join(save_dir, 'figures')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(results_save_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(figures_save_dir, exist_ok=True)
        config.specify_log_file(os.path.join(save_dir, 'log.txt'))
        with open(os.path.join(save_dir, 'settings.json'), 'w') as F:
            json.dump(settings, F, indent=2)
        trainer = ClassifierTrainer(using_wandb=True, **classifier_settings)
        trainer.train_model(settings['total_epochs'], results_save_dir=results_save_dir, model_save_dir=model_save_dir)
    except Exception:
        traceback.print_exc()
        with open(os.path.join(save_dir, 'log.txt'), 'a') as F:
            traceback.print_exc(file=F)
        raise Exception('Trial code crashed.')

def htune_run(settings, trials_per_gpu=1, devices='cpu'):
    if not hasattr(devices, '__len__'):
        devices = [devices]
    wandb_config = settings['wandb_config']
    wandb_config['parameters'] = config.denest_dict(wandb_config['parameters'])
    classifier_settings = {key: val for key, val in settings.items() if key != 'wandb_config'}
    if 'sweep_id' in settings:
        sweep_id = settings['sweep_id']
    else:
        sweep_id = wandb.sweep(sweep=wandb_config, project=settings['wandb_project'])
    config.set_num_agents(trials_per_gpu*len(devices))
    if trials_per_gpu*len(devices) == 1:
        spawn_agent(sweep_id, devices[0], classifier_settings)
    else:
        procs = []
        for dev_idx, dev in enumerate(devices):
            for trial_idx in range(trials_per_gpu):
                p = multiprocessing.Process(
                    target=spawn_agent_,
                    args=(sweep_id, settings['wandb_project'], dev, classifier_settings)
                )
                procs.append(p)
                p.start()
        for p in procs:
            p.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test', default=False, action='store_true', help='Run tests to validate dataset and model code.'
    )
    parser.add_argument(
        '--download-resources', default=False, action='store_true', help='Download all available datasets and pretrained models.'
    )
    parser.add_argument(
        '--train', choices=config.get_available_configs(train=True), nargs='+', default=[],
        help='Training runs to execute, as defined in the respective config files.'
    )
    parser.add_argument(
        '--htune', choices=config.get_available_configs(train=False), nargs='+', default=[],
        help='Hyperparameter tuning runs to execute, as defined in the respective config files.'
    )
    parser.add_argument(
        '--trials-per-gpu', default=1, type=int, help='Maximum number of trials that should be run simultaneously per GPU.'
    )
    parser.add_argument(
        '--save-dir', default=None, type=str, help='Directory in which to save results. Overrides the value in the config file.'
    )
    parser.add_argument(
        '--sweep-id', default=None, type=str, help='WandB sweep ID to use if continuing an existing sweep. Useful for resuming crashed sweeps or using multiple machines to run a sweep.'
    )
    parser.add_argument(
        '--devices', default=None, nargs='+', choices=['cpu', *['cuda:%d'%(dev_idx) for dev_idx in range(torch.cuda.device_count())]],
        help='Devices to use for this trial.'
    )
    parser.add_argument(
        '--cudnn-benchmark', default=False, action='store_true', help='Enables cudNN autotuner to search for an efficient algorithm for convolutions.'
    )
    parser.add_argument(
        '--generate-figs', default=False, action='store_true', help='Save figures to the trial/figures directory after each trial is completed.'
    )
    args = parser.parse_args()
    
    if args.test:
        print('Running tests ...')
        test()
    if args.download_resources:
        print('Downloading resources ...')
        resources.download_all()
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if args.devices is None:
        args.devices = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.sweep_id is not None:
        settings['sweep_id'] = args.sweep_id
    for config_name in args.train:
        print('Executing training run defined in {} ...'.format(os.path.join(config.TRAIN_CONFIGS, config_name)))
        settings = config.load_config(config_name, train=True)
        if args.save_dir is not None:
            settings['save_dir'] = args.save_dir
        print('Settings:')
        print('\n'.join(['\t{}: {}'.format(key, val) for key, val in settings.items()]))
        training_run(settings, device=args.devices, generate_figs=args.generate_figs)
    for config_name in args.htune:
        multiprocessing.set_start_method('spawn')
        print('Executing hyperparameter tuning run defined in {} ...'.format(os.path.join(config.HTUNE_CONFIGS, config_name)))
        settings = config.load_config(config_name, train=False)
        if args.save_dir is not None:
            settings['save_dir'] = args.save_dir
        print('Settings:')
        print('\n'.join(['\t{}: {}'.format(key, val) for key, val in settings.items()]))
        htune_run(settings, trials_per_gpu=args.trials_per_gpu, devices=args.devices)

if __name__ == '__main__':
    main()