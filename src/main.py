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
from train.classifier import ClassifierTrainer, generate_training_figs, generate_eval_figs
from train.weird_gan import GANTrainer

def test():
    from models import test as models_test
    models_test()
    
def get_save_dir(name):
    save_dir = os.path.join(config.RESULTS_BASE_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    if len(os.listdir(save_dir)) > 0:
        save_dir = os.path.join(save_dir, 'trial_%d'%(max(int(f.split('_')[-1]) for f in os.listdir(save_dir))+1))
    else:
        save_dir = os.path.join(save_dir, 'trial_0')
    results_save_dir = os.path.join(save_dir, 'results')
    model_save_dir = os.path.join(save_dir, 'models')
    os.makedirs(results_save_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    return save_dir, results_save_dir, model_save_dir
    
def training_run(settings, device='cpu', generate_figs=False, time_objects=False, print_to_terminal=False, train_gan=False):
    if hasattr(device, '__getitem__'):
        device = device[0]
    settings['device'] = device
    save_dir, results_save_dir, model_save_dir = get_save_dir(settings['save_dir'])
    if not print_to_terminal:
        config.specify_log_file(os.path.join(save_dir, 'log.txt'))
    with open(os.path.join(save_dir, 'settings.json'), 'w') as F:
        json.dump(settings, F, indent=2)
    if train_gan:
        trainer_class = GANTrainer
    else:
        trainer_class = ClassifierTrainer
    trainer = trainer_class(using_wandb=False, **settings)
    trainer.train_model(
        settings['total_epochs'], results_save_dir=results_save_dir, model_save_dir=model_save_dir, time_objects=time_objects
    )
    if generate_figs:
        generate_training_figs(save_dir)

def eval_run(trial_dir, device='cpu', generate_figs=False, print_to_terminal=False):
    if hasattr(device, '__getitem__'):
        device = device[0]
    model_path = os.path.join(trial_dir, 'models', 'best_model.pt')
    assert os.path.exists(model_path)
    config_path = os.path.join(trial_dir, 'settings.json')
    assert os.path.exists(config_path)
    with open(config_path, 'r') as F:
        settings = json.load(F)
    settings['device'] = device
    save_dir = os.path.join(trial_dir, 'results')
    os.makedirs(save_dir, exist_ok=True)
    if not print_to_terminal:
        config.specify_log_file(os.path.join(trial_dir, 'eval_log.txt'))
    trainer = ClassifierTrainer(using_wandb=False, **settings)
    trainer.eval_trained_model(model_path, results_save_dir=save_dir, augmented_datapoints_to_try=None)
    if generate_figs:
        generate_eval_figs(trial_dir)

def gen_figs(trial_dir):
    os.makedirs(os.path.join(trial_dir, 'figures'), exist_ok=True)
    try:
        generate_training_figs(trial_dir)
    except:
        print('Could not generate figures for training metrics.')
        traceback.print_exc()
    try:
        generate_eval_figs(trial_dir)
    except:
        print('Could not generate figures for evaluation metrics.')
        traceback.print_exc()

def spawn_agent_(sweep_id, project, device, classifier_settings, print_to_terminal, train_gan):
    wandb.agent(
        sweep_id, project=project, function=lambda: run_wandb_trial_(device, classifier_settings, print_to_terminal, train_gan)
    )

def run_wandb_trial_(device, settings, print_to_terminal, train_gan):
    try:
        settings = copy.deepcopy(settings)
        wandb.init()
        wandb_config = dict(wandb.config)
        wandb_config = config.nest_dict(wandb_config)
        for wc_key, wc_val in wandb_config.items():
            if (wc_key in settings.items()) and isinstance(wc_val, dict):
                settings[wc_key].update(wc_val)
            else:
                settings[wc_key] = wc_val
        settings['device'] = device
        save_dir, results_save_dir, model_save_dir = get_save_dir(settings['save_dir'])
        if not print_to_terminal:
            config.specify_log_file(os.path.join(save_dir, 'log.txt'))
        with open(os.path.join(save_dir, 'settings.json'), 'w') as F:
            json.dump(settings, F, indent=2)
        if train_gan:
            trainer_class = GANTrainer
        else:
            trainer_class = ClassifierTrainer
        trainer = trainer_class(using_wandb=True, **settings)
        trainer.train_model(settings['total_epochs'], results_save_dir=results_save_dir, model_save_dir=model_save_dir)
    except Exception:
        traceback.print_exc()
        with open(os.path.join(save_dir, 'log.txt'), 'a') as F:
            traceback.print_exc(file=F)
        raise Exception('Trial code crashed.')

def htune_run(settings, trials_per_gpu=1, devices='cpu', print_to_terminal=False, train_gan=False, time_objects=False):
    del time_objects ##FIXME
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
        spawn_agent_(sweep_id, settings['wandb_project'], devices[0], classifier_settings, print_to_terminal, train_gan)
    else:
        multiprocessing.set_start_method('spawn')
        procs = []
        for dev_idx, dev in enumerate(devices):
            for trial_idx in range(trials_per_gpu):
                p = multiprocessing.Process(
                    target=spawn_agent_,
                    args=(sweep_id, settings['wandb_project'], dev, classifier_settings, print_to_terminal, train_gan)
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
        '--eval', default=[], nargs='+', 
        help='Evaluate a trained model for its ability to recover a key given multiple traces.'
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
        '--time-objects', default=False, action='store_true', help='Record time taken for model forward/backward pass and dataloader traversal before start of trial.'
    )
    parser.add_argument(
        '--cudnn-benchmark', default=False, action='store_true', help='Enables cudNN autotuner to search for an efficient algorithm for convolutions.'
    )
    parser.add_argument(
        '--generate-figs', default=False, action='store_true', help='Save figures to the trial/figures directory after each trial is completed.'
    )
    parser.add_argument(
        '--generate-figs-post', default=[], nargs='+', help='Generate figures for older trials listed here.'
    )
    parser.add_argument(
        '--print-to-terminal', default=False, action='store_true',
        help='Print to the terminal instead of to a trial-specific log file.'
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
    for config_name in args.train:
        print('Executing training run defined in {} ...'.format(os.path.join(config.get_config_base_dir(train=True), config_name)))
        settings = config.load_config(config_name, train=True)
        if 'train_gan' in settings.keys():
            train_gan = settings['train_gan']
        else:
            train_gan = False
        if args.save_dir is not None:
            settings['save_dir'] = args.save_dir
        print('Settings:')
        print('\n'.join(['\t{}: {}'.format(key, val) for key, val in settings.items()]))
        training_run(settings, device=args.devices, generate_figs=args.generate_figs, time_objects=args.time_objects, print_to_terminal=args.print_to_terminal, train_gan=train_gan)
    for config_name in args.htune:
        print('Executing hyperparameter tuning run defined in {} ...'.format(os.path.join(config.HTUNE_CONFIGS, config_name)))
        settings = config.load_config(config_name, train=False)
        if args.sweep_id is not None:
            settings['sweep_id'] = args.sweep_id
        if 'train_gan' in settings.keys():
            train_gan = settings['train_gan']
        else:
            train_gan = False
        if args.save_dir is not None:
            settings['save_dir'] = args.save_dir
        print('Settings:')
        print('\n'.join(['\t{}: {}'.format(key, val) for key, val in settings.items()]))
        htune_run(settings, trials_per_gpu=args.trials_per_gpu, devices=args.devices, time_objects=args.time_objects, print_to_terminal=args.print_to_terminal, train_gan=train_gan)
    for trial_name in args.eval:
        trial_dir = os.path.join(config.RESULTS_BASE_DIR, trial_name)
        if os.path.split(trial_dir)[-1].split('_')[0] != 'trial':
            assert os.path.isdir(trial_dir)
            available_trials = [int(f.split('_')[-1]) for f in os.listdir(trial_dir)]
            trial_idx = max(available_trials)
            trial_dir = os.path.join(trial_dir, 'trial_%d'%(trial_idx))
        eval_run(trial_dir, generate_figs=args.generate_figs, device=args.devices, print_to_terminal=args.print_to_terminal)
    for trial_name in args.generate_figs_post:
        trial_dir = os.path.join(config.RESULTS_BASE_DIR, trial_name)
        if os.path.split(trial_dir)[-1].split('_')[0] != 'trial':
            assert os.path.isdir(trial_dir)
            available_trials = [int(f.split('_')[-1]) for f in os.listdir(trial_dir)]
            trial_idx = max(available_trials)
            trial_dir = os.path.join(trial_dir, 'trial_%d'%(trial_idx))
        gen_figs(trial_dir)

if __name__ == '__main__':
    main()
