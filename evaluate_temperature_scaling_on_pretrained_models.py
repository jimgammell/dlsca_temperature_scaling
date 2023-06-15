import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader

from datasets.ascadv1 import ASCADv1
from datasets.zaid_dpav4 import DPAv4_Zaid
from models.pretrained_models import PretrainedProuff, PretrainedZaid
from models.temperature_scaling import ModelWithTemperature, calibrate_temperature
from train.accumulate_traces import eval_data, perform_attacks, plot_attacks

n_attacks = 100
traces_per_attack = 1000
device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu'
trials = [
    {
        'dataset': lambda **kwargs: ASCADv1(desync=0, variable=False, **kwargs),
        'model': (
            lambda: PretrainedProuff(variable=False, desync=0, arch='cnn', device=device),
            lambda: PretrainedProuff(variable=False, desync=0, arch='mlp', device=device),
            lambda: PretrainedZaid(dataset='ASCAD_desync0')
        )
    }, {
        'dataset': lambda **kwargs: ASCADv1(desync=50, variable=False, **kwargs),
        'model': (
            lambda: PretrainedProuff(variable=False, desync=50, arch='cnn', device=device),
            lambda: PretrainedProuff(variable=False, desync=50, arch='mlp', device=device),
            lambda: PretrainedZaid(dataset='ASCAD_desync50')
        )
    }, {
        'dataset': lambda **kwargs: ASCADv1(desync=100, variable=False, **kwargs),
        'model': (
            lambda: PretrainedProuff(variable=False, desync=100, arch='cnn', device=device),
            lambda: PretrainedProuff(variable=False, desync=100, arch='mlp', device=device),
            lambda: PretrainedZaid(dataset='ASCAD_desync100')
        )
    }, {
        'dataset': lambda **kwargs: ASCADv1(desync=0, variable=True, **kwargs),
        'model': (lambda: PretrainedProuff(variable=True, desync=0, device=device),)
    }, {
        'dataset': lambda **kwargs: ASCADv1(desync=50, variable=True, **kwargs),
        'model': (lambda: PretrainedProuff(variable=True, desync=50, device=device),)
    }
]

trial_idx = 0
for trial in trials:
    for model_constructor in trial['model']:
        model = model_constructor()
        #train_dataset = trial['dataset'](train=True)
        #if isinstance(model, PretrainedProuff):
        #    val_split_size = int(0.1*len(train_dataset))
        #elif isinstance(model, PretrainedZaid):
        #    val_split_size = 5000
        
        test_dataset = trial['dataset'](train=False)
        val_dataset = test_dataset
        ## FIXME: Using test dataset for calibration since I don't know which samples were used for training vs. validation
        
        val_dataloader = DataLoader(val_dataset, batch_size=256, num_workers=1)
        test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=1)
        print('Running trial %d ... '%(trial_idx))
        print('Dataset:')
        print(test_dataset)
        print('Model:')
        print(model)
        print('Evaluating predictions on test dataset...')
        logits, plaintexts, keys = eval_data(model, test_dataloader, device)
        print('Computing attacks...')
        uc_rank_evolutions, uc_ttd = perform_attacks(n_attacks, traces_per_attack, logits, plaintexts, keys, 2)
        print('\tMean rank: {}'.format(np.mean(uc_rank_evolutions)))
        print('\tMean traces to disclosure: {}'.format(np.nanmean(uc_ttd)))
        
        print('Calibrating temperature...')
        model = ModelWithTemperature(model).to(device)
        rv = calibrate_temperature(model, val_dataloader, device)
        print('\tDone. Final temperature: {}. NLL: {} -> {} , ECE: {} -> {} .'.format(rv['final_temperature'], rv['pre_nll'], rv['post_nll'], rv['pre_ece'], rv['post_ece']))
        print('Evaluating predictions on test dataset...')
        logits /= nn.functional.softplus(model.temperature).item()
        print('Computing attacks...')
        c_rank_evolutions, c_ttd = perform_attacks(n_attacks, traces_per_attack, logits, plaintexts, keys, 2)
        print('\tMean rank: {}'.format(np.mean(c_rank_evolutions)))
        print('\tMean traces to disclosure: {}'.format(np.nanmean(c_ttd)))
        
        print('Evaluating with excessively-low temperature...')
        logits *= 1e6
        print('Computing attacks...')
        bc_rank_evolutions, bc_ttd = perform_attacks(n_attacks, traces_per_attack, logits, plaintexts, keys, 2)
        print('\tMean rank: {}'.format(np.mean(bc_rank_evolutions)))
        print('\tMean traces to disclosure: {}'.format(np.nanmean(bc_ttd)))
        
        save_dir = os.path.join('.', 'figures', 'temperature_scaling_pretrained')
        os.makedirs(save_dir, exist_ok=True)
        plot_attacks(
            uc_rank_evolutions, c_rank_evolutions, bc_rank_evolutions,
            title='%s: %s'%(test_dataset.name, model.name),
            save_path=os.path.join(save_dir, 'trial_%d.png'%(trial_idx))
        )
        trial_idx += 1