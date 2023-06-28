import os
import json
import random
import time
from tqdm import tqdm
from copy import copy
import pickle
import numpy as np
import torch
from torch import nn, optim
from torchvision.transforms import Compose

import config
from config import printl as print
import train, train.common
import datasets, datasets.common, datasets.transforms
import models, models.common, models.parameter_averaging, models.temperature_scaling
from datasets.transforms import construct_transform

class GANTrainer:
    def __init__(
        self,
        dataset_name=None, dataset_kwargs={},
        discriminator_name=None, discriminator_kwargs={},
        generator_name=None, generator_kwargs={},
        discriminator_optimizer_class=None, discriminator_optimizer_kwargs={},
        generator_optimizer_class=None, generator_optimizer_kwargs={},
        pert_l1_decay=0.0,
        disc_steps_per_gen_step=1,
        cal_temperature=False,
        train_sample_transforms=[], train_target_transforms=[], train_batch_transforms=[],
        eval_sample_transforms=[], eval_target_transforms=[], eval_batch_transforms=[],
        pretrained_disc_path=None,
        seed=None,
        device=None,
        batch_size=32,
        val_split_prop=0.2,
        pretrain_epochs=0,
        using_wandb=False,
        selection_metric=None, maximize_selection_metric=False,
        **kwargs
    ):
        if len(kwargs) > 0:
            print('Warning: unused kwargs with names \'{}\''.format('\', \''.join(list(kwargs.keys()))))
        for var_name, var in locals().items():
            setattr(self, var_name, var)
        
        assert any(x.__name__ == self.dataset_name for x in datasets.AVAILABLE_DATASETS)
        assert isinstance(self.dataset_kwargs, dict)
        assert any(x.__name__ == self.discriminator_name for x in models.AVAILABLE_MODELS)
        assert isinstance(self.discriminator_kwargs, dict)
        assert any(x.__name__ == self.generator_name for x in models.AVAILABLE_MODELS)
        assert isinstance(self.generator_kwargs, dict)
        assert hasattr(optim, self.discriminator_optimizer_class)
        assert isinstance(self.discriminator_optimizer_kwargs, dict)
        assert hasattr(optim, self.generator_optimizer_class)
        assert isinstance(self.generator_optimizer_kwargs, dict)
        assert isinstance(self.pert_l1_decay, float)
        assert isinstance(self.cal_temperature, bool)
        assert isinstance(self.disc_steps_per_gen_step, float)
        for tf_list in [self.train_sample_transforms, self.train_target_transforms, self.train_batch_transforms,
                        self.eval_sample_transforms, self.eval_target_transforms, self.eval_batch_transforms]:
            assert isinstance(tf_list, list)
            assert all(
                hasattr(datasets.transforms, tf) if isinstance(tf, str)
                else hasattr(datasets.transforms, tf[0])
                for tf in tf_list
            )
        if pretrained_disc_path is not None:
            assert isinstance(pretrained_disc_path, str)
        assert (self.seed is None) or isinstance(self.seed, int)
        assert self.device in [None, 'cpu', 'cuda', *['cuda:%d'%(dev_idx) for dev_idx in range(torch.cuda.device_count())]]
        assert isinstance(self.batch_size, int) and (self.batch_size > 0)
        assert isinstance(self.val_split_prop, float) and (0.0 < self.val_split_prop < 1.0)
        assert isinstance(self.using_wandb, bool)
        assert isinstance(self.pretrain_epochs, int)
    
    def reset(self, epochs=None):
        if self.device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self.seed is None:
            self.seed = time.time_ns() & 0xFFFFFFFF
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.train_dataset = datasets.construct_dataset(self.dataset_name, train=True, **self.dataset_kwargs)
        val_split_size = int(self.val_split_prop*len(self.train_dataset))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, (len(self.train_dataset)-val_split_size, val_split_size)
        )
        self.val_dataset.dataset = copy(self.val_dataset.dataset)
        self.test_dataset = datasets.construct_dataset(self.dataset_name, train=False, **self.dataset_kwargs)
        self.train_sample_transform = construct_transform(self.train_sample_transforms)
        self.train_target_transform = construct_transform(self.train_target_transforms)
        self.train_batch_transform = construct_transform(self.train_batch_transforms)
        self.eval_sample_transform = construct_transform(self.eval_sample_transforms)
        self.eval_target_transform = construct_transform(self.eval_target_transforms)
        self.eval_batch_transform = construct_transform(self.eval_batch_transforms)
        self.train_dataset.dataset.transform = self.train_sample_transform
        self.train_dataset.dataset.target_transform = self.train_target_transform
        self.val_dataset.dataset.transform = self.eval_sample_transform
        self.val_dataset.dataset.target_transform = self.eval_target_transform
        self.test_dataset.transform = self.eval_sample_transform
        self.test_dataset.target_transform = self.eval_target_transform
        self.train_dataloader = datasets.common.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = datasets.common.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = datasets.common.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.cal_dataloader = datasets.common.DataLoader(
            torch.utils.data.ConcatDataset([
                self.val_dataset for _ in range(int(np.ceil(len(self.train_dataset)/len(self.val_dataset))))
            ]), batch_size=self.batch_size, shuffle=True
        )
        self.generator = models.construct_model(
            self.generator_name, input_shape=self.train_dataset.dataset.data_shape, **self.generator_kwargs
        )
        self.discriminator = models.construct_model(
            self.discriminator_name, input_shape=self.train_dataset.dataset.data_shape, **self.discriminator_kwargs
        )
        models.temperature_scaling.decorate_model(self.discriminator)
        self.discriminator_temp_optimizer = optim.SGD([self.discriminator.pre_temperature], lr=1e-1, momentum=0.9)
        if not hasattr(self.generator, 'input_shape'):
            setattr(self.generator, 'input_shape', self.train_dataset.dataset.data_shape)
        if not hasattr(self.discriminator, 'input_shape'):
            setattr(self.discriminator, 'input_shape', self.train_dataset.dataset.data_shape)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.generator_optimizer = getattr(optim, self.generator_optimizer_class)(
            self.generator.parameters(), **self.generator_optimizer_kwargs
        )
        self.discriminator_optimizer = getattr(optim, self.discriminator_optimizer_class)(
            [p for pname, p in self.discriminator.named_parameters() if pname != 'pre_temperature'],
            **self.discriminator_optimizer_kwargs
        )
        if self.pretrained_disc_path is not None:
            trial_dir = os.path.join(config.RESULTS_BASE_DIR, self.pretrained_disc_path)
            if os.path.split(trial_dir)[-1].split('_')[0] != 'trial':
                assert os.path.isdir(trial_dir)
                available_trials = [int(f.split('_')[-1]) for f in os.listdir(trial_dir)]
                trial_idx = max(available_trials)
                trial_dir = os.path.join(trial_dir, 'trial_%d'%(trial_idx))
            model_path = os.path.join(trial_dir, 'models', 'best_model.pt')
            assert os.path.exists(model_path)
            config_path = os.path.join(trial_dir, 'settings.json')
            assert os.path.exists(config_path)
            with open(config_path, 'r') as F:
                settings = json.load(F)
            self.pretrained_discriminator = models.construct_model(
                settings['model_name'], input_shape=self.train_dataset.dataset.data_shape, **settings['model_kwargs']
            ).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.pretrained_discriminator.load_state_dict(state_dict)
        self.ece_criterion = models.temperature_scaling.ECELoss()
            
    def pretrain_step(
        self,
        train_batch, val_batch=None,
        **kwargs
    ):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        
        self.discriminator.train()
        self.generator.train()
        train_traces, train_labels = datasets.common.unpack_batch(train_batch, self.device)
        if self.train_batch_transform is not None:
            train_traces, train_labels = self.train_batch_transform((train_traces, train_labels))
        if self.cal_temperature:
            assert val_batch is not None
            val_traces, val_labels = datasets.common.unpack_batch(val_batch, self.device)
            if self.eval_batch_transform is not None:
                val_traces, val_labels = self.eval_batch_transform((val_traces, val_labels))
        batch_size = train_traces.size(0)
        num_classes = 256
        
        perturbed_traces = self.generator(train_traces)
        gen_train_loss = nn.functional.l1_loss(perturbed_traces, train_traces)
        self.generator_optimizer.zero_grad(set_to_none=True)
        gen_train_loss.backward()
        self.generator_optimizer.step()
        perturbed_traces = perturbed_traces.detach()
        
        d_train_logits = self.discriminator(perturbed_traces)
        d_train_loss = nn.functional.cross_entropy(d_train_logits, train_labels)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        d_train_loss.backward()
        self.discriminator_optimizer.step()
        
        if self.cal_temperature:
            self.discriminator.eval()
            logits = self.discriminator(val_traces)
            d_cal_loss = nn.functional.cross_entropy(logits, val_labels)
            self.discriminator_temp_optimizer.zero_grad()
            d_cal_loss.backward()
            self.discriminator_temp_optimizer.step()
            d_cal_ece = self.ece_criterion(logits, val_labels)
        
        end.record()
        torch.cuda.synchronize()
        rv = {
            'msec_per_batch': start.elapsed_time(end),
            'd_temp': self.discriminator.get_temperature().item(),
            'd_train_loss': d_train_loss.item(),
            'd_train_acc': train.metrics.get_acc(d_train_logits, train_labels),
            'gen_train_loss': gen_train_loss.item()
        }
        if self.cal_temperature:
            rv.update({
                'd_cal_loss': d_cal_loss.item(),
                'd_cal_ece': d_cal_ece.item()
            })
        return rv
            
    def train_step(
        self,
        train_batch, val_batch=None,
        train_gen=True,
        train_disc=True
    ):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        
        self.discriminator.train()
        self.generator.train()
        train_traces, train_labels = datasets.common.unpack_batch(train_batch, self.device)
        if self.train_batch_transform is not None:
            train_traces, train_labels = self.train_batch_transform((train_traces, train_labels))
        if self.cal_temperature:
            assert val_batch is not None
            val_traces, val_labels = datasets.common.unpack_batch(val_batch, self.device)
            if self.eval_batch_transform is not None:
                val_traces, val_labels = self.eval_batch_transform((val_traces, val_labels))
        batch_size = train_traces.size(0)
        num_classes = 256
        
        if train_disc:
            # d_train phase: update discriminator parameters to improve loss on a training batch
            with torch.no_grad():
                perturbed_train_traces = self.generator(train_traces)
            d_train_logits = self.discriminator(perturbed_train_traces)
            d_train_loss = nn.functional.cross_entropy(d_train_logits, train_labels)
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            d_train_loss.backward()
            self.discriminator_optimizer.step()

        if self.cal_temperature:
            # d_cal phase: update discriminator softmax temperature to improve loss on a validation batch
            self.discriminator.eval()
            with torch.no_grad():
                perturbed_val_traces = self.generator(val_traces)
            logits = self.discriminator(perturbed_val_traces)
            d_cal_loss = nn.functional.cross_entropy(logits, val_labels)
            self.discriminator_temp_optimizer.zero_grad()
            d_cal_loss.backward()
            self.discriminator_temp_optimizer.step()
            d_cal_ece = self.ece_criterion(logits, val_labels)
        
        if train_gen:
            # perturb phase: find a new trace which confuses the discriminator
            #   (while trying to keep L1 distance from original trace small)
            perturbed_train_traces.requires_grad = True
            perturb_opt = optim.LBFGS([perturbed_train_traces], lr=1e-1)
            def closure():
                perturb_opt.zero_grad()
                perturb_loss = nn.functional.cross_entropy(
                    self.discriminator(perturbed_train_traces),
                    torch.ones(batch_size, num_classes, dtype=torch.float, device=self.device)/num_classes
                ) + self.pert_l1_decay*nn.functional.l1_loss(perturbed_train_traces, train_traces)
                perturb_loss.backward()
                return perturb_loss
            num_perturbation_steps = 1
            prev_val = perturbed_train_traces.detach().clone()
            perturb_opt.step(closure)
            while (num_perturbation_steps < 10) and ((perturbed_train_traces - prev_val).norm(2) > 1e-2):
                prev_val = perturbed_train_traces.clone()
                perturb_opt.step(closure)
                num_perturbation_steps += 1

            # g_train phase: update generator parameters so that it outputs the perturbed trace
            gen_perturbed_traces = self.generator(train_traces)
            gen_train_loss = nn.functional.l1_loss(gen_perturbed_traces, perturbed_train_traces.detach())
            self.generator_optimizer.zero_grad(set_to_none=True)
            gen_train_loss.backward()
            self.generator_optimizer.step()
        
        end.record()
        torch.cuda.synchronize()
        rv = {
            'msec_per_batch': start.elapsed_time(end),
            'd_temp': self.discriminator.get_temperature().item()
        }
        if train_disc:
            rv['d_train_loss'] = d_train_loss.item()
            rv['d_train_acc'] = train.metrics.get_acc(d_train_logits, train_labels)
        if self.cal_temperature:
            rv['d_cal_loss'] = d_cal_loss.item()
            rv['d_cal_ece'] = d_cal_ece.item()
        if train_gen:
            rv['perturbation_steps'] = num_perturbation_steps
            rv['perturbation_dist'] = (gen_perturbed_traces - perturbed_train_traces).norm(p='fro').item()
            rv['gen_train_loss'] = gen_train_loss.item()
        return rv
    
    @torch.no_grad()
    def eval_step(self, batch):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        
        self.generator.eval()
        self.discriminator.eval()
        traces, labels = datasets.common.unpack_batch(batch, self.device)
        if self.eval_batch_transform is not None:
            traces, labels = self.eval_batch_transform((traces, labels))
        batch_size = traces.size(0)
        num_classes = 256
        
        perturbed_traces = self.generator(traces)
        disc_logits = self.discriminator(perturbed_traces)
        disc_loss = nn.functional.cross_entropy(disc_logits, labels)
        disc_ece = self.ece_criterion(disc_logits, labels)
        perturbation_l1_loss = nn.functional.l1_loss(perturbed_traces, traces)
        perturbation_confusion_loss = nn.functional.cross_entropy(
            self.discriminator(perturbed_traces),
            torch.ones(batch_size, num_classes, dtype=torch.float, device=self.device)/num_classes
        )
        
        if hasattr(self, 'pretrained_discriminator'):
            pdisc_logits = self.pretrained_discriminator(perturbed_traces)
            pdisc_loss = nn.functional.cross_entropy(pdisc_logits, labels)
        
        end.record()
        torch.cuda.synchronize()
        rv = {
            'msec_per_batch': start.elapsed_time(end),
            'd_temp': self.discriminator.get_temperature().item(),
            'd_train_loss': disc_loss.item(),
            'd_train_acc': train.metrics.get_acc(disc_logits, labels),
            'd_cal_ece': disc_ece.item(),
            'perturbation_l1_loss': perturbation_l1_loss.item(),
            'perturbation_confusion_loss': perturbation_confusion_loss.item()
        }
        if hasattr(self, 'pretrained_discriminator'):
            rv.update({
                'pdisc_loss': pdisc_loss.item(),
                'pdisc_acc': train.metrics.get_acc(pdisc_logits, labels)
            })
        return rv
    
    def train_epoch(self, pretrain=False, **kwargs):
        disc_steps = gen_steps = 0
        rv = train.metrics.ResultsDict()
        if pretrain:
            step_fn = self.pretrain_step
        else:
            step_fn = self.train_step
        for bidx, (batch, val_batch) in enumerate(zip(tqdm(self.train_dataloader), self.cal_dataloader)):
            if self.disc_steps_per_gen_step > 1:
                disc_steps += 1
                train_disc = True
                train_gen = disc_steps >= self.disc_steps_per_gen_step
                if train_gen:
                    disc_steps -= self.disc_steps_per_gen_step
            elif 1/self.disc_steps_per_gen_step > 1:
                gen_steps += 1
                train_gen = True
                train_disc = gen_steps >= 1/self.disc_steps_per_gen_step
                if train_disc:
                    gen_steps -= 1/self.disc_steps_per_gen_step
            else:
                train_gen = train_disc = True
            step_rv = step_fn(batch, val_batch=val_batch, train_gen=train_gen, train_disc=train_disc, **kwargs)
            rv.update(step_rv)
        rv.reduce(np.mean)
        return rv
    
    def eval_epoch(self, dataloader, **kwargs):
        rv = train.common.run_epoch(dataloader, self.eval_step, use_progress_bar=True, **kwargs)
        return rv
    
    def train_model(
        self,
        epochs,
        starting_checkpoint=None,
        results_save_dir=None,
        model_save_dir=None
    ):
        print('Initializing trial ...')
        self.reset(epochs)
        if results_save_dir is not None:
            print('Results save directory: {}'.format(results_save_dir))
        if model_save_dir is not None:
            print('Model save directory: {}'.format(model_save_dir))
        print('Generator:')
        print(self.generator)
        print('Discriminator:')
        print(self.discriminator)
        print('Train dataset: {}'.format(self.train_dataset.dataset))
        print('Val dataset: {}'.format(self.val_dataset.dataset))
        print('Test dataset: {}'.format(self.test_dataset))
        print('Train/val lengths: {} / {}'.format(len(self.train_dataset), len(self.val_dataset)))
        print('Train/val/test dataloaders: {} / {} / {}'.format(self.train_dataloader, self.val_dataloader, self.test_dataloader))
        print('Gen/disc optimizers: {} / {}'.format(self.generator_optimizer, self.discriminator_optimizer))
        print('Device: {}'.format(self.device))
        print('Seed: {}'.format(self.seed))
        
        if starting_checkpoint is not None:
            print('Loading model from checkpoint at {} ...'.format(starting_checkpoint))
            starting_checkpoint = torch.load(starting_checkpoint, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state'])
            self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state'])
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state'])
            starting_epoch = checkpoint['epoch']
        else:
            starting_epoch = 0
        
        print('Starting pretraining.')
        for epoch_idx in range(-self.pretrain_epochs, 0):
            print('\nStarting epoch {} ...'.format(epoch_idx))
            train_erv = self.train_epoch(pretrain=True)
            print('Training results:')
            for key, val in train_erv.items():
                print('\t{}: {}'.format(key, val))
            val_erv = self.eval_epoch(self.val_dataloader)
            print('Validation results:')
            for key, val in val_erv.items():
                print('\t{}: {}'.format(key, val))
            test_erv = self.eval_epoch(self.test_dataloader)
            print('Test results:')
            for key, val in test_erv.items():
                print('\t{}: {}'.format(key, val))
            rv = {'train': train_erv.data(), 'val': val_erv.data(), 'test': test_erv.data()}
            if self.using_wandb:
                wandb.log(rv, step=epoch_idx)
        
        print('Starting training.')
        best_metric = -np.inf
        for epoch_idx in range(starting_epoch, epochs):
            print('\nStarting epoch {} ...'.format(epoch_idx))
            train_erv = self.train_epoch()
            print('Training results:')
            for key, val in train_erv.items():
                print('\t{}: {}'.format(key, val))
            val_erv = self.eval_epoch(self.val_dataloader)
            print('Validation results:')
            for key, val in val_erv.items():
                print('\t{}: {}'.format(key, val))
            test_erv = self.eval_epoch(self.test_dataloader)
            print('Test results:')
            for key, val in test_erv.items():
                print('\t{}: {}'.format(key, val))
            rv = {'train': train_erv.data(), 'val': val_erv.data(), 'test': test_erv.data()}
            if results_save_dir is not None:
                with open(os.path.join(results_save_dir, 'epoch_%d.pickle'%(epoch_idx)), 'wb') as F:
                    pickle.dump(rv, F)
            if model_save_dir is not None:
                torch.save({
                    'epoch': epoch_idx,
                    'generator_state': self.generator.state_dict(),
                    'discriminator_state': self.discriminator.state_dict(),
                    'generator_optimizer_state': self.generator_optimizer.state_dict(),
                    'discriminator_optimizer_state': self.discriminator_optimizer.state_dict()
                }, os.path.join(model_save_dir, 'training_checkpoint.pt'))
            if self.using_wandb:
                wandb.log(rv, step=epoch_idx)
            if self.selection_metric is not None:
                metric = val_erv[self.selection_metric]
                if not self.maximize_selection_metric:
                    metric *= -1
                if metric > best_metric:
                    print('New best model found.')
                    best_metric = metric
                    if model_save_dir is not None:
                        torch.save(self.generator.state_dict(), os.path.join(model_save_dir, 'best_generator.pt'))
                    if self.using_wandb:
                        wandb.summary.update({
                            'best_epoch': epoch_idx,
                            **{'best_train_%s'%(key): val for key, val in train_erv.items()},
                            **{'best_val_%s'%(key): val for key, val in val_erv.items()},
                            **{'best_test_%s'%(key): val for key, val in test_erv.items()}
                        })

def hinge_loss(logits, y):
    return nn.functional.relu(1-y*logits).mean()