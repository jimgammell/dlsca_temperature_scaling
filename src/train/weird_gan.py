import os
import json
import random
from numbers import Number
import time
from tqdm import tqdm
from copy import copy
import pickle
import numpy as np
import torch
from torch import nn, optim
from torchvision.transforms import Compose
import wandb

import config
from config import printl as print
import train, train.common
import datasets, datasets.common, datasets.transforms
import models, models.common, models.parameter_averaging, models.temperature_scaling
from datasets.transforms import construct_transform
from train.analyze_dataset import DatasetAnalyzer

class GANTrainer:
    def __init__(
        self,
        dataset_name=None, dataset_kwargs={},
        discriminator_name=None, discriminator_kwargs={},
        generator_name=None, generator_kwargs={},
        classifier_name=None, classifier_kwargs={},
        discriminator_optimizer_class=None, discriminator_optimizer_kwargs={},
        generator_optimizer_class=None, generator_optimizer_kwargs={},
        classifier_optimizer_class=None, classifier_optimizer_kwargs={},
        pert_decay=0.0,
        pert_metric='l1',
        adv_loss='negative',
        gen_drift_decay=0.0,
        percentile_to_clip=0,
        max_pert=1.0,
        max_l1sum_out=None,
        disc_steps_per_gen_step=1,
        unperturbed_prob=0.0,
        cal_temperature=False,
        train_sample_transforms=[], train_target_transforms=[], train_batch_transforms=[],
        eval_sample_transforms=[], eval_target_transforms=[], eval_batch_transforms=[],
        pretrained_disc_path=None,
        seed=None,
        device=None,
        batch_size=32,
        val_split_prop=0.2,
        pretrain_epochs=0,
        posttrain_epochs=0,
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
        assert isinstance(self.pert_decay, Number)
        assert isinstance(self.pert_metric, str) and self.pert_metric in ['l1', 'bce']
        assert isinstance(self.adv_loss, str) and self.adv_loss in ['negative', 'confusion']
        assert isinstance(self.gen_drift_decay, Number)
        assert isinstance(self.max_pert, Number)
        assert isinstance(self.percentile_to_clip, Number)
        assert isinstance(self.cal_temperature, bool)
        assert isinstance(self.disc_steps_per_gen_step, Number)
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
        self.classifier = models.construct_model(
            self.classifier_name, input_shape=self.train_dataset.dataset.data_shape, **self.classifier_kwargs
        )
        models.temperature_scaling.decorate_model(self.discriminator)
        #models.parameter_averaging.decorate_model(self.generator)
        self.discriminator_temp_optimizer = optim.SGD([self.discriminator.pre_temperature], lr=1e-1, momentum=0.9)
        if not hasattr(self.generator, 'input_shape'):
            setattr(self.generator, 'input_shape', self.train_dataset.dataset.data_shape)
        if not hasattr(self.discriminator, 'input_shape'):
            setattr(self.discriminator, 'input_shape', self.train_dataset.dataset.data_shape)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.classifier = self.classifier.to(self.device)
        self.generator_optimizer = getattr(optim, self.generator_optimizer_class)(
            self.generator.parameters(), **self.generator_optimizer_kwargs
        )
        self.discriminator_optimizer = getattr(optim, self.discriminator_optimizer_class)(
            [p for pname, p in self.discriminator.named_parameters() if pname != 'pre_temperature'],
            **self.discriminator_optimizer_kwargs
        )
        self.classifier_optimizer = getattr(optim, self.classifier_optimizer_class)(
            self.classifier.parameters(), **self.classifier_optimizer_kwargs
        )
        if self.pretrained_disc_path is not None:
            trial_dir = os.path.join(config.RESULTS_BASE_DIR, self.pretrained_disc_path)
            if not os.path.exists(trial_dir):
                print('Warning: pretrained model not found at {}'.format(trial_dir))
            if os.path.split(trial_dir)[-1].split('_')[0] != 'trial':
                assert os.path.isdir(trial_dir)
                available_trials = [int(f.split('_')[-1]) for f in os.listdir(trial_dir) if f.split('_')[0] == 'trial']
                trial_idx = max(available_trials)
                trial_dir = os.path.join(trial_dir, 'trial_%d'%(trial_idx))
            model_path = os.path.join(trial_dir, 'models', 'best_model.pt')
            if not os.path.exists(model_path):
                print('Warning: pretrained model not found at {}'.format(model_path))
            else:
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
        self.dataset_analyzer = DatasetAnalyzer(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=self.test_dataset,
            generator=self.generator,
            seed=self.seed,
            device=self.device
        )
            
    def pert_penalty_fn(self, mask):
        if self.pert_metric == 'l1':
            return nn.functional.l1_loss(mask, torch.zeros_like(mask))
        elif self.pert_metric == 'bce':
            return nn.functional.binary_cross_entropy(mask, torch.zeros_like(mask))
        else:
            assert False
    
    def adv_loss_fn(self, disc_logits, labels):
        if self.adv_loss == 'negative':
            return -nn.functional.cross_entropy(disc_logits, labels)
        elif self.adv_loss == 'confusion':
            return nn.functional.cross_entropy(disc_logits, torch.ones_like(disc_logits)/256)
        else:
            assert False
            
    def pretrain_step(
        self,
        train_batch, val_batch=None,
        **kwargs
    ):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        
        self.discriminator.train()
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
        
        gen_logits = self.generator(train_traces)
        perturbed_traces, mask = self.get_traces(gen_logits, train_traces, return_mask=True)
        gen_train_loss = self.pert_penalty_fn(mask)
        self.generator_optimizer.zero_grad(set_to_none=True)
        gen_train_loss.backward()
        self.generator_optimizer.step()
        
        d_train_logits = self.discriminator(train_traces)
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
            
    def get_traces(self, perturbation, orig_trace, return_mask=False, posteval=False):
        if posteval:
            perturbation = torch.where(perturbation > 0.0, torch.ones_like(perturbation), torch.zeros_like(perturbation))
        else:
            perturbation = nn.functional.sigmoid(perturbation)
        if posteval:
            noise = orig_trace.mean() * torch.ones_like(orig_trace)
        else:
            noise = torch.randn_like(orig_trace)
        perturbed_trace = perturbation*noise + (1-perturbation)*orig_trace
        if return_mask:
            return perturbed_trace, perturbation
        else:
            return perturbed_trace
    
    def gen_criterion(self, gen_traces, target_traces):
        return nn.functional.mse_loss(gen_traces, target_traces) + nn.functional.l1_loss(gen_traces, target_traces)
    
    def update_perturbation(self, initial_logits, traces):
        logits = initial_logits.detach().clone()
        logits.requires_grad = True
        perturb_opt = optim.LBFGS([logits], lr=1e-1, max_iter=50, history_size=50)
        def closure():
            perturb_opt.zero_grad()
            perturbed_traces = self.get_traces(logits, traces)
            perturb_loss = nn.functional.cross_entropy(
                self.discriminator(perturbed_traces),
                torch.ones(perturbed_traces.size(0), 256, dtype=torch.float, device=self.device)/256
            ) + self.pert_decay*nn.functional.l1_loss(perturbed_traces, traces)
            perturb_loss.backward()
            return perturb_loss
        perturb_opt.step(closure)
        return self.get_traces(logits.detach(), traces)
        
    def posttrain_step(self, batch, **kwargs):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        
        traces, labels = datasets.common.unpack_batch(batch, self.device)
        self.generator.eval()
        self.classifier.train()
        with torch.no_grad():
            gen_logits = self.generator(traces)
            gen_perturbed_traces = self.get_traces(gen_logits, traces, posteval=True)
        classifier_logits = self.classifier(gen_perturbed_traces)
        classifier_loss = nn.functional.cross_entropy(classifier_logits, labels)
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()
        end.record()
        torch.cuda.synchronize()
        return {
            'msec_per_batch': start.elapsed_time(end),
            'classifier_loss': classifier_loss.item(),
            'classifier_acc': train.metrics.get_acc(classifier_logits, labels)
        }
    
    @torch.no_grad()
    def posteval_step(self, batch):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        
        traces, labels = datasets.common.unpack_batch(batch, self.device)
        self.generator.eval()
        self.classifier.eval()
        gen_logits = self.generator(traces)
        gen_perturbed_traces = self.get_traces(gen_logits, traces, posteval=True)
        classifier_logits = self.classifier(gen_perturbed_traces)
        classifier_loss = nn.functional.cross_entropy(classifier_logits, labels)
        end.record()
        torch.cuda.synchronize()
        return {
            'msec_per_batch': start.elapsed_time(end),
            'classifier_loss': classifier_loss.item(),
            'classifier_acc': train.metrics.get_acc(classifier_logits, labels)
        }
        
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
            with torch.no_grad():
                gen_train_logits = self.generator(train_traces)
                gen_perturbed_traces = self.get_traces(gen_train_logits, train_traces)
            d_train_logits = self.discriminator(gen_perturbed_traces)
            d_train_loss = nn.functional.cross_entropy(d_train_logits, train_labels)
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            d_train_loss.backward()
            self.discriminator_optimizer.step()
            
            if self.cal_temperature:
                self.discriminator.eval()
                with torch.no_grad():
                    perturbed_val_traces = self.get_traces(self.generator(val_traces), val_traces)
                logits = self.discriminator(perturbed_val_traces)
                d_cal_loss = nn.functional.cross_entropy(logits, val_labels)
                self.discriminator_temp_optimizer.zero_grad()
                d_cal_loss.backward()
                self.discriminator_temp_optimizer.step()
                d_cal_ece = self.ece_criterion(logits, val_labels)
        
        if train_gen:
            # perturb phase: find a new trace which confuses the discriminator
            #   (while trying to keep L1 distance from original trace small)
            self.generator.train()
            gen_logits = self.generator(train_traces)
            gen_perturbed_traces, mask = self.get_traces(gen_logits, train_traces, return_mask=True)
            gen_loss = self.adv_loss_fn(self.discriminator(gen_perturbed_traces), train_labels) + self.pert_decay*self.pert_penalty_fn(mask)
            self.generator_optimizer.zero_grad()
            gen_loss.backward()
            self.generator_optimizer.step()
            self.generator.update_avg_buffer()
        
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
            rv['gen_loss'] = gen_loss.item()
            rv['gen_mask_l1_size'] = nn.functional.l1_loss(mask, torch.zeros_like(mask)).item()
            rv['gen_mask_prop'] = (torch.count_nonzero(mask > 0.5) / mask.numel()).item()
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
        
        perturbed_traces, mask = self.get_traces(self.generator(traces), traces, return_mask=True)
        disc_logits = self.discriminator(perturbed_traces)
        disc_loss = nn.functional.cross_entropy(disc_logits, labels)
        disc_ece = self.ece_criterion(disc_logits, labels)
        perturbation_l1_loss = nn.functional.l1_loss(perturbed_traces, traces)
        perturbation_mse_loss = nn.functional.mse_loss(perturbed_traces, traces)
        perturbation_linf_loss = (perturbed_traces-traces).norm(p=np.inf, dim=-1).mean()
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
            'perturbation_confusion_loss': perturbation_confusion_loss.item(),
            'gen_mask_l1_size': nn.functional.l1_loss(mask, torch.zeros_like(mask)).item(),
            'gen_mask_prop': (torch.count_nonzero(mask > 0.5) / mask.numel()).item()
        }
        if hasattr(self, 'pretrained_discriminator'):
            rv.update({
                'pdisc_loss': pdisc_loss.item(),
                'pdisc_acc': train.metrics.get_acc(pdisc_logits, labels)
            })
        return rv
    
    def train_epoch(self, pretrain=False, posttrain=False, **kwargs):
        disc_steps = gen_steps = 0
        rv = train.metrics.ResultsDict()
        if pretrain:
            step_fn = self.pretrain_step
        elif posttrain:
            step_fn = self.posttrain_step
        else:
            step_fn = self.train_step
        for bidx, (batch, val_batch) in enumerate(zip(self.train_dataloader, self.cal_dataloader)):
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
    
    def eval_epoch(self, dataloader, posttrain=False, **kwargs):
        if posttrain:
            step_fn = self.posteval_step
        else:
            step_fn = self.eval_step
        rv = train.common.run_epoch(dataloader, step_fn, use_progress_bar=False, **kwargs)
        per_key_means = self.dataset_analyzer.get_per_key_means(dataloader.dataset)
        sum_of_differences = self.dataset_analyzer.compute_sum_of_differences(dataloader.dataset, per_key_means=per_key_means)
        snr = self.dataset_analyzer.compute_snr(dataloader.dataset, per_key_means=per_key_means)
        rv['sum_of_differences'] = {'mean': np.mean(sum_of_differences), 'max': np.max(sum_of_differences)}
        rv['snr'] = {'mean': np.mean(snr), 'max': np.max(snr)}
        return rv
    
    def train_model(
        self,
        epochs,
        starting_checkpoint=None,
        results_save_dir=None,
        model_save_dir=None,
        posttrain_only=False
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
            self.generator.load_state_dict(starting_checkpoint['generator_state'])
            self.discriminator.load_state_dict(starting_checkpoint['discriminator_state'])
            self.generator_optimizer.load_state_dict(starting_checkpoint['generator_optimizer_state'])
            self.discriminator_optimizer.load_state_dict(starting_checkpoint['discriminator_optimizer_state'])
            starting_epoch = starting_checkpoint['epoch']
        else:
            starting_epoch = 0
        
        if not(posttrain_only) and (starting_epoch == 0):
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
        
        if not(posttrain_only):
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
                print('Using WANDB: {}'.format(self.using_wandb))
                if self.using_wandb:
                    wandb.log(rv, step=epoch_idx)
                if results_save_dir is not None:
                    with open(os.path.join(results_save_dir, 'epoch_%d.pickle'%(epoch_idx)), 'wb') as F:
                        pickle.dump(rv, F)
                if model_save_dir is not None:
                    torch.save({
                        'epoch': epoch_idx+1,
                        'generator_state': self.generator.state_dict(),
                        'discriminator_state': self.discriminator.state_dict(),
                        'generator_optimizer_state': self.generator_optimizer.state_dict(),
                        'discriminator_optimizer_state': self.discriminator_optimizer.state_dict()
                    }, os.path.join(model_save_dir, 'training_checkpoint.pt'))
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
        if (self.selection_metric is None) and self.using_wandb:
            wandb.summary.update({
                **{'final_train_%s'%(key): val for key, val in train_erv.items()},
                **{'final_val_%s'%(key): val for key, val in val_erv.items()},
                **{'final_test_%s'%(key): val for key, val in test_erv.items()}
            })
        
        print('Starting posttraining.')
        max_val_acc = -np.inf
        self.posteval_generator = torch.Generator(self.device)
        for epoch_idx in range(epochs, epochs+self.posttrain_epochs):
            print('\nStarting epoch {} ...'.format(epoch_idx))
            self.posteval_generator.manual_seed(self.seed)
            train_erv = self.train_epoch(posttrain=True)
            print('Training results:')
            for key, val in train_erv.items():
                print('\t{}: {}'.format(key, val))
            val_erv = self.eval_epoch(self.val_dataloader, posttrain=True)
            val_acc = val_erv['classifier_acc']
            if val_acc > max_val_acc:
                max_val_acc = val_acc
            print('Validation results:')
            for key, val in val_erv.items():
                print('\t{}: {}'.format(key, val))
            test_erv = self.eval_epoch(self.test_dataloader, posttrain=True)
            print('Test results:')
            for key, val in test_erv.items():
                print('\t{}: {}'.format(key, val))
            rv = {'train': train_erv.data(), 'val': val_erv.data(), 'test': test_erv.data()}
            if results_save_dir is not None:
                with open(os.path.join(results_save_dir, 'epoch_%d.pickle'%(epoch_idx)), 'wb') as F:
                    pickle.dump(rv, F)
            if self.using_wandb:
                wandb.log(rv, step=epoch_idx)
                wandb.summary.update({'best_classifier_acc': max_val_acc})

def hinge_loss(logits, y):
    return nn.functional.relu(1-y*logits).mean()
