import os
import random
import time
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
import models, models.common

class CycleGANTrainer:
    def __init__(
        self,
        dataset_name=None, dataset_kwargs={},
        discriminator_name=None, discriminator_kwargs={},
        generator_name=None, generator_kwargs={},
        discriminator_optimizer_class=None, discriminator_optimizer_kwargs={},
        generator_optimizer_class=None, generator_optimizer_kwargs={},
        train_sample_transforms=[], train_target_transforms=[], train_batch_transforms=[],
        eval_sample_transforms=[], eval_target_transforms=[], eval_batch_transforms=[],
        seed=None,
        device=None,
        batch_size=32,
        val_split_prop=0.2,
        using_wandb=False,
        selection_metric=None, maximize_selection_metric=False,
        **kwargs
    ):
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
        for tf_list in [self.train_sample_transforms, self.train_target_transforms, self.train_batch_transforms,
                        self.eval_sample_transforms, self.eval_target_transforms, self.eval_batch_transforms]:
            assert isinstance(tf_list, list)
            assert all(hasattr(datasets.transforms, tf) for tf in tf_list)
        assert (self.seed is None) or isinstance(self.seed, int)
        assert self.device in [None, 'cpu', 'cuda', *['cuda:%d'%(dev_idx) for dev_idx in range(torch.cuda.device_count())]]
        assert isinstance(self.batch_size, int) and (self.batch_size > 0)
        assert isinstance(self.val_split_prop, float) and (0.0 < self.val_split_prop < 1.0)
        assert isinstance(self.using_wandb, bool)
    
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
        self.train_sample_transform = Compose([getattr(datasets.transforms, tf)() for tf in self.train_sample_transforms])
        self.train_target_transform = Compose([getattr(datasets.transforms, tf)() for tf in self.train_target_transforms])
        self.train_batch_transform = Compose([getattr(datasets.transforms, tf)() for tf in self.train_batch_transforms])
        self.eval_sample_transform = Compose([getattr(datasets.transforms, tf)() for tf in self.eval_sample_transforms])
        self.eval_target_transform = Compose([getattr(datasets.transforms, tf)() for tf in self.eval_target_transforms])
        self.eval_batch_transform = Compose([getattr(datasets.transforms, tf)() for tf in self.eval_batch_transforms])
        self.train_dataset.dataset.transform = self.train_sample_transform
        self.train_dataset.dataset.target_transform = self.train_target_transform
        self.val_dataset.dataset.transform = self.eval_sample_transform
        self.val_dataset.dataset.target_transform = self.eval_target_transform
        self.test_dataset.transform = self.eval_sample_transform
        self.test_dataset.target_transform = self.eval_target_transform
        self.train_dataloader = datasets.common.get_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = datasets.common.get_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = datasets.common.get_dataloader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.generator = models.construct_model(
            self.generator_name, input_shape=self.train_dataset.dataset.data_shape, **self.generator_kwargs
        )
        self.discriminator = models.construct_model(
            self.discriminator_name, input_shape=self.train_dataset.dataset.data_shape, **self.discriminator_kwargs
        )
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.generator_optimizer = getattr(optim, self.generator_optimizer_class)(
            self.generator.parameters(), **self.generator_optimizer_kwargs
        )
        self.discriminator_optimizer = getattr(optim, self.discriminator_optimizer_class)(
            self.discriminator.parameters(), **self.discriminator_optimizer_kwargs
        )
    
    def train_step(self, batch, train_gen=True, train_disc=True):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        
        rv = train.metrics.ResultsDict()
        traces, labels = datasets.common.unpack_batch(batch, self.device)
        if self.train_batch_transform is not None:
            traces, labels = self.train_batch_transform((traces, labels))
        rec_labels = torch.randint(0, self.train_dataset.num_outputs, size=labels.size(0), dtype=torch.long, device=y.device)
        
        if train_disc:
            self.generator.eval()
            self.discriminator.train()
            
            with torch.no_grad():
                traces_rec = self.generator(traces, rec_labels)
            disc_features_orig = self.discriminator.extract_features(traces)
            disc_features_rec = self.discriminator.extract_features(traces_rec)
            
            disc_logits_orig_leakage = self.discriminator.classify_leakage(disc_features_orig)
            disc_loss_orig_leakage = nn.functional.cross_entropy_loss(disc_logits_orig_leakage, labels)
            disc_logits_rec_leakage = self.discriminator.classify_leakage(disc_features_rec)
            disc_loss_rec_leakage = nn.functional.cross_entropy_loss(disc_logits_rec_leakage, labels)
            
            disc_logits_orig_realism = self.discriminator.classify_realism(disc_features_orig, labels)
            disc_loss_orig_realism = hinge_loss(disc_logits_orig_realism, 1)
            disc_logits_rec_realism = self.discriminator.classify_realism(disc_features_rec, labels_rec)
            disc_loss_rec_realism = hinge_loss(disc_logits_rec_realism, -1)
            disc_loss_realism = disc_loss_orig_realism + disc_loss_rec_realism
            
            disc_loss = disc_loss_leakage + disc_loss_realism
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()
            
            rv.update({
                'disc_loss_realism': disc_loss_realism.item(),
                'disc_loss_leakage': disc_loss_leakage.item(),
                'disc_loss_orig_realism': disc_loss_orig_realism.item(),
                'disc_loss_rec_realism': disc_loss_rec_realism.item(),
                'disc_acc_realism': 0.5*train.metrics.hinge_acc(disc_logits_orig_realism, 1) + 0.5*train.metrics.hinge_acc(disc_logits_rec_realism, -1),
                'disc_acc_orig_leakage': train.metrics.accuracy(disc_logits_orig_leakage, labels),
                'disc_acc_rec_leakage': train.metrics.accuracy(disc_logits_rec_leakage, labels)
            })
            rv.update({'disc_acc_leakage': 0.5*rv['disc_acc_orig_leakage'] + 0.5*rv['disc_acc_rec_leakage']})
            
        if train_gen:
            self.generator.train()
            self.discriminator.eval()
            
            traces_rec = self.generator(traces, rec_labels)
            disc_features_rec = self.discriminator.extract_features(traces_rec)
            
            disc_logits_rec_leakage = self.discriminator.classify_leakage(disc_features_rec)
            gen_loss_leakage = nn.functional.cross_entropy_loss(disc_logits_rec_leakage, labels_rec)
            
            disc_logits_rec_realism = self.discriminator.classify_realism(disc_features_rec, labels_rec)
            gen_loss_realism = -disc_logits_rec_realism.mean()
            
            gen_loss = gen_loss_leakage + gen_loss_realism
            
            if self.perturbation_l1_penalty > 0:
                gen_perturbation_l1_loss = nn.functional.l1_loss(traces, traces_rec)
                gen_loss += self.perturbation_l1_penalty*gen_perturbation_l1_loss
                rv.update({'gen_perturbation_l1_loss': gen_perturbation_l1_loss.item()})
            
            if self.cyclical_l1_penalty > 0:
                traces_crec = self.generator(traces_rec, labels)
                gen_cyclical_l1_loss = nn.functional.l1_loss(traces, traces_crec)
                gen_loss += self.cyclical_l1_penalty*gen_cyclical_l1_loss
                rv.update({'gen_cyclical_l1_loss': gen_cyclical_l1_loss.item()})
            
            if self.parameter_drift_penalty > 0:
                gen_parameter_drift_loss = self.generator.get_parameter_drift_loss()
                gen_loss += self.parameter_drift_penalty*gen_parameter_drift_loss
                rv.update({'gen_parameter_drift_loss': gen_parameter_drift_loss.item()})
            
            rv.update({
                'gen_loss_leakage': gen_loss_leakage.item(),
                'gen_loss_realism': gen_loss_realism.item(),
                'gen_loss': gen_loss.item()
            })
            
            self.generator_optimizer.zero_grad()
            gen_loss.backward()
            self.generator_optimizer.step()
        
            if self.parameter_drift_penalty > 0:
                self.generator.update_average_parameters()            
        
        end.record()
        torch.cuda.synchronize()
        rv['time_per_batch'] = start.elapsed_time(end)
        return rv
    
    @torch.no_grad()
    def eval_step(self, batch):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        rv = train.common.ResultsDict()
        self.generator.eval()
        self.discriminator.eval()
        
        traces, labels = datasets.common.unpack_batch(batch, self.device)
        if self.eval_batch_transform is not None:
            traces, labels = self.eval_batch_transform((traces, labels))
        labels_rec = torch.randint(0, self.train_dataset.num_outputs, size=labels.size(0), dtype=torch.long, device=self.device)
        traces_rec = self.generator(traces, labels_rec)
        disc_features_orig = self.discriminator.extract_features(traces)
        disc_features_rec = self.discriminator.extract_features(traces_rec)
        
        disc_logits_orig_leakage = self.discriminator.classify_leakage(disc_features_orig)
        disc_logits_rec_leakage = self.discriminator.classify_leakage(disc_features_rec)
        disc_loss_orig_leakage = nn.functional.cross_entropy_loss(disc_logits_orig_leakage, labels)
        disc_loss_rec_leakage = nn.functional.cross_entropy_loss(disc_logits_rec_leakage, labels_rec)
        disc_loss_leakage = disc_loss_orig_leakage + disc_loss_rec_leakage
        
        disc_logits_orig_realism = disc.classify_realism(disc_features_orig, labels)
        disc_logits_rec_realism = disc.classify_realism(disc_features_rec, labels_rec)
        disc_loss_orig_realism = hinge_loss(disc_logits_orig_realism, 1)
        disc_loss_rec_realism = hinge_loss(disc_logits_rec_realism, -1)
        disc_loss_realism = disc_loss_orig_realism + disc_loss_rec_realism
        
        disc_loss = disc_loss_leakage + disc_loss_realism
        rv.update({
            'disc_loss_realism': disc_loss_realism.item(),
            'disc_loss_leakage': disc_loss_leakage.item(),
            'disc_loss': disc_loss.item(),
            'disc_acc_realism': 0.5*train.metrics.hinge_acc(disc_logits_orig_realism, 1) + 0.5*train.metrics.hinge_acc(disc_logits_rec_realism, -1),
            'disc_acc_leakage': 0.5*train.metrics.accuracy(disc_logits_orig_leakage, labels) + 0.5*train.metrics.accuracy(disc_logits_rec_leakage, labels_rec)
        })
        
        gen_loss_leakage = nn.functional.cross_entropy_loss(disc_logits_rec_leakage, labels_rec)
        gen_loss_realism = -disc_logits_rec_realism.mean()
        rv.update({
            'gen_loss_leakage': gen_loss_leakage.item(),
            'gen_loss_realism': gen_loss_realism.item()
        })
        
        if self.perturbation_l1_penalty > 0:
            gen_perturbation_l1_loss = nn.functional.l1_loss(traces, traces_rec)
            gen_loss += self.perturbation_l1_penalty*gen_perturbation_l1_loss
            rv.update({'gen_perturbation_l1_loss': gen_perturbation_l1_loss.item()})
            
        if self.cyclical_l1_penalty > 0:
            traces_crec = self.generator(traces_rec, labels)
            gen_cyclical_l1_loss = nn.functional.l1_loss(traces, traces_crec)
            gen_loss += self.cyclical_l1_penalty*gen_cyclical_l1_loss
            rv.update({'gen_cyclical_l1_loss': gen_cyclical_l1_loss.item()})
            
        if self.parameter_drift_penalty > 0:
            gen_parameter_drift_loss = self.generator.get_parameter_drift_loss()
            gen_loss += self.parameter_drift_penalty*gen_parameter_drift_loss
            rv.update({'gen_parameter_drift_loss': gen_parameter_drift_loss.item()})
        
        rv.update({'gen_loss': gen_loss})
        end.record()
        torch.cuda.synchronize()
        rv['time_per_batch'] = start.elapsed_time(end)
        return rv
    
    def train_epoch(self, **kwargs):
        disc_steps = gen_steps = 0
        rv = train.metrics.ResultsDict()
        for bidx, batch in enumerate(self.train_dataloader):
            if self.disc_steps_per_gen_step > 1:
                disc_steps += 1
                train_disc = True
                train_gen = disc_steps >= self.disc_steps_per_gen_step
                if train_gen:
                    disc_steps -= self.disc_steps_per_gen_step
            elif 1/disc_steps_per_gen_step > 1:
                gen_steps += 1
                train_gen = True
                train_disc = gen_steps >= 1/self.disc_steps_per_gen_step
                if train_disc:
                    gen_steps -= 1/self.disc_steps_per_gen_step
            else:
                train_gen = train_disc = True
            step_rv = self.train_step(batch, train_gen=train_gen, train_disc=train_disc, **kwargs)
            rv.update(step_rv)
        return rv
    
    def eval_epoch(self, dataloader, **kwargs):
        rv = train.common.run_epoch(dataloader, self.eval_step, **kwargs)
        return rv
    
    def train_model(
        self,
        epochs,
        starting_checkpoint=None,
        results_save_dir=None,
        model_save_dir=one
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
        print('Train/val/test datasets: {} / {} / {}'.format(self.train_dataset, self.val_dataset, self.test_dataset))
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
