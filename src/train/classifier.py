import os
import random
import time
from copy import copy
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision.transforms import Compose

import config
from config import printl as print
import train, train.common, train.time_objects
import datasets, datasets.common, datasets.transforms
import models, models.common

class ClassifierTrainer:
    def __init__(
        self,
        dataset_name=None, dataset_kwargs={}, # Name and configuration of dataset
        model_name=None, model_kwargs={}, # Name and configuration of model architecture
        optimizer_class=None, optimizer_kwargs={}, # Name and configuration of optimizer
        criterion_class=None, criterion_kwargs={}, # Name and configuration of loss function
        use_sam=False, sam_kwargs={}, # Whether or not to use Sharpness Aware Minimization (SAM), and configuration if so
        lr_scheduler_class=None, lr_scheduler_kwargs={}, # Learning rate scheduler class and configuration (None for no scheduler)
        rescale_temperature=False, # Whether or not to calibrate the softmax temperature after each training epoch
        precise_bn_stats=False, # Whether or not to compute precise batchnorm stats after each training epoch
        train_sample_transforms=[], train_target_transforms=[], train_batch_transforms=[], # Transforms to apply at training time
        eval_sample_transforms=[], eval_target_transforms=[], eval_batch_transforms=[], # Transforms to apply at evaluation time
        seed=None, # Random seed to initialize default RNG for Python, Numpy, and PyTorch. None to use an arbitrary seed.
        device=None, # Device on which to train the mode. None will use first available GPU, or CPU if no GPU is present.
        batch_size=32, # Batch size for both training and evaluation
        val_split_prop=0.2, # Proportion of training dataset to use for validation
        using_wandb=False, # Whether or not to track this trial using Weights and Biases.
        training_metrics={}, eval_metrics={}, # Metrics to run after every training + evaluation batch.
        selection_metric=None, maximize_selection_metric=False, # Which metric to use to find the best model.
        **kwargs
    ):
        if len(kwargs) > 0:
            print('Warning: unused kwargs with names \'{}\''.format('\', \''.join(list(kwargs.keys()))))
        # Save the arguments for future reference
        for var_name, var in locals().items():
            setattr(self, var_name, var)
        
        # Ensure arguments are valid
        assert any(x.__name__ == self.dataset_name for x in datasets.AVAILABLE_DATASETS)
        assert isinstance(self.dataset_kwargs, dict)
        assert any(x.__name__ == self.model_name for x in models.AVAILABLE_MODELS)
        assert isinstance(self.model_kwargs, dict)
        assert hasattr(optim, self.optimizer_class)
        assert isinstance(self.optimizer_kwargs, dict)
        assert hasattr(nn, self.criterion_class)
        assert isinstance(self.criterion_kwargs, dict)
        assert (self.lr_scheduler_class is None) or hasattr(optim.lr_scheduler, self.lr_scheduler_class)
        assert isinstance(self.lr_scheduler_kwargs, dict)
        assert isinstance(self.rescale_temperature, bool)
        assert isinstance(self.precise_bn_stats, bool)
        for tf_list in [self.train_sample_transforms, self.train_target_transforms, self.train_batch_transforms,
                        self.eval_sample_transforms, self.eval_target_transforms, self.eval_batch_transforms]:
            assert isinstance(tf_list, list)
            assert all(
                hasattr(datasets.transforms, tf) if isinstance(tf, str)
                else hasattr(datasets.transforms, tf[0])
                for tf in tf_list
            )
        assert (self.seed is None) or isinstance(self.seed, int)
        assert self.device in [None, 'cpu', 'cuda', *['cuda:%d'%(dev_idx) for dev_idx in range(torch.cuda.device_count())]]
        assert isinstance(self.batch_size, int) and (self.batch_size > 0)
        assert isinstance(self.val_split_prop, float) and (0.0 < self.val_split_prop < 1.0)
        assert isinstance(self.using_wandb, bool)
        assert isinstance(self.training_metrics, dict) \
          and all((isinstance(name, str) and hasattr(train.metrics, fn)) for name, fn in self.training_metrics.items())
        assert isinstance(self.eval_metrics, dict) \
          and all((isinstance(name, str) and hasattr(train.metrics, fn)) for name, fn in self.eval_metrics.items())
        assert self.selection_metric in self.eval_metrics.keys()
        assert isinstance(self.maximize_selection_metric, bool)
        
    # Initialize the datasets, dataloaders, model, optimizer and learning rate scheduler.
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
        self.train_dataloader = datasets.common.get_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = datasets.common.get_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = datasets.common.get_dataloader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.model = models.construct_model(self.model_name, input_shape=self.train_dataset.dataset.data_shape, **self.model_kwargs)
        if not hasattr(self.model, 'input_shape'):
            model.input_shape = self.train_dataset.data_shape
        self.model = self.model.to(self.device)
        if self.rescale_temperature:
            self.model = models.temperature_scaling(self.model)
            self.model = self.model.to(self.device)
        if self.use_sam:
            self.optimizer = train.sam.SAM(
                model.parameters(), getattr(optim, self.optimizer_class), **self.sam_kwargs, **self.optimizer_kwargs
            )
        else:
            self.optimizer = getattr(optim, self.optimizer_class)(self.model.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler_class == 'OneCycleLR':
            self.lr_scheduler_kwargs['total_steps'] = epochs*len(self.train_dataloader)
        elif self.lr_scheduler_class == 'CosineAnnealingLR':
            self.lr_scheduler_kwargs['T_max'] = epochs*len(self.train_dataloader)
        self.lr_scheduler = getattr(optim.lr_scheduler, self.lr_scheduler_class)(self.optimizer, **self.lr_scheduler_kwargs)
        self.criterion = getattr(nn, self.criterion_class)(**self.criterion_kwargs)
        self.training_metrics = {
            mname: getattr(train.metrics, mclass) if isinstance(mclass, str) else mclass
            for mname, mclass in self.training_metrics.items()
        }
        self.eval_metrics = {
            mname: getattr(train.metrics, mclass) if isinstance(mclass, str) else mclass
            for mname, mclass in self.eval_metrics.items()
        }
    
    # This function is called for each training minibatch to update the model parameters. It will do the forwards and backwards pass
    #   to compute the gradients, update the weights as specified by the chosen optimizer, and report the metrics recorded for this
    #   batch (e.g. loss, accuracy, rank).
    def train_step(self, batch):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        rv = train.metrics.ResultsDict() # Store training metrics recorded for this batch
        self.model.train() # Set model to training mode (e.g. enable dropout, batchnorm uses single-minibatch stats)
        traces, labels = datasets.common.unpack_batch(batch, self.device) # Move batch to appropriate device (probably a GPU)
        if self.train_batch_transform is not None: # Apply data transforms that operate on the full minibatch, e.g. mixup, cutmix
            traces, labels = self.train_batch_transform((traces, labels))
        
        # Function to do the forward and backward passes. Necessary for optimizers such as SAM which use multiple passes per
        #   parameter update.
        def closure(
            update_rv=False # Whether to report training metrics. We generally want to do this once per batch.
        ):
            nonlocal rv
            logits = self.model(traces) # Do the forward pass
            loss = self.criterion(logits, labels) # Compute the training loss for this pass
            loss.backward() # Do the backward pass
            if update_rv:
                rv['loss'] = loss.item() # Convert loss to standard Python number -- saving/returning PyTorch tensors causes issues
                for metric_name, metric_fn in self.training_metrics.items(): # Compute additional metrics
                    rv[metric_name] = metric_fn(logits.detach(), labels)
                norms = train.metrics.get_norms(self.model) # Compute model parameter weight/gradient norms for debugging
                rv.update(norms)
            return loss # This is needed by the optimizer
        self.optimizer.zero_grad(set_to_none=True) # Reset the gradient buffer before the forwards/backwards pass
        _ = closure(update_rv=True) # Do the first forwards and backwards pass
        if self.use_sam: # Do the optimizer step, which will entail more forwards/backwards passes for e.g. SAM
            self.optimizer.step(closure)
        else:
            self.optimizer.step()
        if self.lr_scheduler is not None: # Update the learning rate if we are using a scheduler
            rv['lr'] = self.lr_scheduler.get_last_lr()
            self.lr_scheduler.step()
        else:
            rv['lr'] = [g['lr'] for g in self.optimizer.param_groups][0]
        end.record()
        torch.cuda.synchronize()
        rv['time_per_batch'] = start.elapsed_time(end)
        return rv
    
    @torch.no_grad() # Don't record the computation graph within this function since we aren't doing a backwards pass.
    def eval_step(self, batch):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        rv = train.common.ResultsDict() # Store metrics recorded during this batch
        self.model.eval() # Set model to evaluation mode (e.g. disable dropout, batchnorm uses running stats over full dataset)
        traces, labels = datasets.common.unpack_batch(batch, self.device) # Move batch to appropriate device
        if self.eval_batch_transform is not None: # Apply data transforms that operate on the full minibatch
            traces, labels = self.eval_batch_transform((traces, labels))
        logits = self.model(traces) # Do the forward pass
        loss = self.criterion(logits, labels) # Compute loss value for this batch
        rv['loss'] = loss.item() # Convert loss to Python number
        for metric_name, metric_fn in self.eval_metrics.items(): # Compute additional metrics
            rv[metric_name] = metric_fn(logits, labels)
        end.record()
        torch.cuda.synchronize()
        rv['time_per_batch'] = start.elapsed_time(end)
        return rv
    
    # Run the model on a dataset in order to compute accurate activation statistics for the batchnorm layers. Can be run once before
    #   evaluation on the validation/test dataset to improve performance.
    # The default PyTorch behavior is to record an exponentially-weighted moving average of layer activation means/variances during 
    #   training time, then use these stats at test-time so that the model isn't reliant on having a full minibatch of test examples 
    #   for every inference. However, these EMA stats lag the true stats, which has been shown to degrade test-time performance and
    #   can lead to the illusion of overfitting.
    @torch.no_grad() # Don't record computation graph
    def update_bn_stats(self):
        self.model.eval() # Set the model to evaluation mode
        train_dataloader_transform = self.train_dataloader.dataset.dataset.transform
        self.train_dataloader.dataset.dataset.transform = self.val_dataloader.dataset.dataset.transform
        for module in self.model.modules():
            if isinstance(module, models.common.PreciseBatchNorm):
                module.reset_running_stats() # Discard existing batchnorm statistics so we can record new ones
                module.train() # Put the batchnorm layers in training mode so that the statistics will be updated
        for batch in self.train_dataloader: # Do a forward pass for each batch, which will automatically update the batchnorm statistics.
            traces, labels = datasets.common.unpack_batch(batch, self.device)
            if self.eval_batch_transform is not None:
                traces, labels = self.eval_batch_transform((traces, labels))
            _ = self.model(traces)
        self.train_dataloader.dataset.dataset.transform = train_dataloader_transform
        
    def train_epoch(self, **kwargs):
        rv = train.common.run_epoch(self.train_dataloader, self.train_step, **kwargs)
        if self.precise_bn_stats:
            self.update_bn_stats()
        if self.rescale_temperature:
            self.model.set_temperature(self.val_dataloader)
        return rv
        
    def eval_epoch(self, dataloader, **kwargs):
        rv = train.common.run_epoch(dataloader, self.eval_step, **kwargs)
        return rv
    
    def train_model(
        self,
        epochs, # Number of epochs the model will be trained for
        time_objects=False,
        starting_checkpoint=None, # Path to checkpoint to use as starting point.
        results_save_dir=None, # Directory in which to save the metrics recorded during training
        model_save_dir=None # Directory in which to save model checkpoints and/or best model found
    ):
        print('Initializing trial ...')
        self.reset(epochs)
        if results_save_dir is not None:
            print('Results save directory: {}'.format(results_save_dir))
        if model_save_dir is not None:
            print('Model save directory: {}'.format(model_save_dir))
        print('Model:')
        print(self.model)
        print('Train dataset: {}'.format(self.train_dataset.dataset))
        print('Test dataset: {}'.format(self.test_dataset))
        print('Train/val lengths: {} / {}'.format(len(self.train_dataset), len(self.val_dataset)))
        print('Train/val/test dataloaders: {} / {} / {}'.format(self.train_dataloader, self.val_dataloader, self.test_dataloader))
        print('Optimizer: {}'.format(self.optimizer))
        print('Criterion: {}'.format(self.criterion))
        print('Learning rate scheduler: {}'.format(self.lr_scheduler))
        print('Device: {}'.format(self.device))
        print('Seed: {}'.format(self.seed))
        if self.using_wandb:
            wandb.summary.update({'param_count': sum(p.numel() for p in self.model.parameters() if p.requires_grad)})
        
        if time_objects:
            print('\nTesting model forward/backward pass and dataloader iterate times ...')
            forward_pass_time = train.time_objects.time_model_forward_pass(
                self.model, batch_size=self.batch_size, device=self.device
            )
            print('\tForward pass time: {} msec/batch'.format(forward_pass_time))
            backward_pass_time = train.time_objects.time_model_backward_pass(
                self.model, batch_size=self.batch_size, device=self.device
            )
            print('\tBackward pass time: {} msec/batch'.format(backward_pass_time))
            train_dataloader_time = train.time_objects.time_dataloader(self.train_dataloader) / 1000
            print('\tTrain dataloader time: {} sec/epoch'.format(train_dataloader_time))
            val_dataloader_time = train.time_objects.time_dataloader(self.val_dataloader) / 1000
            print('\tVal dataloader time: {} sec/epoch'.format(val_dataloader_time))
            test_dataloader_time = train.time_objects.time_dataloader(self.test_dataloader) / 1000
            print('\tTest dataloader time: {} sec/epoch'.format(test_dataloader_time))
            if self.using_wandb:
                wandb.summary.update({
                    'forward_pass_msec': forward_pass_time,
                    'backward_pass_msec': backward_pass_time,
                    'train_dataloader_sec': train_dataloader_time,
                    'val_dataloader_sec': val_dataloader_time,
                    'test_dataloader_sec': test_dataloader_time
                })
        
        if starting_checkpoint is not None:
            print('\nLoading model from checkpoint at {} ...'.format(starting_checkpoint))
            starting_checkpoint = torch.load(starting_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
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
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
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
                        torch.save(self.model.state_dict(), os.path.join(model_save_dir, 'best_model.pt'))
                    if self.using_wandb:
                        wandb.summary.update({
                            'best_epoch': epoch_idx,
                            **{'best_train_%s'%(key): val for key, val in train_erv.items()},
                            **{'best_val_%s'%(key): val for key, val in val_erv.items()},
                            **{'best_test_%s'%(key): val for key, val in test_erv.items()}
                        })
        
def generate_figs(trial_dir):
    # Load results into memory
    results_dir = os.path.join(trial_dir, 'results')
    epochs = []
    collected_results = {'train': [], 'val': [], 'test': []}
    for filename in os.listdir(results_dir):
        epoch = int(filename.split('_')[-1].split('.')[0])
        epochs.append(epoch)
        with open(os.path.join(results_dir, filename), 'rb') as F:
            results = pickle.load(F)
        for phase, phase_results in results.items():
            for metric_name, metric_sample in phase_results.items():
                if not metric_name in collected_results[phase].keys():
                    collected_results[phase][metric_name] = []
                collected_results[phase][metric_name].append(metric_sample)
    epochs = np.array(epochs)
    sorted_indices = np.argsort(epochs)
    epochs = epochs[sorted_indices]
    for phase, phase_results in collected_results.items():
        for metric_name, metric_trace in phase_results.items():
            collected_results[phase][metric_name] = np.array(metric_trace)[sorted_indices]
    
    # Plot results
    num_metrics = len(collected_results['train'])
    assert num_metrics == len(collected_results['val']) == len(collected_results['test'])
    rows = int(np.sqrt(num_metrics))
    cols = int(np.ceil(num_metrics))
    (fig, axes) = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    for phase in ['train', 'val']:
        for ax, (metric_name, metric_trace) in zip(axes.flatten(), collected_results[phase].items()):
            ax.plot(
                epochs, metric_trace,
                color='blue', linestyle={'train': '--', 'val': '-'}[phase], label='{}_{}'.format(phase, metric_name)
            )
            if ax.get_ylabel() != 'metric_name':
                ax.set_ylabel(metric_name)
    for ax in axes.flatten():
        ax.set_xlabel('Epochs')
        ax.legend()
        ax.grid(True)
        
    # Save results
    fig.savefig(os.path.join(trial_dir, 'figures', 'collected_results.pdf'))

def construct_transform(transforms):
    constructed_transforms = []
    for val in transforms:
        if isinstance(val, list):
            tf, tf_kwargs = val
        else:
            tf = val
            tf_kwargs = {}
        if isinstance(tf, str):
            tf = getattr(datasets.transforms, tf)
        constructed_transforms.append(tf(**tf_kwargs))
    composed_transform = Compose(constructed_transforms)
    return composed_transform