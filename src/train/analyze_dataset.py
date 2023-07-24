import os
import json
import time
from tqdm import tqdm
import random
import numpy as np
import torch
from torchvision.transforms import Compose, Lambda
from matplotlib import pyplot as plt

import datasets, models
from datasets import transforms
from models.template_attack import TemplateAttacker

class DatasetAnalyzer:
    def __init__(
        self,
        dataset_name=None, dataset_kwargs={},
        train_dataset=None, val_dataset=None, test_dataset=None,
        trial_dir=None,
        generator=None,
        seed=None,
        device=None,
        val_split_prop=0.2
    ):
        if seed is None:
            seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if all(x is not None for x in (train_dataset, val_dataset, test_dataset)):
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
        elif all(x is None for x in (train_dataset, val_dataset, test_dataset)):
            transform = Compose([
                transforms.ToFloatTensor(),
                transforms.Downsample(downsample_ratio=4),
                transforms.Standardize(mean=-0.2751, stdev=0.1296)
            ])
            self.train_dataset = datasets.construct_dataset(dataset_name, train=True, transform=transform, **dataset_kwargs)
            self.data_shape = self.train_dataset.data_shape[1:]
            val_split_size = int(val_split_prop*len(self.train_dataset))
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, (len(self.train_dataset)-val_split_size, val_split_size)
            )
            self.test_dataset = datasets.construct_dataset(dataset_name, train=False, transform=transform, **dataset_kwargs)
        else:
            assert False
        if trial_dir is not None:
            settings_path = os.path.join(trial_dir, 'settings.json')
            with open(settings_path, 'r') as F:
                settings = json.load(F)
            self.generator = models.construct_model(
                settings['generator_name'],
                input_shape=self.train_dataset.dataset.data_shape,
                **settings['generator_kwargs']
            )
            generator_path = os.path.join(trial_dir, 'models', 'training_checkpoint.pt')
            checkpoint = torch.load(generator_path, map_location='cpu')
            self.generator.load_state_dict({
                key: val for key, val in checkpoint['generator_state'].items() if not key.split('__')[0] == 'avg'
            })
            if device is None:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.generator = self.generator.to(device)
            self.generator.eval()
        elif generator is not None:
            self.generator = generator

        if hasattr(self, 'generator'):
            self.train_dataset.dataset.transform = Compose([self.train_dataset.dataset.transform, Lambda(self.obfuscate_sample)])
            self.val_dataset.dataset.transform = Compose([self.val_dataset.dataset.transform, Lambda(self.obfuscate_sample)])
            self.test_dataset.transform = Compose([self.test_dataset.transform, Lambda(self.obfuscate_sample)])

    def obfuscate_sample(self, sample):
        self.generator.eval()
        sample = sample.to(self.device)
        sample = sample.unsqueeze(0)
        with torch.no_grad():
            mask = self.generator(sample)
        mask = torch.where(mask > 0.0, torch.ones_like(mask), torch.zeros_like(mask))
        noise = sample.mean() * torch.ones_like(sample)
        sample = mask*noise + (1-mask)*sample
        sample = sample.squeeze(0)
        sample = sample.to('cpu')
        return sample
    
    def compute_trace_offsets(self, dataset, reference_idx=0, max_offset=100, progress_bar=False):
        reference_trace, _ = dataset[reference_idx]
        reference_trace = reference_trace.squeeze()
        if isinstance(reference_trace, torch.Tensor):
            reference_trace = reference_trace.numpy()
        offsets = []
        wrapper = tqdm if progress_bar else lambda x: x
        for trace, _ in wrapper(self.train_dataset):
            trace = trace.squeeze()
            if isinstance(trace, torch.Tensor):
                trace = trace.numpy()
            correlations = np.correlate(trace[max_offset:-max_offset], reference_trace, mode='valid')
            offset = np.argmax(correlations) - max_offset
            offsets.append(offset)
        return offsets
    
    def get_per_key_means(self, dataset, progress_bar=False):
        per_key_means = {}
        wrapper = tqdm if progress_bar else lambda x: x
        for trace, key in wrapper(dataset):
            trace = trace.squeeze()
            if isinstance(trace, torch.Tensor):
                trace = trace.numpy()
            if isinstance(key, torch.Tensor):
                key = key.item()
            if not key in per_key_means.keys():
                per_key_means[key] = [0, np.zeros_like(trace)]
            mn_samples, mn_trace = per_key_means[key]
            per_key_means[key][1] = (mn_samples/(mn_samples+1))*mn_trace + (1/(mn_samples+1))*trace
            per_key_means[key][0] += 1
        per_key_means = {key: val[1] for key, val in per_key_means.items()}
        return per_key_means
    
    def compute_sum_of_differences(self, dataset, per_key_means=None, progress_bar=False):
        if per_key_means is None:
            per_key_means = self.get_per_key_means(dataset)
        sum_diffs = np.zeros_like(list(per_key_means.values())[0])
        if progress_bar:
            pbar = tqdm(total=len(per_key_means)**2 - len(per_key_means))
        for key1 in per_key_means.keys():
            for key2 in per_key_means.keys():
                if key1 == key2:
                    continue
                sum_diffs += np.abs(per_key_means[key1] - per_key_means[key2])
                if progress_bar:
                    pbar.update(1)
        return sum_diffs
    
    def compute_snr(self, dataset, per_key_means=None, progress_bar=False):
        if per_key_means is None:
            per_key_means = self.get_per_key_means(dataset)
        signal_variance = np.var(np.array(list(per_key_means.values())), axis=0)
        noise_variance = np.zeros_like(list(per_key_means.values())[0])
        wrapper = tqdm if progress_bar else lambda x: x
        for trace, key in wrapper(dataset):
            trace = trace.squeeze()
            if isinstance(trace, torch.Tensor):
                trace = trace.numpy()
            if isinstance(key, torch.Tensor):
                key = key.item()
            noise_variance += (trace - per_key_means[key])**2
        noise_variance /= len(dataset)
        snr = signal_variance / noise_variance
        return snr
    
    def __call__(self, figs_dir=None, max_offset=100):
        print('Computing per-key means ...')
        per_key_means = self.get_per_key_means(self.train_dataset, progress_bar=True)
        print('Computing sums of differences ...')
        sum_of_differences = self.compute_sum_of_differences(self.train_dataset, per_key_means=per_key_means, progress_bar=True)
        print('Computing signal-to-noise ratios ...')
        snr = self.compute_snr(self.train_dataset, per_key_means=per_key_means, progress_bar=True)
        print('Computing trace offsets ...')
        trace_offsets = self.compute_trace_offsets(self.train_dataset, max_offset=max_offset, progress_bar=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        axes[0].plot(sum_of_differences, color='blue')
        axes[0].set_title('SOD method')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Sum of differences')
        axes[0].grid(True)
        #axes[0].set_yscale('log')
        axes[1].plot(snr, color='blue')
        axes[1].set_title('SNR method')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Signal-noise ratio')
        axes[1].grid(True)
        #axes[1].set_yscale('log')
        plt.tight_layout()
        if figs_dir is not None:
            fig.savefig(os.path.join(figs_dir, 'poi_method_comparison.png'))
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(trace_offsets, color='blue', bins=np.arange(-max_offset, max_offset+1), density=True)
        ax.set_yscale('symlog', linthresh=1e0)
        ax.set_title('Trace offsets')
        ax.set_xlabel('Offset value')
        ax.set_ylabel('Count')
        if figs_dir is not None:
            fig.savefig(os.path.join(figs_dir, 'offsets.png'))
