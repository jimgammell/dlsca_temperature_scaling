import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from datasets.common import AES_SBOX

def get_attack_point(key, plaintext):
    return AES_SBOX[key ^ plaintext]

def get_hamming_weight(val):
    hamming_weight = 0
    while val > 0:
        hamming_weight += 1
        val &= val - 1
    return int(hamming_weight)

class SyntheticAES__Finite(Dataset):
    def __init__(
        self,
        dataset_size=10000,
        num_samples=100,
        points=None,
        num_leaking_samples=1,
        num_xor_samples=0,
        fixed_noise_stdev=1.0,
        noise_momentum=0.0,
        varying_noise_stdev=0.5,
        hamming_weight_variance_prop=0.5,
        target='hamming_weight',
        delay_size=0,
        transform=None,
        target_transform=None,
        rng=None
    ):
        for var_name, var in locals().items():
            setattr(self, var_name, var)
        super().__init__()
        
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        
        assert (num_xor_samples >= 0) and (num_xor_samples != 1)
        if points is None:
            points = self.rng.choice(np.arange(delay_size, num_samples), size=num_leaking_samples+num_xor_samples, replace=False)
        self.leakage_points_mask = np.zeros((1, num_samples), dtype=bool)
        if num_leaking_samples > 0:
            self.leakage_points = points[:num_leaking_samples]
            for pt in self.leakage_points:
                self.leakage_points_mask[:, pt] = True
                self.leakage_points_mask[:, pt-delay_size//2-delay_size%2:pt+delay_size//2] = True
        else:
            self.leakage_points = None
        self.xor_samples_mask = np.zeros((1, num_samples), dtype=bool)
        if num_xor_samples > 0:
            self.xor_samples = points[-num_xor_samples:]
            for pt in self.xor_samples:
                self.xor_samples_mask[:, pt] = True
                self.xor_samples_mask[:, pt-delay_size//2-delay_size%2:pt+delay_size//2] = True
        else:
            self.xor_samples = None
        self.fixed_noise = fixed_noise_stdev*self.rng.standard_normal((1, num_samples+delay_size), dtype=float)
        
        self.traces, self.targets, self.metadata = [], [], []
        for _ in range(dataset_size):
            trace, target, metadata = self.get_datapoint()
            self.traces.append(trace)
            self.targets.append(target)
            self.metadata.append(metadata)
        
    def get_power_consumption(self, val, size=1):
        if not isinstance(self.hamming_weight_variance_prop, list):
            self.hamming_weight_variance_prop = size*[self.hamming_weight_variance_prop]
        power_consumption = np.empty(size, dtype=float)
        bit_consumption = ((np.unpackbits(val).astype(bool)).sum() - 4) / np.sqrt(2)
        for idx in range(size):
            random_consumption = self.rng.standard_normal(dtype=float)
            power_consumption[idx] = self.varying_noise_stdev*(np.sqrt(1-self.hamming_weight_variance_prop[idx])*random_consumption + np.sqrt(self.hamming_weight_variance_prop[idx])*bit_consumption)
        return power_consumption
        
    def get_datapoint(self, key=None, plaintext=None, attack_point=None):
        if (attack_point is None) and (key is None) and (plaintext is None):
            key = self.rng.integers(256, dtype=np.uint8)
            plaintext = self.rng.integers(256, dtype=np.uint8)
            attack_point = get_attack_point(key, plaintext)
            if self.xor_samples is not None:
                masks = self.rng.integers(256, dtype=np.uint8, size=len(self.xor_samples)-1)
                masked_attack_point = attack_point
                for mask in masks:
                    masked_attack_point = mask ^ masked_attack_point
        trace = self.fixed_noise.copy()
        trace += self.varying_noise_stdev*self.rng.standard_normal((1, self.num_samples+self.delay_size), dtype=float)
        if self.leakage_points is not None:
            trace[:, self.leakage_points + self.delay_size//2] = self.fixed_noise[:, self.leakage_points]
            trace[:, self.leakage_points + self.delay_size//2] += self.get_power_consumption(attack_point, size=len(self.leakage_points))
        if self.xor_samples is not None:
            trace[:, self.xor_samples + self.delay_size//2] = self.fixed_noise[:, self.xor_samples]
            for point, val in zip(self.xor_samples, list(masks) + [masked_attack_point]):
                trace[:, point + self.delay_size//2] += self.get_power_consumption(val, size=len(self.xor_samples))
        if self.noise_momentum > 0:
            trace = np.concatenate((trace, trace), axis=1)
            for idx in range(1, trace.shape[1]):
                trace[:, idx] = self.noise_momentum*trace[:, idx-1] + trace[:, idx]
            trace = trace[:, trace.shape[1]//2:]
        if self.delay_size > 0:
            delay = self.rng.integers(self.delay_size+1)
            trace = trace[:, delay:]
            trace = trace[:, :self.num_samples]
        
        metadata = {
            'attack_point': attack_point,
            'key': key,
            'plaintext': plaintext
        }
        if self.xor_samples is not None:
            metadata.update({
                'masks': masks,
                'masked_attack_point': masked_attack_point
            })
        if self.target == 'identity':
            target = attack_point
        elif self.target == 'hamming_weight':
            target = np.unpackbits(attack_point).sum()
        else:
            assert False
        return trace, target, metadata
    
    def __getitem__(self, idx, return_metadata=False):
        trace = self.traces[idx]
        target = self.targets[idx]
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if return_metadata:
            metadata = self.metadata[idx]
            return trace, target, metadata
        else:
            return trace, target
    
    def __len__(self):
        return self.dataset_size

class SyntheticAES__Infinite(IterableDataset):
    def __init__(
        self,
        num_samples=100,
        num_leaking_samples=1,
        num_xor_samples=0,
        fixed_noise_stdev=1.0,
        noise_momentum=0.0,
        varying_noise_stdev=0.5,
        mean_bit_consumption=0.1,
        stdev_bit_consumption=0.025,
        target='hamming_weight',
        delay_size=0,
        transform=None,
        target_transform=None,
        rng=None
    ):
        for var_name, var in locals().items():
            setattr(self, var_name, var)
        super().__init__()
        
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        
        assert (num_xor_samples >= 0) and (num_xor_samples != 1)
        points = self.rng.choice(np.arange(delay_size, num_samples), size=num_leaking_samples+num_xor_samples, replace=False)
        if num_leaking_samples > 0:
            self.leakage_points = points[:num_leaking_samples]
            self.leakage_points_mask = np.zeros((1, num_samples+delay_size), dtype=bool)
            self.leakage_points_mask[:, self.leakage_points] = True
        else:
            self.leakage_points = None
        if num_xor_samples > 0:
            self.xor_samples = points[-num_xor_samples:]
            self.xor_samples_mask = np.zeros((1, num_samples+delay_size), dtype=bool)
            self.xor_samples_mask[:, self.xor_samples] = True
        else:
            self.xor_samples = None
        self.fixed_noise = fixed_noise_stdev*self.rng.standard_normal((1, num_samples+delay_size), dtype=float)
        self.bit_power_consumption = np.empty((8,), dtype=float)
        for bidx in range(8):
            power_consumption = -np.inf
            while not(0.25*mean_bit_consumption <= power_consumption <= 4*mean_bit_consumption):
                power_consumption = stdev_bit_consumption*self.rng.standard_normal(dtype=float) + mean_bit_consumption
            self.bit_power_consumption[bidx] = power_consumption
        
    def get_power_consumption(self, val):
        bit_consumption = (self.bit_power_consumption * np.unpackbits(val).astype(bool)).sum() - 4*np.mean(self.bit_power_consumption)
        random_consumption = np.sqrt(max(self.varying_noise_stdev**2 - 2*self.mean_bit_consumption**2, 0))*self.rng.standard_normal(dtype=float)
        return bit_consumption + random_consumption
        
    def get_datapoint(self, key=None, plaintext=None, attack_point=None):
        if (attack_point is None) and (key is None) and (plaintext is None):
            key = self.rng.integers(256, dtype=np.uint8)
            plaintext = self.rng.integers(256, dtype=np.uint8)
            attack_point = get_attack_point(key, plaintext)
            if self.xor_samples is not None:
                masks = self.rng.integers(256, dtype=np.uint8, size=len(self.xor_samples)-1)
                masked_attack_point = attack_point
                for mask in masks:
                    masked_attack_point = mask ^ masked_attack_point
        trace = self.fixed_noise.copy()
        trace += self.varying_noise_stdev*self.rng.standard_normal((1, self.num_samples+self.delay_size), dtype=float)
        if self.leakage_points is not None:
            trace[:, self.leakage_points + self.delay_size//2] = self.fixed_noise[self.leakage_points_mask]
            trace[:, self.leakage_points + self.delay_size//2] += self.get_power_consumption(attack_point)
        if self.xor_samples is not None:
            trace[:, self.xor_samples + self.delay_size//2] = self.fixed_noise[self.xor_samples_mask]
            for point, val in zip(self.xor_samples, list(masks) + [masked_attack_point]):
                trace[:, point + self.delay_size//2] += self.get_power_consumption(val)
        if self.noise_momentum > 0:
            trace = np.concatenate((trace, trace), axis=1)
            for idx in range(1, trace.shape[1]):
                trace[:, idx] = self.noise_momentum*trace[:, idx-1] + trace[:, idx]
            trace = trace[:, trace.shape[1]//2:]
        if self.delay_size > 0:
            delay = self.rng.integers(self.delay_size)
            trace = trace[:, delay:]
            trace = trace[:, :self.num_samples]
        
        metadata = {
            'attack_point': attack_point,
            'key': key,
            'plaintext': plaintext
        }
        if self.xor_samples is not None:
            metadata.update({
                'masks': masks,
                'masked_attack_point': masked_attack_point
            })
        if self.target == 'identity':
            target = attack_point
        elif self.target == 'hamming_weight':
            target = np.unpackbits(attack_point).sum()
        else:
            assert False
        return trace, target, metadata
    
    def __next__(self, key=None, plaintext=None, return_metadata=False):
        
        trace, target, metadata = self.get_datapoint(key=key, plaintext=plaintext)
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if return_metadata:
            return trace, target, metadata
        else:
            return trace, target
    
    def __iter__(self):
        return self

AVAILABLE_DATASETS = [SyntheticAES__Finite, SyntheticAES__Infinite]