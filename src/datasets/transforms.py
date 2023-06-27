import sys
import numpy as np
import torch
from torch import nn
from torchvision.transforms import Compose

def construct_transform(transforms):
    constructed_transforms = []
    for val in transforms:
        if isinstance(val, list):
            tf, tf_kwargs = val
        else:
            tf = val
            tf_kwargs = {}
        if isinstance(tf, str):
            tf = getattr(sys.modules[__name__], tf)
        constructed_transforms.append(tf(**tf_kwargs))
    composed_transform = Compose(constructed_transforms)
    return composed_transform

class Transform(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        for var_name, var in kwargs.items():
            setattr(self, var_name, var)
    
    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(
            ', '.join(['{}={}'.format(key, val) for key, val in self.kwargs.items()])
        )

# Converts a numpy float array trace to a PyTorch tensor
class ToFloatTensor(Transform):
    def forward(self, x):
        return torch.as_tensor(x, dtype=torch.float)

class ToLongTensor(Transform):
    def forward(self, x):
        return torch.as_tensor(x, dtype=torch.long)

# Converts a numpy integer array label to a PyTorch label vector in one-hot form
class ToOneHot(Transform):
    def forward(self, x):
        return nn.functional.one_hot(x, num_classes=256).to(torch.float)

class Downsample(Transform):
    def __init__(self, downsample_ratio):
        super().__init__(downsample_ratio=downsample_ratio)
        
    def forward(self, x):
        return nn.functional.max_pool1d(x, self.downsample_ratio)

class Standardize(Transform):
    def __init__(self, mean=0.0, stdev=1.0):
        super().__init__(mean=mean, stdev=stdev)
        
    def forward(self, x):
        return (x - self.mean) / self.stdev
    
class Normalize(Transform):
    def __init__(self, min=-1.0, max=1.0):
        super().__init__(min=min, max=max)
    
    def forward(self, x):
        return 2 * (x - 0.5*(self.min+self.max)) / (self.max-self.min)

# Smooths out one-hot labels to reflect that there is some uncertainty in the true label
class LabelSmoothing(Transform):
    def __init__(
        self,
        eps=0.1 # Probability that the target label is incorrect
    ):
        super().__init__(eps=eps)
    
    def forward(self, x):
        return (1-self.eps)*x + self.eps*torch.ones_like(x)/(x.size(-1)-1)

# Randomly crop out a subset of a trace to simulate desynchronization and promote invariance to translation
class RandomCrop(Transform):
    def __init__(
        self,
        length_to_remove=50 # How many elements of the input vector to crop away from the sides
    ):
        super().__init__(length_to_remove=length_to_remove)
        
    def forward(self, x):
        start_idx = np.random.randint(self.length_to_remove)
        end_idx = x.size(1) + start_idx - self.length_to_remove
        return x[:, start_idx:end_idx]

# Add uniform random noise from bin width to reflect that we only know the true sample value up to measurement resolution
class SmoothBins(Transform):
    def __init__(
        self,
        bin_width # Minimum positive distance between elements of the input vector
    ):
        super().__init__(bin_width=bin_width)
        
    def forward(self, x):
        return x + self.bin_width*torch.rand_like(x)

# Add Gaussian noise to the samples while preserving mean and standard deviation, to reflect that there is some noise unrelated
#   to the AES key in the actual measurements
class AddGaussianNoise(Transform):
    def __init__(
        self,
        noise_stdev=0.1 # Standard deviation of the Gaussian noise to be added to the input vector
    ):
        super().__init__(noise_stdev=noise_stdev)
        
    def forward(self, x):
        # Add noise and rescale input so the standard deviation remains unchanged
        return self.noise_stdev*torch.randn_like(x) + np.sqrt(1-self.noise_stdev**2)*x

# Randomly replace some interval of the trace with random noise while preserving mean and standard deviation, to force the model to
#   consider as much of the trace as possible rather than relying only on a small subinterval
class RandomErasing(Transform):
    def __init__(
        self,
        erase_prob=0.25, # Probability with which to erase part of the input
        max_erased_prop=0.25 # Maximum proportion of the input which may be erased
    ):
        super().__init__(erase_prob=erase_prob, max_erased_prob=max_erased_prob)
        
    def forward(self, x):
        if np.random.rand() < self.erase_prob: # Execute this branch with probability self.erase_prob
            x = x.clone() # Convenient to avoid modifying the original input
            x_mn, x_std = torch.std_mean(x, dim=-1, keepdim=True)
            target_length = int(self.max_erased_prop*np.random.rand()*x.size(-1))
            start_sample = np.random.randint(x.size(-1)-target_length)
            x[:, start_sample:start_sample+target_length] = x_std*torch.randn(1, target_length, device=x.device) + x_mn
        return x
    
# Apply a low pass filter to the trace to reflect that as experimental conditions change, there may be different amounts of parasitic
#   capacitance and inductance in the measurement setup
class RandomLowPassFilter(Transform):
    def __init__(
        self,
        filter_prob=0.25, # Probability with which to low pass filter the input
        max_kernel_radius=5 # Maximum-size radius of the kernel; kernel width will be 1+2*kernel_radius
    ):
        super().__init__(filter_prob=filter_prob, max_kernel_radius=max_kernel_radius)
    
    def forward(self, x):
        if np.random.rand() < self.filter_prob:
            kernel_width = 2*np.random.randint(1, self.max_kernel_radius+1) + 1
            kernel = torch.ones(1, 1, kernel_width, device=x.device, dtype=x.dtype)/kernel_width
            return nn.functional.conv1d(x, kernel, padding=kernel_width//2)
        else:
            return x
    
# Apply high pass filter to reflect that as experimental conditions change there may be different amounts of parasitic capacitance
#   and inductance in the measurement setup
class RandomHighPassFilter(Transform):
    def __init__(
        self,
        filter_prob=0.25, # Probability with which to high pass filter the input
        max_kernel_radius=5 # Maximum-size radius of the kernel; kernel width will be 1+2*kernel_radius
    ):
        super().__init__(filter_prob=filter_prob, max_kernel_radius=max_kernel_radius)
        
        self.lpf = RandomLowPassFilter(filter_prob, max_kernel_radius)
        
    def forward(self, x):
        return x - self.lpf(x)
    
# Linearly-combine pairs of traces and labels, to promote the model linearly-interpolating its predictions as we linearly-interpolate
#   between traces. This has been shown to boost performance and robustness in computer vision models.
class Mixup(Transform):
    def __init__(
        self,
        mixup_prob=1.0, # Probability with which to apply mixup to a batch
        alpha=0.2 # Controls coefficients for the linear interpolation. With larger values, batches will deviate more from real
                  #   examples.
    ):
        super().__init__(mixup_prob=mixup_prob, alpha=alpha)
        
    def forward(self, batch):
        traces, labels = batch
        if np.random.rand() < self.mixup_prob:
            indices = torch.randperm(traces.size(0), device=traces.device, dtype=torch.long)
            lbd = np.random.beta(self.alpha, self.alpha)
            traces = lbd*traces + (1-lbd)*traces[indices, :]
            labels = lbd*labels + (1-lbd)*labels[indices]
        return (traces, labels)

# Combine different intervals of traces and reweight labels in proportion to the interval length of the trace it reflects. Similarly
#   to mixup, this has been shown to boost model performance and robustness in computer vision models.
class Cutmix(Transform):
    def __init__(
        self,
        cutmix_prob=0.5, # Probability with which to apply cutmix to a batch
        alpha=0.2 # Controls coefficients for the sizes of each region. Larger values mean it will tend to be closer to a 50/50
                  #  combination of two traces.
    ):
        super().__init__(cutmix_prob=cutmix_prob, alpha=alpha)
        
    def forward(self, x):
        traces, labels = batch
        if np.random.rand() < self.cutmix_prob:
            indices = torch.randperm(traces.size(0), device=self.device, dtype=torch.long)
            lbd = np.random.beta(self.alpha, self.alpha)
            cut_length = int(traces.size(-1)*lbd)
            start_idx = np.random.randint(traces.size(-1)-cut_length)
            end_idx = start_idx + cut_length
            traces[:, :, start_idx:end_idx] = traces[indices, :, start_idx:end_idx]
            labels = (1-lbd)*labels + lbd*labels[indices]
        return (traces, labels)