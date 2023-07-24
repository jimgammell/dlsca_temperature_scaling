import numpy as np
import torch
from torch import nn

class TemplateAttacker(nn.Module):
    def __init__(self, train_dataset, scores, samples_to_use=5, exclude_radius=0, num_classes=256):
        super().__init__()
        
        assert samples_to_use > 0
        
        indices_to_use = np.empty((samples_to_use,), dtype=int)
        for idx in range(samples_to_use):
            new_idx = scores.argsort()[-1]
            indices_to_use[idx] = new_idx
            scores[max(0, new_idx-exclude_radius) : min(new_idx+exclude_radius+1, len(scores))] = -np.inf
        self.indices_to_use = indices_to_use
        data = np.empty((len(train_dataset), samples_to_use), dtype=float)
        labels = np.empty((len(train_dataset),), dtype=int)
        for idx, (trace, label) in enumerate(train_dataset):
            data[idx] = trace.squeeze().numpy()[indices_to_use]
            labels[idx] = label.item()
        
        means = np.empty((num_classes, samples_to_use), dtype=float)
        covariances = np.empty((num_classes, samples_to_use, samples_to_use), dtype=float)
        for label in range(num_classes):
            x = data[labels==label]
            means[label, :] = np.mean(x, axis=0)
            x_centered = x - means[label, :]
            covariances[label, :, :] = x_centered.T @ x_centered / x_centered.shape[0]
        
        self.distributions = [
            torch.distributions.multivariate_normal.MultivariateNormal(
                torch.tensor(mean, dtype=torch.float), covariance_matrix=torch.tensor(covariance, dtype=torch.float)
            ) for mean, covariance in zip(means, covariances)
        ]
        
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for distribution in self.distributions:
            distribution.loc = distribution.loc.to(*args, **kwargs)
            distribution._unbroadcasted_scale_tril = distribution._unbroadcasted_scale_tril.to(*args, **kwargs)
        
    def forward(self, x):
        x_poi = x[:, :, self.indices_to_use]
        log_probs = torch.cat([
            dist.log_prob(x_poi) for dist in self.distributions
        ], dim=-1)
        return log_probs