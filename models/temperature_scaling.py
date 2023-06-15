# Adapted from https://github.com/gpleiss/temperature_scaling/

import numpy as np
import torch
from torch import nn, optim

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1)*1.5)
        self.name = self.model.name
    
    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        temperature = nn.functional.softplus(temperature)
        return logits / temperature
    
    def forward(self, x, dont_rescale_temperature=False):
        logits = self.model(x)
        if not(self.model.training) and not(dont_rescale_temperature):
            logits = self.temperature_scale(logits)
        return logits

class ECELoss(nn.Module): # expected calibration error
    def __init__(self, n_bins=15):
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins+1)
        self.bins_lower = bin_boundaries[:-1]
        self.bins_upper = bin_boundaries[1:]
        
    def forward(self, logits, labels):
        distributions = nn.functional.softmax(logits, dim=-1)
        confidences, predictions = torch.max(distributions, 1)
        accuracies = predictions.eq(labels)
        
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bins_lower, self.bins_upper):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
    
    def __repr__(self):
        return self.__class__.__name__+'()'
    
def calibrate_temperature(model, dataloader, device):
    nll_criterion = nn.CrossEntropyLoss().to(device)
    ece_criterion = ECELoss().to(device)
    model.eval()
    
    logits_list, labels_list = [], []
    with torch.no_grad():
        for trace, label in dataloader:
            trace = trace.to(device)
            logits = model(trace, dont_rescale_temperature=True)
            logits_list.append(logits)
            labels_list.append(label)
        logits, labels = torch.cat(logits_list).to(device), torch.cat(labels_list).to(device)
    
    pre_nll = nll_criterion(logits, labels).item()
    pre_ece = ece_criterion(logits, labels).item()
    
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(model.temperature_scale(logits), labels)
        loss.backward()
        return loss
    temps = [model.temperature.item()]
    optimizer.step(closure)
    temps.append(model.temperature.item())
    while np.abs(temps[-1] - temps[-2]) > 1e-5:
        optimizer.step(closure)
        temps.append(model.temperature.item())
    
    post_nll = nll_criterion(model.temperature_scale(logits), labels).item()
    post_ece = ece_criterion(model.temperature_scale(logits), labels).item()
    
    return {'final_temperature': model.temperature.item(), 'pre_nll': pre_nll, 'post_nll': post_nll, 'pre_ece': pre_ece, 'post_ece': post_ece}
    