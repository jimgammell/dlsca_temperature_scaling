import numpy as np
import torch
from torch import nn

class ResultsDict(dict):
    def append(self, key, value):
        if not key in self.keys():
            self[key] = np.array([])
        self[key] = np.append(self[key], value)
    
    def update(self, new_dict):
        for key, value in new_dict.items():
            self.append(key, value)
    
    def reduce(self, reduce_fn):
        for key, value in self.items():
            if isinstance(reduce_fn, dict):
                self[key] = {rfkey: rf(value) for rfkey, rf in reduce_fn.items()}
            elif isinstance(reduce_fn, list):
                self[key] = [rf(value) for rf in reduce_fn]
            else:
                self[key] = reduce_fn(value)
    
    def data(self):
        return {key: val for key, val in self.items()}
    
def get_acc(logits, labels):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if labels.shape == logits.shape:
        labels = np.argmax(labels, axis=-1)
    else:
        assert labels.shape[0] == logits.shape[0]
    predictions = np.argmax(logits, axis=-1)
    correct = np.equal(predictions, labels)
    accuracy = np.mean(correct)
    return accuracy

def get_cosine_similarity(logits, labels):
    if not isinstance(logits, torch.tensor):
        logits = torch.tensor(logits)
    if not isinstance(labels, torch.tensor):
        labels = torch.tensor(labels)
    if labels.shape != logits.shape:
        assert labels.shape[0] == logits.shape[0]
        labels = nn.functional.one_hot(labels, num_classes=256).to(torch.float)
    logits, labels = logits.detach(), labels.detach()
    pred_dist = nn.functional.softmax(logits, dim=-1)
    cos_sim = nn.functional.cosine_similarity(pred_dist, labels)
    cos_sim = cos_sim.numpy()
    return cos_sim

def get_rank(logits, labels):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if labels.shape == logits.shape:
        labels = np.argmax(labels, axis=-1)
    else:
        assert labels.shape[0] == logits.shape[0]
    rank = (-logits).argsort(axis=-1).argsort(axis=-1)
    correct_rank = rank[np.arange(len(rank)), labels].mean()
    return correct_rank

def get_norms(model):
    total_weight_norm, total_grad_norm = 0.0, 0.0
    for param in model.parameters():
        weight_norm = param.detach().data.norm(2).item()
        total_weight_norm += weight_norm**2
        if param.requires_grad and (param.grad is not None):
            grad_norm = param.grad.detach().data.norm(2).item()
            total_grad_norm += grad_norm**2
    total_weight_norm = np.sqrt(total_weight_norm)
    total_grad_norm = np.sqrt(total_grad_norm)
    return {'weight_norm': total_weight_norm, 'grad_norm': total_grad_norm}
