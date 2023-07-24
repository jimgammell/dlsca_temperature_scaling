from tqdm import tqdm
import numpy as np
import torch

def get_trace_offset(x, x_ref, max_offset=100):
    x, x_ref = x[21:], x_ref[:-21]
    x_corr = np.correlate(x[max_offset:-max_offset], x_ref, mode='valid')
    offset = np.argmax(x_corr) - max_offset
    return offset, x_corr

def sum_of_differences(dataset):
    progress_bar = tqdm(total=len(dataset) + 256**2 - 256)
    
    # Compute the average power trace for each possible key value
    means_per_key = {}
    for trace, key in dataset:
        trace = trace.squeeze()
        if isinstance(trace, torch.Tensor):
            trace = trace.cpu().numpy()
        if isinstance(key, torch.Tensor):
            key = key.item()
        if not key in means_per_key.keys():
            means_per_key[key] = [0, np.zeros_like(trace)]
        mn_samples, mn_trace = means_per_key[key]
        means_per_key[key][1] = (mn_samples/(mn_samples+1))*mn_trace + (1/(mn_samples+1))*trace
        means_per_key[key][0] += 1
        progress_bar.update(1)
    
    # Compute the sum of absolute pairwise differences between average trace for different operations
    sum_diffs = np.zeros_like(trace)
    for key1 in means_per_key.keys():
        for key2 in means_per_key.keys():
            if key1 == key2:
                continue
            #sum_diffs += np.abs(means_per_key[key1][1] - means_per_key[key2][1])
            sum_diffs = np.maximum(np.abs(means_per_key[key1][1] - means_per_key[key2][1]), sum_diffs)
            progress_bar.update(1)
    
    return sum_diffs