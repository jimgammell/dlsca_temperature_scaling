# Code for accumulating predictions over test traces.
# Adapted from https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/

import numpy as np
from scipy.special import softmax, log_softmax
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from datasets import AES_SBOX

@torch.no_grad()
def eval_data(model, dataloader, device):
    model.eval()
    full_logits, full_plaintexts, full_keys = [], [], []
    return_metadata = dataloader.dataset.return_metadata
    dataloader.dataset.return_metadata = True
    for batch in tqdm(dataloader):
        traces, _, metadata = batch
        traces = traces.to(device)
        logits = model(traces)
        logits = logits.cpu().numpy()
        plaintexts = metadata['plaintext']
        keys = metadata['key']
        full_logits.append(logits)
        full_plaintexts.append(plaintexts)
        full_keys.append(keys)
    dataloader.dataset.return_metadata = return_metadata
    full_logits, full_plaintexts, full_keys = np.concatenate(full_logits), np.concatenate(full_plaintexts), np.concatenate(full_keys)
    return full_logits, full_plaintexts, full_keys

def rank_key(ranked_array, key):
    key_val = ranked_array[key]
    return np.where(np.sort(ranked_array)[::-1] == key_val)[0][0]

def rank_compute(logits, plaintext, key, byte):
    (n_traces, n_candidates) = logits.shape
    key_log_prob = np.zeros(n_candidates)
    rank_evolution = np.full(n_traces, 255)
    prediction = np.log(softmax(logits)+1e-40) 
    #prediction = log_softmax(logits) # more-accurate than log(softmax(.))
    traces_to_disclosure = 0
    for trace_idx in range(n_traces):
        for candidate_idx in range(n_candidates):
            key_log_prob[candidate_idx] += prediction[trace_idx, AES_SBOX[candidate_idx ^ plaintext[trace_idx, byte]]]
        rank_evolution[trace_idx] = rank_key(key_log_prob, key[trace_idx, byte])
        if (rank_evolution[trace_idx] > 0) or np.isnan(rank_evolution[trace_idx]):
            traces_to_disclosure = trace_idx+1
    return rank_evolution, traces_to_disclosure

def perform_attacks(n_attacks, traces_per_attack, logits, plaintexts, keys, byte):
    assert len(logits) == len(plaintexts) >= traces_per_attack
    all_rank_evolutions = np.zeros((n_attacks, traces_per_attack))
    all_traces_to_disclosure = np.zeros((n_attacks,))
    for attack_idx in tqdm(range(n_attacks)):
        shuffled_indices = np.arange(len(logits))
        rng = np.random.RandomState(attack_idx)
        rng.shuffle(shuffled_indices)
        shuffled_indices = shuffled_indices[:traces_per_attack]
        attack_logits, attack_plaintexts, key = logits[shuffled_indices], plaintexts[shuffled_indices], keys[shuffled_indices]
        rank_evolution, traces_to_disclosure = rank_compute(attack_logits, attack_plaintexts, key, byte)
        all_rank_evolutions[attack_idx] = rank_evolution
        all_traces_to_disclosure[attack_idx] = traces_to_disclosure
    return all_rank_evolutions, all_traces_to_disclosure

def plot_attacks(uncalibrated_evolutions, calibrated_evolutions, bad_calibration_evolutions, title='', save_path=None, fig=None, ax=None):
    if fig is None:
        assert ax is None
        fig, ax = plt.subplots()
    else:
        assert ax is not None
    traces = np.arange(1, uncalibrated_evolutions.shape[1]+1)
    def plot(evolutions, color, label):
        mn, std = np.mean(evolutions, axis=0), np.std(evolutions, axis=0)
        ax.plot(traces, mn, color=color, linestyle='-', label=label)
        ax.plot(traces, mn+std, color=color, linestyle='--')
        ax.plot(traces, mn-std, color=color, linestyle='--')
        ax.fill_between(traces, np.min(evolutions, axis=0), np.max(evolutions, axis=0), color=color, alpha=0.5)
    plot(uncalibrated_evolutions, 'blue', 'uncalibrated')
    plot(calibrated_evolutions, 'red', 'calibrated')
    plot(bad_calibration_evolutions, 'green', 'excessively_confident')
    ax.legend()
    ax.grid(True)
    ax.set_xscale('log')
    ax.set_xlabel('Traces observed')
    ax.set_ylabel('Rank of true key')
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)