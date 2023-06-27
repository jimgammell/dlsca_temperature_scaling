import os
import time
import pickle
import h5py
import numpy as np
from scipy.special import softmax, log_softmax
from tqdm import tqdm
from torch.utils.data import IterableDataset

import resources
from datasets.common import DatasetBase

SBREV = np.array([
    82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251,
    124, 227, 57, 130, 155, 47, 255, 135, 52, 142, 67, 68, 196, 222, 233, 203,
    84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78, 8,
    46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114,
    248, 246, 100, 134, 104, 152, 22, 212, 164, 92, 204, 93, 101, 182, 146, 108,
    112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157, 132, 144,
    216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44,
    30, 143, 202, 63, 15, 2, 193, 175, 189, 3, 1, 19, 138, 107, 58, 145, 17, 65,
    79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115, 150, 172, 116,
    34, 231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26,
    113, 29, 41, 197, 137, 111, 183, 98, 14, 170, 24, 190, 27, 252, 86, 62, 75,
    198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244, 31, 221, 168, 51,
    136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25,
    181, 74, 13, 45, 229, 122, 159, 147, 201, 156, 239, 160, 224, 59, 77, 174,
    42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119,
    214, 38, 225, 105, 20, 99, 85, 33, 12, 125
])

def logits_to_key_pred(logits, plaintexts, attack_point):
    if logits.ndim == 2:
        predictions = np.mean(softmax(logits, axis=-1), axis=0) 
    elif logits.ndim == 1:
        predictions = softmax(logits, axis=-1)
    else:
        assert False
    if attack_point == 'keys':
        return predictions
    elif attack_point == 'sub_bytes_in':
        class_ids = np.arange(256)
        kv = class_ids ^ plaintexts
        key_probas = np.take(predictions, kv)
        return key_probas
    elif attack_point == 'sub_bytes_out':
        kidxs = SBREV ^ plaintexts
        probas = np.zeros(256)
        for pidx, kidx in enumerate(kidxs):
            probas[kidx] = predictions[pidx]
        return probas
    else:
        raise NotImplementedError('{}={}'.format('attack_point', attack_point))

class GoogleTinyAES(DatasetBase):
    def __init__(
        self,
        train=True,
        resource_path=None,
        target_byte=0,
        interval_to_use=[0, 20000],
        target_attack_point='sub_bytes_in',
        **kwargs
    ):
        self.interval_to_use = interval_to_use
        self.target_byte = target_byte
        self.target_attack_point = target_attack_point
        
        if resource_path is None:
            from resources import google_scaaml
            resource_path = google_scaaml.get_dataset_path(train)
        self.resource_path = resource_path
        database_file = h5py.File(os.path.join(resource_path, 'data.hdf5'))
        self.data = database_file['traces']
        self.targets = database_file['{}__{}'.format(target_attack_point, target_byte)]
        self.metadata = {key: val for key, val in database_file.items() if key != 'traces'}
        self.length = len(self.data)
        assert self.length == len(self.targets)
        assert all(self.length == len(val) for val in self.metadata.values())
        self.name = 'Google_TinyAES'
        self.data_mean = -0.2751
        self.data_stdev = 0.1296
        
        super().__init__(train=train, **kwargs)
    
    def load_idx(self, idx):
        trace = self.data[idx, :, self.interval_to_use[0]:self.interval_to_use[1]]
        target = self.targets[idx]
        if hasattr(self, 'return_metadata') and self.return_metadata:
            metadata = {key: val[idx] for key, val in self.metadata.items()}
            return trace, target, metadata
        else:
            return trace, target
        
    def eval_attack_efficacy(self, logits, metadata, n_attacks=10):
        logits_by_key = {
            key: [(x, {k: v[idx] for k, v in metadata.items()})
                   for idx, x in enumerate(logits) if metadata['{}__{}'.format('keys', self.target_byte)][idx] == key
                 ] for key in np.arange(256)
        }
        logits_by_key = {
            key: val[:256] for key, val in logits_by_key.items() if len(val) > 0
        }
        ttd_vals = []
        traces = []
        for key, data in logits_by_key.items():
            for _ in range(n_attacks):
                traces.append([])
                pred = np.zeros((len(data),))
                ttd = 1
                indices = np.arange(len(data))
                np.random.shuffle(indices)
                for trace_idx, idx in enumerate(indices):
                    logits_, metadata_ = data[idx]
                    trace_pred = logits_to_key_pred(
                        logits_, metadata_['{}__{}'.format('pts', self.target_byte)], self.target_attack_point
                    )
                    pred += np.log(trace_pred + 1e-22)
                    rank = (-pred).argsort().argsort()
                    correct_rank = rank[key]
                    traces[-1].append(correct_rank)
                    if np.argmax(pred) != key:
                        ttd = trace_idx + 2
                ttd_vals.append(ttd)
        return np.mean(ttd_vals), np.array(traces)

AVAILABLE_DATASETS = [GoogleTinyAES]