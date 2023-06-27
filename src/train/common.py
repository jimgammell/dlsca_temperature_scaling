import numpy as np
from tqdm import tqdm

from train.metrics import ResultsDict

def run_epoch(dataloader, step_fn, *step_args, average_batches=True, use_progress_bar=False, **step_kwargs):
    rv = ResultsDict()
    for bidx, batch in enumerate(tqdm(dataloader) if use_progress_bar else dataloader):
        step_rv = step_fn(batch, *step_args, **step_kwargs)
        rv.update(step_rv)
    if average_batches:
        rv.reduce(np.mean)
    return rv