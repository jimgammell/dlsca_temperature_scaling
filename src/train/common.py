import numpy as np

from train.metrics import ResultsDict

def run_epoch(dataloader, step_fn, *step_args, average_batches=True, **step_kwargs):
    rv = ResultsDict()
    for bidx, batch in enumerate(dataloader):
        step_rv = step_fn(batch, *step_args, **step_kwargs)
        rv.update(step_rv)
    if average_batches:
        rv.reduce(np.mean)
    return rv