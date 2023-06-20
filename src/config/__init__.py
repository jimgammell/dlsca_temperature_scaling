import os
import json
import copy

NUM_AGENTS = 1
CONFIG_DIR = os.path.join('..', 'config_files')
TRAIN_CONFIGS = os.path.join(CONFIG_DIR, 'train')
HTUNE_CONFIGS = os.path.join(CONFIG_DIR, 'htune')
RESULTS_BASE_DIR = os.path.join('..', 'results')
LOG_FILES = {}

def specify_log_file(path):
    LOG_FILES[os.getpid()] = path
    with open(path, 'w') as _:
        pass

def printl(*args, **kwargs):
    assert not('file' in kwargs.keys())
    if not os.getpid() in LOG_FILES.keys():
        print(*args, **kwargs)
    else:
        with open(LOG_FILES[os.getpid()], 'a') as F:
            print(*args, file=F, **kwargs)

# Specify the number of trials to run in parallel on this machine.
def set_num_agents(val):
    global NUM_AGENTS
    NUM_AGENTS = val

# Retrieve the number of trials to run in parallel on this machine.
def get_num_agents():
    return NUM_AGENTS

# List the configuration files found in the config directory. Directory depends on whether the config files specify a single run or
#   a WandB hyperparameter sweep.
def get_available_configs(train=True):
    base_dir = TRAIN_CONFIGS if train else HTUNE_CONFIGS
    return [x.split('.')[0] for x in os.listdir(base_dir) if x.split('.')[-1] == 'json']

# Load the dictionary encoded by the specified configuration file.
def load_config(config, train=True):
    available_configs = get_available_configs(train=train)
    if not config in available_configs:
        raise Exception('Invalid config argument. Valid options: [\n{}\n]'.format(',\n\t'.join(available_configs)))
    if config.split('.')[-1] != 'json':
        config = config + '.json'
    with open(os.path.join(CONFIG_DIR, TRAIN_CONFIGS if train else HTUNE_CONFIGS, config), 'r') as F:
        settings = json.load(F)
    return settings

# Remove all nested dictionaries and concatenate nested keys, i.e. {key1: {key2: val}} becomes {key1-key2: val}.
#   Necessary because the WandB hyperparameter tuners can't handle nested dictionaries.
def denest_dict(d, delim='-'):
    if any(delim in key for k in d.keys()):
        raise Exception('Delimiter character is used in one or more keys: \'{}\''.format(
            delim, '\', \''.join(list(d.keys()))
        ))
    for key, val in copy.deepcopy(d).items():
        if isinstance(val, dict) and any(isinstance(subval, dict) for subval in val.values()):
            for subkey, subval in val.items():
                d[delim.join((key, subkey))] = subval
            del d[key]
    return d

# Nest a dictionary that has previously been de-nested by the above function.
def nest_dict(d, delim='-'):
    while any(delim in key for key in d.keys()):
        for key, val in copy.deepcopy(d).items():
            if delim in key:
                outer_key, inner_key = key.split(delim, maxsplit=1)
                if not outer_key in d.keys():
                    d[outer_key] = {}
                d[outer_key][inner_key] = val
                del d[key]
    return d