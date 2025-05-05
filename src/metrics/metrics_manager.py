import sys

_perplexity = None
_execution_time_ms = None
_average_energy_per_token_mj = None
_average_time_per_token_ms = None
_total_energy = None
_allocated_memory = None
_layer_idx = None
_head_idxs = None
_temperature = None
_saved_metrics = dict()
header = (
    'label',
    'layer_idx',
    'head_idxs',
    'perplexity',
    'average_energy_per_token_mj',
    'average_time_per_token_ms',
    'allocated_memory',
    'temperature'
)
    
def clear():
    _perplexity = None
    _execution_time_ms = None
    _average_energy_per_token_mj = None
    _average_time_per_token_ms = None
    _total_energy = None
    _allocated_memory = None
    _layer_idx = None
    _head_idxs = None

def clear_saved():
    global _saved_metrics
    _saved_metrics = dict()

def perplexity(perplexity):
    global _perplexity
    _perplexity = perplexity
    return sys.modules[__name__]

def execution_time_ms(execution_time_ms):
    global _execution_time_ms
    _execution_time_ms = int(execution_time_ms)
    return sys.modules[__name__]

def average_energy_per_token_mj(average_energy_per_token_mj):
    global _average_energy_per_token_mj
    _average_energy_per_token_mj = average_energy_per_token_mj
    return sys.modules[__name__]

def average_time_per_token_ms(average_time_per_token_ms):
    global _average_time_per_token_ms
    _average_time_per_token_ms = average_time_per_token_ms
    return sys.modules[__name__]

def total_energy(total_energy):
    global _total_energy
    _total_energy = total_energy
    return sys.modules[__name__]

def allocated_memory(allocated_memory):
    global _allocated_memory
    _allocated_memory = allocated_memory
    return sys.modules[__name__]

def temperature(temperature):
    global _temperature
    _temperature = temperature
    return sys.modules[__name__]

def layer_idx(layer_idx):
    global _layer_idx
    _layer_idx = layer_idx
    return sys.modules[__name__]

def head_idxs(head_idxs):
    global _head_idxs
    _head_idxs = head_idxs
    return sys.modules[__name__]

def save_metrics(label):
    _saved_metrics[label] = (
        label, _layer_idx, _head_idxs, _perplexity, _average_energy_per_token_mj,
        _average_time_per_token_ms, _allocated_memory, _temperature)
    return sys.modules[__name__]

def get_metrics():
    return [header] + list(_saved_metrics.values())
