import torch

def torch_gc():
    if torch.cuda.is_available():
        # with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
