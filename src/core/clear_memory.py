import torch
import gc


def clear_memory():
    # Clear PyTorch cache
    torch.cuda.empty_cache()

    # Delete all PyTorch tensors and models
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            del obj

    # Run garbage collector
    gc.collect()