import torch

def get_device(force_cpu: bool= False):
    """ Automatically detects and initializes the best hardware backend available. """
    
    if force_cpu: return torch.device("cpu")

    # 1. Check for NVIDIA CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # 2. Check for AMD / Intel via DirectML
    try:
        import torch_directml
        # DirectML returns its own custom device context object
        dml_device = torch_directml.device()
        return dml_device
    except ImportError:
        pass
        
    # 3. Fallback to CPU
    return torch.device("cpu")
