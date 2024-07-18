import random
import numpy as np
import torch

def set_random_seed(seed, deterministic=False):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed value.
        deterministic (bool): If True, sets deterministic options for CuDNN backend (default: False).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Example usage:
if __name__ == "__main__":
    # Set a random seed (e.g., 42)
    set_random_seed(42)

    # Example code that involves random operations
    # Random number generation
    print("Random number from numpy:", np.random.rand())
    print("Random number from random module:", random.random())

    # Example PyTorch random operations
    tensor = torch.rand(3, 3)
    print("Random tensor from PyTorch:", tensor)













import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from mmcv.runner import load_checkpoint

def custom_multi_gpu_test(model, data_loader, tmpdir='./tmp', gpu_collect=False):
    """
    Test models with multiple GPUs.

    Args:
        model (nn.Module): The model to be tested.
        data_loader (DataLoader): The data loader for testing data.
        tmpdir (str): Directory for saving temporary results.
        gpu_collect (bool): Whether to use GPU to collect results.

    Returns:
        list: A list of evaluation results.
    """
    model = model.cuda()
    model = DDP(model, device_ids=[torch.cuda.current_device()])

    # Load checkpoint
    load_checkpoint(model, 'path_to_checkpoint.pth', map_location='cpu')

    results = []
    model.eval()

    if gpu_collect:
        model.forward = model.module.forward
        model.cuda()
    else:
        model.forward = model.module.forward

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(**data)  # Replace with actual input arguments
        results.append(result)

    return results

# Example usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from mmcv.parallel import collate

    # Example dataset and dataloader initialization
    # Replace with actual dataset and dataloader setup
    dataset = YourCustomDataset()
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate, shuffle=False)

    # Example model initialization (replace with your model)
    model = YourCustomModel()

    # Example multi-GPU testing
    results = custom_multi_gpu_test(model, data_loader, gpu_collect=True)

    # Print or process evaluation results
    print("Evaluation results:", results)
