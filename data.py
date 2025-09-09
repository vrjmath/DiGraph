import torch

def load_dataset(file_path):
    """
    Load a PyG dataset stored as a .pth file.
    """
    return torch.load(file_path, weights_only=False)  # full pickle loading
