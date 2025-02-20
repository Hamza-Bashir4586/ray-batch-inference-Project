import torch
from src.model import load_model

def test_model_load():
    """
    Ensures the model loads correctly for distributed inference.
    """
    model = load_model()
    assert model is not None, "Model failed to load"
    assert isinstance(model, torch.nn.Module), "Loaded model is not a valid PyTorch module"

def test_gpu_availability():
    """
    Ensures a GPU is available for high-speed batch processing.
    """
    assert torch.cuda.is_available(), "No GPU found!"
