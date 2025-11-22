import torch


def get_device(device_config: str = 'auto') -> torch.device:
    """Get the appropriate PyTorch device.

    Args:
        device_config: Device configuration string ('auto', 'cuda', or 'cpu').

    Returns:
        PyTorch device object.
    """
    if device_config == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_config)
