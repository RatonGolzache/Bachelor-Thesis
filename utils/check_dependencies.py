import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (PyTorch built with):", torch.version.cuda)

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

try:
    import apex
    print("APEX is installed.")
except ImportError:
    print("APEX is NOT installed.")

try:
    from apex import amp
    print("APEX AMP (mixed precision) is available.")
except ImportError:
    print("APEX AMP not available.")