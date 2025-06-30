import ctypes
import torch

_lib = ctypes.CDLL("/usr/local/aica/lib/libaicart.so")

_device_tensor = []

def torch_randn_aica(*args, **kwargs) -> tuple[torch.Tensor, ctypes.c_void_p]:
    """
    return original_tensor and d_ptr
    """
    tensor = torch.randn(*args, **kwargs)

    tensor_d = ctypes.c_void_p()
    _lib.aicaMalloc(ctypes.byref(tensor_d), tensor.numel() * tensor.element_size())
    _lib.aicaMemcpy(ctypes.c_void_p(tensor_d.value), ctypes.c_void_p(tensor.data_ptr()), tensor.numel() * tensor.element_size(), 1)

    _device_tensor.append(tensor_d)

    return tensor, tensor_d


def torch_empty_aica(*args, **kwargs) -> tuple[torch.Tensor, ctypes.c_void_p]:
    """
    return original_tensor and d_ptr
    """
    tensor = torch.empty(*args, **kwargs)

    tensor_d = ctypes.c_void_p()
    _lib.aicaMalloc(ctypes.byref(tensor_d), tensor.numel() * tensor.element_size())

    _device_tensor.append(tensor_d)
    return tensor, tensor_d

def torch_dump_aica(tensor: torch.Tensor, d_ptr: ctypes.c_void_p) -> None:
    """
    Copy tensor from device to host and free device memory.
    """
    _lib.aicaMemcpy(ctypes.c_void_p(tensor.data_ptr()), d_ptr, tensor.numel() * tensor.element_size(), 2)


def release_all() -> None:
    """
    Free all device memory allocated by torch_randn_aica and torch_empty_aica.
    """
    for d_ptr in _device_tensor:
        _lib.aicaFree(d_ptr)
    _device_tensor.clear()