import ctypes
import argparse
import torch
from aicart import torch_randn_aica, torch_empty_aica, torch_dump_aica, release_all

parser = argparse.ArgumentParser(description="RMS Norm Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n

compute_lib = ctypes.CDLL("./kernel_lib.so")

# --- 1. Setup Host Tensors ---
# Use float16 (half precision) as indicated by the kernel signature (half_t)
a, a_d = torch_randn_aica(M, N, dtype=torch.float)
b, b_d = torch_empty_aica(M, N, dtype=torch.float)

# --- 4. Launch Kernel on Device ---
# The kernel must be called with the DEVICE pointers (a_d, b_d, c_d).
print("Launching kernel...")
compute_lib.call(a_d, b_d)

# --- 5. Copy Result from Device to Host ---
# aicaMemcpyDeviceToHost has enum value 2.
# The source is the device pointer, destination is the host pointer.
print("Copying result from device to host...")
torch_dump_aica(b, b_d)

# --- 6. Free Device Memory ---
release_all()

# --- 7. Verify the Result ---
print("Verifying result...")
# PyTorch's matmul with float16 inputs might require casting for precision on some hardware.
ref_b = a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + 1e-12)

# Print a small slice of the tensors to visually inspect
print("Reference C (slice):\n", ref_b[0, :16])
print("\nAICA C (slice):\n", b[0, :16])

# Use torch testing utility to check for correctness
# torch.testing.assert_close(ref_c, c, rtol=1e-2, atol=1e-2)
# torch.testing.assert_close(ref_c[:,0], c[:,0], rtol=1e-2, atol=1e-2)
torch.testing.assert_close(ref_b, b, rtol=1e-2, atol=1e-2)
print("\nTest passed!")