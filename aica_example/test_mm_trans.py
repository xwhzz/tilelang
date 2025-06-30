import ctypes
import torch
import argparse
from aicart import torch_randn_aica, torch_empty_aica, torch_dump_aica, release_all

parser = argparse.ArgumentParser(description="AICA Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k

compute_lib = ctypes.CDLL("./kernel_lib.so")

# --- 1. Setup Host Tensors ---
# Use float16 (half precision) as indicated by the kernel signature (half_t)
a, a_d = torch_randn_aica(M, K, dtype=torch.float16)
b, b_d = torch_randn_aica(K, N, dtype=torch.float16)
c, c_d = torch_empty_aica(M, N, dtype=torch.float16)

# --- 4. Launch Kernel on Device ---
# The kernel must be called with the DEVICE pointers (a_d, b_d, c_d).
print("Launching kernel...")
compute_lib.call(a_d, b_d, c_d)

# --- 5. Copy Result from Device to Host ---
# aicaMemcpyDeviceToHost has enum value 2.
# The source is the device pointer, destination is the host pointer.
print("Copying result from device to host...")
torch_dump_aica(c, c_d)

# --- 6. Free Device Memory ---
print("Freeing device memory...")
release_all()


# --- 7. Verify the Result ---
print("Verifying result...")
# PyTorch's matmul with float16 inputs might require casting for precision on some hardware.
ref_c = a @ b.T #torch.matmul(a.float(), b.float()).half()

# Print a small slice of the tensors to visually inspect
print("Reference C (slice):\n", ref_c[:16, :16])
print("\nAICA C (slice):\n", c[:16, :16])

# Use torch testing utility to check for correctness
torch.testing.assert_close(ref_c, c, rtol=1e-2, atol=1e-2)
print("\nTest passed!")