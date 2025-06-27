import ctypes
import torch
import argparse

parser = argparse.ArgumentParser(description="AICA Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k

# Load the AICA runtime and the compiled kernel library
lib = ctypes.CDLL("/usr/local/aica/lib/libaicart.so")
compute_lib = ctypes.CDLL("./kernel_lib.so")

# --- 1. Setup Host Tensors ---
# Use float16 (half precision) as indicated by the kernel signature (half_t)
a = torch.randn(M, K, dtype=torch.float16)
b = torch.randn(K, N, dtype=torch.float16)
c = torch.empty(M, N, dtype=torch.float16)

# Get host memory pointers (as integers)
a_ptr = a.data_ptr()
b_ptr = b.data_ptr()
c_ptr = c.data_ptr()

# --- 2. Allocate Device Memory ---
# Create ctypes void pointers that will hold the device addresses.
# These will be populated by the aicaMalloc call.
a_d = ctypes.c_void_p()
b_d = ctypes.c_void_p()
c_d = ctypes.c_void_p()

# aicaMalloc expects a pointer to a pointer (void**) to write the address into.
# ctypes.byref(a_d) correctly creates this reference.
print("Allocating device memory...")
lib.aicaMalloc(ctypes.byref(a_d), a.numel() * a.element_size())
lib.aicaMalloc(ctypes.byref(b_d), b.numel() * b.element_size())
lib.aicaMalloc(ctypes.byref(c_d), c.numel() * c.element_size())

# --- 3. Copy Data from Host to Device ---
# aicaMemcpyHostToDevice has enum value 1.
# The source is the host pointer, destination is the device pointer.
print("Copying data from host to device...")
lib.aicaMemcpy(a_d, ctypes.c_void_p(a_ptr), a.numel() * a.element_size(), 1)
lib.aicaMemcpy(b_d, ctypes.c_void_p(b_ptr), b.numel() * b.element_size(), 1)

# --- 4. Launch Kernel on Device ---
# The kernel must be called with the DEVICE pointers (a_d, b_d, c_d).
print("Launching kernel...")
compute_lib.call(a_d, b_d, c_d)

# --- 5. Copy Result from Device to Host ---
# aicaMemcpyDeviceToHost has enum value 2.
# The source is the device pointer, destination is the host pointer.
print("Copying result from device to host...")
lib.aicaMemcpy(ctypes.c_void_p(c_ptr), c_d, c.numel() * c.element_size(), 2)

# --- 6. Free Device Memory ---
print("Freeing device memory...")
lib.aicaFree(a_d)
lib.aicaFree(b_d)
lib.aicaFree(c_d)

# --- 7. Verify the Result ---
print("Verifying result...")
# PyTorch's matmul with float16 inputs might require casting for precision on some hardware.
ref_c = torch.matmul(a.float(), b.float()).half()

# Print a small slice of the tensors to visually inspect
print("Reference C (slice):\n", ref_c[0, :16])
print("\nAICA C (slice):\n", c[0, :16])

# Use torch testing utility to check for correctness
# torch.testing.assert_close(ref_c, c, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(ref_c[:,0], c[:,0], rtol=1e-2, atol=1e-2)
torch.testing.assert_close(ref_c, c, rtol=1e-2, atol=1e-2)
print("\nTest passed!")