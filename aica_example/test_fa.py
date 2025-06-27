import ctypes
import torch
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="AICA Kernel Compilation")
parser.add_argument('--batch', type=int, default=8, help='batch size')
parser.add_argument('--heads', type=int, default=32, help='heads')
parser.add_argument('--seq_len', type=int, default=128, help='sequence length')
parser.add_argument('--dim', type=int, default=64, help='dim')
args = parser.parse_args()

batch = args.batch
heads = args.heads
seq_len = args.seq_len
dim = args.dim

# Load the AICA runtime and the compiled kernel library
lib = ctypes.CDLL("/usr/local/aica/lib/libaicart.so")
compute_lib = ctypes.CDLL("./kernel_lib.so")
shape = (batch, seq_len, heads, dim)
# --- 1. Setup Host Tensors ---
# Use float16 (half precision) as indicated by the kernel signature (half_t)
q = torch.randn(shape, dtype=torch.float16)
k = torch.randn(shape, dtype=torch.float16)
v = torch.randn(shape, dtype=torch.float16)

out = torch.empty(shape, dtype=torch.float16)

# Get host memory pointers (as integers)
q_ptr = q.data_ptr()
k_ptr = k.data_ptr()
v_ptr = v.data_ptr()
out_ptr = out.data_ptr()


# --- 2. Allocate Device Memory ---
# Create ctypes void pointers that will hold the device addresses.
# These will be populated by the aicaMalloc call.
q_d = ctypes.c_void_p()
k_d = ctypes.c_void_p()
v_d = ctypes.c_void_p()
out_d = ctypes.c_void_p()

# aicaMalloc expects a pointer to a pointer (void**) to write the address into.
# ctypes.byref(a_d) correctly creates this reference.
print("Allocating device memory...")
lib.aicaMalloc(ctypes.byref(q_d), q.numel() * q.element_size())
lib.aicaMalloc(ctypes.byref(k_d), k.numel() * k.element_size())
lib.aicaMalloc(ctypes.byref(v_d), v.numel() * v.element_size())
lib.aicaMalloc(ctypes.byref(out_d), out.numel() * out.element_size())

# --- 3. Copy Data from Host to Device ---
# aicaMemcpyHostToDevice has enum value 1.
# The source is the host pointer, destination is the device pointer.
print("Copying data from host to device...")
lib.aicaMemcpy(q_d, ctypes.c_void_p(q_ptr), q.numel() * q.element_size(), 1)
lib.aicaMemcpy(k_d, ctypes.c_void_p(k_ptr), k.numel() * k.element_size(), 1)
lib.aicaMemcpy(v_d, ctypes.c_void_p(v_ptr), v.numel() * v.element_size(), 1)

# --- 4. Launch Kernel on Device ---
# The kernel must be called with the DEVICE pointers (a_d, b_d, c_d).
print("Launching kernel...")
compute_lib.call(q_d, k_d, v_d, out_d)

# --- 5. Copy Result from Device to Host ---
# aicaMemcpyDeviceToHost has enum value 2.
# The source is the device pointer, destination is the host pointer.
print("Copying result from device to host...")
lib.aicaMemcpy(ctypes.c_void_p(out_ptr), out_d, out.numel() * out.element_size(), 2)

# --- 6. Free Device Memory ---
print("Freeing device memory...")
lib.aicaFree(q_d)
lib.aicaFree(k_d)
lib.aicaFree(v_d)
lib.aicaFree(out_d)
# --- 7. Verify the Result ---
print("Verifying result...")

scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
attention_weights = F.softmax(scores, dim=-1)
output_ref = torch.einsum('bhqk,bkhd->bqhd', attention_weights, v)

print(output_ref[0,0,:8, :8], "\n" ,out[0,0,:8, :8])

torch.testing.assert_close(output_ref, out, rtol=1e-2, atol=1e-2)
print("\nTest passed!")