import ctypes
import torch
import argparse
import torch.nn.functional as F
from aicart import torch_randn_aica, torch_empty_aica, torch_dump_aica, release_all

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

compute_lib = ctypes.CDLL("./kernel_lib.so")
shape = (batch, seq_len, heads, dim)
# --- 1. Setup Host Tensors ---
# Use float16 (half precision) as indicated by the kernel signature (half_t)
q, q_d = torch_randn_aica(shape, dtype=torch.float16)
k, k_d = torch_randn_aica(shape, dtype=torch.float16)
v, v_d = torch_randn_aica(shape, dtype=torch.float16)
out, out_d = torch_empty_aica(shape, dtype=torch.float16)


# --- 4. Launch Kernel on Device ---
# The kernel must be called with the DEVICE pointers (a_d, b_d, c_d).
print("Launching kernel...")
compute_lib.call(q_d, k_d, v_d, out_d)

# --- 5. Copy Result from Device to Host ---
# aicaMemcpyDeviceToHost has enum value 2.
# The source is the device pointer, destination is the host pointer.
print("Copying result from device to host...")
torch_dump_aica(out, out_d)

# --- 6. Free Device Memory ---
print("Freeing device memory...")
release_all()
# --- 7. Verify the Result ---
print("Verifying result...")

scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
attention_weights = F.softmax(scores, dim=-1)
output_ref = torch.einsum('bhqk,bkhd->bqhd', attention_weights, v)

print(output_ref[0,0,:8, :8], "\n" ,out[0,0,:8, :8])

torch.testing.assert_close(output_ref, out, rtol=1e-2, atol=1e-2)
print("\nTest passed!")