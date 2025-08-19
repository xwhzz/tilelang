import torch


def torch_convert_bit_twiddling(tensor):

    def _convert(val0, val1, pos) -> torch.bfloat16:
        assert val0.dtype == torch.uint8
        assert val1.dtype == torch.uint8
        val0 = val0.view(torch.uint8)
        val1 = val1.view(torch.uint8)
        val_concat = (val0.item() << 8) | val1.item()
        mask = 0b1000000111000000
        if pos == 0:
            bf16 = val_concat & mask
        elif pos == 1:
            bf16 = (val_concat << 3) & mask
        elif pos == 2:
            bf16 = (val_concat << 6) & mask
        elif pos == 3:
            mask1 = 0b1000000000000000
            mask2 = 0b0000000110000000
            mask3 = 0b0000000001000000
            bf16 = ((val_concat << 1) & mask1) | ((val_concat >> 3) & mask2) | (
                (val_concat >> 7) & mask3)
        bf16_new = torch.tensor([bf16], dtype=torch.uint16, device=val0.device).view(torch.bfloat16)
        # Add bias for change from fp4 to bf16
        bf16_new = bf16_new.item() * (2**126)
        return bf16_new

    N = tensor.shape[0]
    K = tensor.shape[1]
    new_tensor = torch.empty(N, K * 2, dtype=torch.bfloat16, device=tensor.device)
    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            new_tensor[i][j] = _convert(tensor[i][j // 4 * 2], tensor[i][j // 4 * 2 + 1], j % 4)
    return new_tensor


def torch_convert(tensor, scale_size=None, Scale=None):

    def _convert(val, pos, scale=None):
        assert val.dtype == torch.uint8
        # val = val.view(torch.int8)
        mask = (1 << 4) - 1
        f4 = ((val >> (pos * 4)) & mask).to(torch.int16)
        s = f4 >> 3
        e_f4 = (f4 & 6) >> 1
        e_f16 = e_f4 + 126
        if scale is not None:
            e_f16 = min(e_f16 + scale, (1 << 8) - 1)
        m_f4 = f4 & 1
        m_f16 = m_f4
        val_f16 = (((e_f16 | (s << 8)) << 7) | (m_f16 << 6)) & 0xFFFF
        lower_16_bits = (val_f16 & 0xFFFF).to(torch.uint16)
        return lower_16_bits.view(torch.bfloat16)

    N = tensor.shape[0]
    K = tensor.shape[1]
    new_tensor = torch.empty(N, K * 2, dtype=torch.bfloat16, device=tensor.device)
    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            if scale_size is not None:
                new_tensor[i][j] = _convert(tensor[i][j // 2], j % 2, Scale[i][j // scale_size])
            else:
                new_tensor[i][j] = _convert(tensor[i][j // 2], j % 2)
    return new_tensor


def print_bit(name, val):
    val_cpu = val.cpu().item()
    binary_repr = f'{val_cpu:032b}'
    print(name, binary_repr)
