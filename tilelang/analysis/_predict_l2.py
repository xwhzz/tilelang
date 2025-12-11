import math
import random
from dataclasses import dataclass

from scipy.special import comb

def norm_cdf_approx_3(x):
    """标准正态分布 CDF 近似"""
    a1, a2, a3 = 0.4361836, 0.1201676, 0.937298
    k = 1.0 / (1.0 + 0.33267 * abs(x))
    approx = 1.0 - (1.0 / (math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)) * \
             (a1 * k + a2 * k**2 + a3 * k**3 )
    return approx if x >= 0 else 1 - approx

def sdcm(D, A, B):
    """
    Statistical Direct/Set-Associative Cache Model
    D: Reuse Distance (number of unique cache lines accessed since last use)
    A: Associativity (ways)
    B: Total Cache Lines (L2 Capacity / Line Size)
    """
    if B == 0: return 0
    p = A / B
    
    # 如果距离非常小，必然命中
    if D <= A - 1:
        return 1.0
    # 如果距离过大，命中率为0 (近似优化)
    if D >= B: 
        return 0.0

    # 大数情况下使用正态分布近似二项分布
    # 我们计算的是 P(X < A)，即在这个距离内被挤出去的概率小于A（意味着还没被完全挤出去？）
    # 更准确的说是：Conflict Miss Probability。
    # 这里沿用你之前代码的逻辑：计算保留在 Cache 中的概率。
    
    mu = D * p
    sigma = math.sqrt(D * p * (1 - p))

    # 如果 D 比较小，用精确的组合数公式 (二项分布)
    if D <= 32: # 稍微调大一点阈值
        prob = 0
        for a in range(int(A)):
            prob += comb(int(D), a) * (p)**a * ((1-p))**(int(D)-a)
        return prob
    else:
        # 使用正态分布近似
        # 我们想求 P(Conflicts < A)
        # Standardize: Z = (A - 1 + 0.5 - mu) / sigma   (0.5 是连续性修正)
        norm_input = (A - 0.5 - mu) / sigma
        prob = norm_cdf_approx_3(norm_input)
        
    return max(0.0, min(1.0, prob))

@dataclass
class WorkUnit:
    name: str
    coord: tuple
    size: int

    def get_id(self):
        return (self.name, self.coord)

    def __repr__(self):
        return f"{self.name}[{self.coord}]"

def analyze_l2(sequence: list[WorkUnit], l2_cap_bytes: int, l2_assoc=8, cache_line_size=128, sm_count=114) -> tuple[float, float]:
    total_lines = l2_cap_bytes / cache_line_size
    last_seen = {}

    l2_io = 0
    ddr_io = 0
    
    total_access = len(sequence)
    
    num_waves = math.ceil(total_access / sm_count)
    for idx in range(num_waves):
        seq = sequence[idx * sm_count: (idx + 1) * sm_count]

        random.shuffle(seq)

        for idy, item in enumerate(seq):
            curr_idx = idx * sm_count + idy
            item_id = item.get_id()

            if item_id in last_seen:
                prev_idx = last_seen[item_id]

                interval_trace = sequence[prev_idx + 1 : curr_idx]

                unique_items = set()
                unique_volume_lines = 0.0

                for trace_item in interval_trace:
                    tid = trace_item.get_id()
                    if tid not in unique_items:
                        unique_items.add(tid)
                        unique_volume_lines += trace_item.size

                prob = sdcm(unique_volume_lines, l2_assoc, total_lines)
            else:
                prob = 0.0
                
            # two parition
            l2_io += item.size * prob + item.size * (1 - prob) * 2
            ddr_io += (1 - prob) * item.size
            last_seen[item_id] = curr_idx
    
    return l2_io, ddr_io
