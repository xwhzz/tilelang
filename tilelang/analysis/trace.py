from scipy.special import comb
import math
import hashlib
from dataclasses import dataclass

def norm_cdf_approx_3(x):
    a1, a2, a3 = 0.4361836, 0.1201676, 0.937298
    k = 1.0 / (1.0 + 0.33267 * abs(x))
    approx = 1.0 - (1.0 / (math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)) * \
             (a1 * k + a2 * k**2 + a3 * k**3 )
    return approx if x >= 0 else 1 - approx

def sdcm(D, A, B):
    if (B==0): return 0
    p = A / B
    mu = D * p
    sigma = math.sqrt(D * p * (1 - p))

    prob = 0
    D = math.ceil(D)
    
    if D <= A - 1:
        prob = 1
    elif D >= 1e8:
        prob = 0
    elif D <= 8:
        for a in range(A):
            prob += comb(D, a) * (A/B)**a * ((B-A)/B)**(D-a)
    else:
        norm_input=(A - 1 + 0.5 - mu) / sigma
        prob = norm_cdf_approx_3(norm_input)
    
    return prob

# ==========================================
# 2. Data Structure Abstraction
# ==========================================

@dataclass
class TraceEvent:
    tensor_name: str    # e.g., 'A', 'B', 'C'
    coords: tuple       # e.g., (by, ko)
    size_bytes: int     # Size of the tile in bytes
    is_write: bool      # True for Store, False for Load

    def get_hash_key(self):
        """Returns a unique hashable key for this specific tile."""
        return (self.tensor_name, self.coords)

# ==========================================
# 3. Partition-Aware Simulator
# ==========================================

class CachePartition:
    def __init__(self, capacity_lines, assoc):
        self.capacity_lines = capacity_lines
        self.assoc = assoc
        self.current_volume = 0.0
        # Map: (Tensor, Coords) -> Last Volume Timestamp
        self.history = {}
        
    def access(self, key, num_lines):
        prob = 0.0
        
        if key in self.history:
            # Calculate reuse distance based on volume processed *within this partition*
            distance = self.current_volume - self.history[key]
            prob = sdcm(distance, self.assoc, self.capacity_lines)
        else:
            prob = 0.0 # Cold miss
            
        self.history[key] = self.current_volume
        self.current_volume += num_lines
        return prob

class GeneralizedL2Simulator:
    def __init__(self, l2_cap_bytes, num_partitions=2, assoc=8, line_bytes=128):
        self.line_bytes = line_bytes
        self.num_partitions = num_partitions
        
        # Split capacity across partitions
        partition_cap_lines = (l2_cap_bytes / line_bytes) / num_partitions
        
        self.partitions = [
            CachePartition(partition_cap_lines, assoc) 
            for _ in range(num_partitions)
        ]
        
        # Stats
        self.stats = {
            'read_hits_bytes': 0,
            'read_miss_bytes': 0,
            'write_bytes': 0
        }

    def _get_partition_id(self, key):
        # Hash the tile key to find which L2 slice it lives in
        # We use a string hash to ensure deterministic distribution
        s_key = f"{key[0]}_{key[1]}"
        # transform to base 10 integer
        hash_val = int(hashlib.md5(s_key.encode()).hexdigest(), 16)
        return hash_val % self.num_partitions

    def run(self, trace_sequence):
        """
        Input: trace_sequence (List[TraceEvent])
        """
        for event in trace_sequence:
            num_lines = event.size_bytes / self.line_bytes
            key = event.get_hash_key()
            
            # 1. Determine L2 Partition
            pid = self._get_partition_id(key)
            partition = self.partitions[pid]
            
            # 2. Simulate Access
            if event.is_write:
                # Stores typically update history (Write-Allocate) but don't return "Hit Data"
                partition.access(key, num_lines)
                self.stats['write_bytes'] += event.size_bytes
            else:
                # Reads
                hit_prob = partition.access(key, num_lines)
                
                # We accumulate fractional bytes based on probability
                self.stats['read_hits_bytes'] += event.size_bytes * hit_prob
                self.stats['read_miss_bytes'] += event.size_bytes * (1.0 - hit_prob)

    def get_io_metrics(self):
        # Base Data
        r_hit = self.stats['read_hits_bytes']
        r_miss = self.stats['read_miss_bytes']
        w_total = self.stats['write_bytes']
        r_total = r_hit + r_miss
        
        hit_rate = r_hit / r_total if r_total > 0 else 0
        
        # --- IO Calculation Logic (Aligned with Groundtruth Formulas) ---
        
        # Groundtruth L2 IO Formula:
        # l2_io = l2_read_io*(hit) + l2_read_io*(miss)*2 + l2_store_io*2
        # Interpretation: 
        #   - Read Hit: 1 transaction
        #   - Read Miss: 2 transactions (Request + Fill)
        #   - Write: 2 transactions (Request + Data?)
        # Note: Groundtruth applies a *2 multiplier to the RAW store bytes input. 
        # We assume w_total is raw bytes, so we follow the formula structure strictly.
        
        # In groundtruth: l2_store_io = gridM*gridN*tb_size * 2. 
        # We will assume our w_total is the raw size, so we apply the factor 2 explicitly if needed
        # or assume the formula in groundtruth meant "Cost of Store".
        # Let's align strictly:
        
        l2_io = (r_total * hit_rate) + \
                (r_total * (1 - hit_rate) * 2) + \
                (w_total * 2) 

        # DDR IO Formula:
        # ddr_io = l2_read_io*(1-hit_rate) + l2_store_io
        # Note: In groundtruth, l2_store_io variable included a *2 factor.
        # But logically DDR IO is just Miss Bytes + Writeback Bytes.
        # We will stick to the components:
        ddr_io = r_miss + w_total

        return hit_rate, l2_io, ddr_io