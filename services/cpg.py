import numpy as np


def compute_global_cpg(seq: str) -> float:
    return (sum(seq[i:i+2] == "CG" for i in range(len(seq) - 1)) /
            max(len(seq) - 1, 1) * 100)


def compute_windows_cpg(seq: str, ws: int, ss: int) -> list[dict]:
    return [
        {"start": i, "end": i + ws,
         "cpg_percent": compute_global_cpg(seq[i:i + ws])}
        for i in range(0, len(seq) - ws + 1, ss)
    ]


def compute_windows_cpg(seq: str, ws: int, ss: int) -> list[dict]:
    seq_arr = np.frombuffer(seq.encode(), dtype='S1')
    n = len(seq_arr)
    
    is_c = (seq_arr == b'C')
    is_g = (seq_arr == b'G')
    
    is_cg = is_c[:-1] & is_g[1:]
    
    cumsum_cg = np.cumsum(is_cg, dtype=int)
    cumsum_cg = np.insert(cumsum_cg, 0, 0)
    
    # Those are both np arrays
    starts = np.arange(0, n - ws + 1, ss)
    ends = starts + ws
    
    cg_counts = cumsum_cg[ends-1] - cumsum_cg[starts]
    
    cg_percentages = cg_counts / (ws - 1) * 100
    
    return [{"start": int(s), "end": int(e), "cpg_percent": float(p)} 
            for s, e, p in zip(starts, ends, cg_percentages)]