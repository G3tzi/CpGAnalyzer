import numpy as np

from utils import extract_sequence, reduced
from config import GC_THRESHOLD, OE_THRESHOLD


def recompute_stats(start: int, end: int, cumsum_c: np.ndarray,
                     cumsum_g: np.ndarray, cumsum_cg: np.ndarray) -> tuple:

    end_incl = end - 1
    num_c = cumsum_c[end_incl] - (cumsum_c[start - 1] if start > 0 else 0)
    num_g = cumsum_g[end_incl] - (cumsum_g[start - 1] if start > 0 else 0)
    num_cg = cumsum_cg[end_incl - 1] - (cumsum_cg[start - 1]
                                        if start > 0 else 0)
    length = end_incl - start + 1
    gc_content = (num_c + num_g) / length
    expected_cg = (num_c * num_g) / length if length > 0 else 0
    oe_ratio = (num_cg / expected_cg) if expected_cg > 0 else 0
    return gc_content, expected_cg, oe_ratio


def detect_islands(sequence: str, fmt: str, window_size: int, step_size: int,
                   gc_threshold: float = GC_THRESHOLD,
                   oe_threshold: float = OE_THRESHOLD) -> list[dict]:
    # Gardiner-Garden and Frommer CpG island detection algorithm
    seq = extract_sequence(sequence, fmt).upper()
    window_size = window_size
    step_size = step_size
    n = len(seq)

    is_c = np.fromiter((base == 'C' for base in seq), dtype=np.bool_, count=n)
    is_g = np.fromiter((base == 'G' for base in seq), dtype=np.bool_, count=n)
    is_n = np.fromiter((base not in reduced for base in seq), dtype=np.bool_, count=n)
    is_cg = np.fromiter((seq[i:i+2] == 'CG' for i in range(n-1)),
                        dtype=np.bool_, count=n-1)

    cumsum_c = np.cumsum(is_c)
    cumsum_g = np.cumsum(is_g)
    cumsum_cg = np.cumsum(is_cg)
    cumsum_n = np.cumsum(is_n)

    islands = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        gc_content, expected_cg, oe_ratio = recompute_stats(start, end, cumsum_c,
                                                             cumsum_g, cumsum_cg)
        num_n = cumsum_n[end - 1] - (cumsum_n[start - 1] if start > 0 else 0)
        if gc_content >= gc_threshold and oe_ratio >= oe_threshold:
            islands.append({"start": start, "end": end, "ambiguous_content": bool(num_n > 0)})

    if not islands:
        return []

    merged_islands = [islands[0]]
    for island in islands[1:]:
        last = merged_islands[-1]
        if island["start"] <= last["end"]:
            last["end"] = max(last["end"], island["end"])
        else:
            merged_islands.append(island)

    final_islands = []
    for island in merged_islands:
        s, e = island["start"], island["end"]
        gc_content, expected_cg, oe_ratio = recompute_stats(s, e, cumsum_c,
                                                             cumsum_g, cumsum_cg)
        final_islands.append({
            "start": s,
            "end": e,
            "gc_content": gc_content,
            "expected_cg": expected_cg,
            "oe_ratio": oe_ratio,
            "ambiguous_content": island["ambiguous_content"]
        })

    return final_islands
