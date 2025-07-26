import re
import sys
import logging
import numpy as np

reduced = {'A', 'C', 'G', 'T'}
extended = {'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V', 'N'}
allowed = reduced | extended

def extract_sequence(data: str, fmt: str) -> str:
    if fmt == "fasta":
        lines = data.strip().splitlines()
        data = ''.join(line for line in lines if not line.startswith('>'))
    data = data.strip().upper()
    if not data:
        raise ValueError("Sequence cannot be empty.")
    if not valid_dna(data):
        raise ValueError("Invalid DNA sequence.")
    return data


def extract_region_from_fasta_header(header: str) -> tuple | None:
    match = re.search(r"(chr\w+):(\d+)-(\d+)", header)
    if match:
        return match.groups()
    return None


def valid_dna(seq: str) -> bool:
    return all(base in allowed for base in seq.upper())


def windows_summary(windows: list[dict]) -> dict:
    values = np.array([w["cpg_percent"] for w in windows])
    return {
        "mean": round(values.mean(), 2),
        "min": round(values.min(), 2),
        "max": round(values.max(), 2),
        "std": round(values.std(), 2),
    }


def parse_encode_ccre_response(files: list[dict]) -> list[dict]:
    """Extract metadata for cCREs from ENCODE project API response dictionaries."""
    regulatory_elements = []
    for f in files:
        coords = re.search(r'(chr\w+):(\d+)-(\d+)', f.get("description", ""))
        if coords:
            chrom, start, end = coords.groups()
            regulatory_elements.append({
                "accession": f.get("accession"),
                "biosample": f.get("biosample_ontology", {}).get("term_name"),
                "lab": f.get("lab", {}).get("title"),
                "start": int(start),
                "end": int(end),
                "type": "cCRE",
                "status": f.get("status"),
                "assembly": f.get("assembly"),
            })
    return regulatory_elements


def get_logger(name: str = "cpg_tool"):
    logger = logging.getLogger(name)
    if not logger.handlers:  # prevent adding handlers multiple times
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
