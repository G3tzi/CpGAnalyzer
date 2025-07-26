import httpx
import asyncio

from utils import extract_region_from_fasta_header, get_logger

logger = get_logger()


async def fetch_encode_data(chrom: str, start: str, end: str) -> list[dict]:
    """Fetch regulatory element data from ENCODE API for a given genomic range."""
    url = "https://www.encodeproject.org/search/"

    params = {
        "type": "Annotation",
        "annotation_type": "candidate Cis-Regulatory Elements",
        "assembly": "GRCh38",
        "region": f"{chrom}:{start}-{end}",
        "format": "json",
    }

    headers = {
        "Accept": "application/json"
    }

    try:
        async with httpx.AsyncClient(trust_env=True) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        msg = f"An error occured during the API call: {e}"
        logger.error(msg)
        return [msg]

    # Check if '@graph' contains data (It is the attrib used by ENCODE)
    if "@graph" in data and isinstance(data["@graph"], list):
        graph_data = data["@graph"]
    else:
        logger.warning("No relevant data found in '@graph'.")
        return []

    annotations = []

    for item in graph_data:
        # Extract relevant information
        annotation = {
            "accession": item.get("accession"),
            "type": item.get("@type", ["unknown"])[0],
            "description": item.get("description", "No description available"),
            "lab": item.get("lab", {}).get("title", "No lab info"),
            "organism": item.get("organism", {}).get("scientific_name",
                                                     "Unknown organism"),
            "status": item.get("status", "Unknown status"),
            "audit": item.get("audit", {}),
        }

        # Add audit information if it exists
        if "audit" in annotation and isinstance(annotation["audit"], dict):
            audit_details = []
            for audit_entry in annotation["audit"].get("INTERNAL_ACTION", []):
                audit_details.append({
                    "path": audit_entry.get("path"),
                    "level": audit_entry.get("level"),
                    "name": audit_entry.get("name"),
                    "detail": audit_entry.get("detail"),
                    "category": audit_entry.get("category")
                })
            annotation["audit"] = audit_details

        annotations.append(annotation)

    return annotations


async def fetch_and_annotate_islands(islands: list, request):
    """
    Fetch ENCODE annotations asynchronously for a list of CpG islands.
    """

    sequence: str = request.sequence

    if request.format == "fasta":
        header = sequence.splitlines()[0]
        region_info = extract_region_from_fasta_header(header)
        if region_info:
            chrom, seq_start, seq_stop = region_info
    else:
        chrom = ("chr" + request.chrom if not request.chrom.startswith("chr")
                 else request.chrom)
        seq_start = request.start

    if not (chrom and seq_start):
        logger.error("Cannot annotate without region context")
        return islands

    if request.genome != "hg38":
        logger.error("ENCODE annotations are only available for hg38")
        return islands

    tasks = []
    for island in islands:
        start = str(int(island["start"]) + int(seq_start))
        end = str(int(island["end"]) + int(seq_start))
        tasks.append(fetch_encode_data(chrom, start, end))

    annotations_list = await asyncio.gather(*tasks)

    annotated_islands = []
    for island, annotations in zip(islands, annotations_list):
        annotated_islands.append({
            "start": island["start"],
            "end": island["end"],
            "length": island["end"] - island["start"],
            "gc_content": island.get("gc_content"),
            "expected_cg": island.get("expected_cg"),
            "oe_ratio": island.get("oe_ratio"),
            "ambiguous_content": island.get("ambiguous_content"),
            "annotations": annotations
        })

    return annotated_islands
