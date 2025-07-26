# Project Directory Structure

- [client.py](#client)
- [server.py](#server)
- [utils.py](#utils)
- [config.py](#config)
- config.json
- requirements.txt
- README.md
- STRUCTURE.md
- LICENSE
- client/
   - [requests.py](#client-requests)
   - [utils.py](#client-utils)
- models/
   - [requests.py](#models-requests)
- server/
   - [api.py](#api)
   - [routers.py](#routers)
- services/
    - [cpg.py](#cpg)
    - [encode_api.py](#encode-api)
    - [island_detection.py](#islands)

---
# Functions descriptions

## `client.py` Module Documentation {#client}

This module runs the client application.

---

### Functions Overview

**main**
```python
async def main():
    """Main function to orchestrate the processing and sending of requests."""
    parser = parse_parser()
    args = parser.parse_args()

    # Get sequences from either file or sequence string
    try:
        seqs: tuple[str, list] = get_sequences(args.seq or
                                               read_file(args.file))
    except ValueError:
        exit()

    if seqs[0] != "raw" and (args.chrom or args.start):
        parser.error("The location (--chrom, --start) can be\
                     specified only if the input format is raw\
                     (in a fasta input the location can be \
                     specified in the header).")

    # Prepare the data to be sent in the request
    data: list[dict[str, str]] = [
            {"sequence": s, "format": seqs[0]} for s in seqs[1]
        ]

    if args.req in {2, 3}:
        for d in data:
            d["window_size"] = args.ws
            d["step_size"] = args.ss
            if args.req in {3}:
                d["chrom"] = str(args.chrom)
                d["start"] = args.start
                d["genome"] = args.genome

    # Construct URL
    url: str = construct_url(args.host, args.port, args.req)

    # Send asynchronous requests
    tasks = [send_request(d, url) for d in data]
    try:
        responses = await asyncio.gather(*tasks)
        handle_responses(responses, args)
    except Exception as e:
        logger.error(e)
```
Gets the user params → gets the sequences from the input → prepares the data to be sent in the request → constructs the URL, sends the asynchronous requests → and handles the responses.

---
## `server.py` Module Documentation {#server}

This module runs the server application.

---
## `utils.py` Module Documentation {#utils}

This module handles the utility functions for the program.

### Functions Overview

**extract_sequence**
```python
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
```
If the format is "fasta", it extracts the sequence from the data by removing the header lines.
Raises ValueError if the sequence is empty or invalid.

**extract_region_from_fasta_header**
```python
def extract_region_from_fasta_header(header: str) -> tuple | None:
    match = re.search(r"(chr\w+):(\d+)-(\d+)", header)
    if match:
        return match.groups()
    return None
```
Extracts the chromosome, start, and end from the FASTA header.

**valid_dna**
```python
def valid_dna(seq: str) -> bool:
    return all(base in allowed for base in seq.upper())
```
Checks if the sequence is a valid DNA sequence. The validity is based on the `iupac nucleic acid codes`.

**windows_summary**
```python
def windows_summary(windows: list[dict]) -> dict:
    values = np.array([w["cpg_percent"] for w in windows])
    return {
        "mean": round(values.mean(), 2),
        "min": round(values.min(), 2),
        "max": round(values.max(), 2),
        "std": round(values.std(), 2),
    }
```
Compute standard metrics for each window.

**parse_encode_ccre_response**
```python
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
```
Parse a list of ENCODE cCRE metadata dictionaries to extract key attributes like coordinates, biosample, lab, and assembly.

**get_logger**
```python
def get_logger(name: str = "cpg_tool"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```
This function creates a logger with a specified name and sets up a stream handler to output logs to the console. If the logger already has handlers, it will not add another one.

---
## `config.py` Module Documentation {#config}

```python
import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "config.json")

with open(file_path) as f:
    config = json.load(f)

HOST = config.get("server_host", "127.0.0.1")
PORT = config.get("server_port", 8000)
DEF_HOST = config.get("default_host", "127.0.0.1")
DEF_PORT = config.get("default_port", 8000)
GC_THRESHOLD = config.get("default_gc_threshold", 0.5)
OE_THRESHOLD = config.get("default_oe_threshold", 0.6)
```

Loads the configuration settings in config.json

---
## `client/requests.py` Module Documentation {#client-requests}
This module handle the requests sendings.

---
### Functions Overview
**send_request**
```python
async def send_request(d: dict, url: str):
    """Send an asynchronous POST request to the server and return the response."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=d)
            if response.status_code == 200:
                return response
            else:
                raise Exception(f"Error: {response.status_code} {response.text}")
    except httpx.HTTPStatusError as e:
        raise Exception(f"Error: {e.response.status_code} {e.response.text}")
    except httpx.RequestError as e:
        raise Exception(f"Error: Unable to send request - {str(e)}")
```
Sends post requests and raise exceptions if the request is not satisfied correctly.

---
## `client/utils.py` Module Documentation {#client-utils}
This module handles the utility functions for the program.

---
### Functions Overview
**positive_int**
```python
def positive_int(value: str) -> int:
    """Validate that the value is a positive integer."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue
```
Function defined to validate the user input checking if the parameter is a positive integer.
**chromosome**
```python
def chromosome(value: str) -> str:
    """Validate that the value is a positive chromosome number (1-22, X, or Y)."""
    try:
        ivalue = int(value)
        if 1 <= ivalue <= 22:
            return ivalue
    except ValueError:
        if str(value).upper() in {'X', 'Y'}:
            return value.upper()

    raise argparse.ArgumentTypeError(f"{value} is not a valid chromosome number")
```
Function defined to validate the user input checking if the parameter is a valid chromosome number (or string: X and Y).
**read_file**
```python
def read_file(file: str) -> str:
    """Read the content of the file and return as a string."""
    with open(file, "r") as f:
        return f.read()
```
Reads the content of the file spcified by the path `file`.
**get_sequences**
```python
def get_sequences(seq: str) -> tuple[str, list[str]]:
    """Parse the input sequence or file into a tuple with format and sequences."""
    seqs = []
    fmt = "raw"

    if not seq:
        logger.error("Invalid empty input")
        raise ValueError("Invalid empty input")

    # Check if the sequence is in FASTA format
    if '>' in seq:
        fmt = "fasta"
        # Handle multiple reads in a single FASTA file
        seqs = [s for s in re.split(r'(?=>)', seq) if s]
    else:
        seqs.append(seq)

    # Validate each sequence
    for s in seqs:
        sequence_part = ''.join(s.strip().
                                splitlines()[1:]) if s.startswith('>') else s
        if not valid_dna(sequence_part):
            logger.error("Invalid input format")
            raise ValueError("Invalid input format")

    return fmt, seqs
```
Extract the sequences from the original one:
* if it is a fasta format, it splits on the headers and add every sequence to a list, otherwise it adds the seq directly to the seq.
* It then validate the sequences.
* return a tuple with the format and the sequences.

**parse_parser**
```python
def parse_parser() -> argparse.ArgumentParser:
    """Parse command-line arguments and handle validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default=DEF_HOST,
                        help=f'Server IP (default: {DEF_HOST})')
    parser.add_argument('-p', '--port', type=positive_int, default=int(DEF_PORT),
                        help=f'Server port (default: {DEF_PORT})')
    parser.add_argument('--req', type=int, choices=[1, 2, 3], default=1,
                        help='Request type: 1 = global CpG, 2 = sliding\
                              windows CpG, 3 = find CpG islands (default: 1)')
    parser.add_argument('-f', '--file', type=str, help='Input file name')
    parser.add_argument('-s', '--seq', type=str, help='Input sequence')
    parser.add_argument('--ws', type=positive_int,
                        help='Window size (required for sliding windows request)')
    parser.add_argument('--ss', type=positive_int,
                        help='Step size (required for sliding windows request)')
    parser.add_argument('-c', '--chrom', type=chromosome,
                        help='Chromosome location of the raw DNA seq')
    parser.add_argument('--start', type=positive_int,
                        help='Starting position of the raw DNA seq')
    parser.add_argument('-g', '--genome', type=str, choices=["hg19", "hg38"],
                        default="hg38", help='Genome: hg19, hg38 (default: hg38)')
    parser.add_argument('-o', '--output', type=str, help='Outdir')

    args = parser.parse_args()

    if args.req in {2, 3} and not (args.ws and args.ss):
        parser.error("--ws and --ss are required if --req is 2 or 3!")

    if args.req not in {2, 3} and (args.ws or args.ss):
        parser.error("--ws and --ss are not allowed with this kind of --req")

    if bool(args.file) == bool(args.seq):
        parser.error("You must provide either --f or --s, but not both.")

    return parser
```
Handles command-line arguments and their validation.
* If the `--req` is 2 or 3, `--ws` and `--ss` are required. Otherwise they must be absent.
* Only one between the two kind of input can be used (`--file` or `--seq`).
* Return the parser in order to raise parser.error for other cases.

**construct_url**
```python
def construct_url(host: str, port: int, req: int) -> str:
    """Construct the URL for the request based on the host, port, and request type."""
    endpoint = {
        1: "global",
        2: "sliding",
        3: "islands"
    }[req]
    return f"http://{host}:{port}/cpg/{endpoint}"
```
Construct the request URL based on: `--host`, `--port` and `--req`.
**handle_responses**
```python
def handle_responses(responses: list[str], req) -> None:
    final_output = {}
    for index, response in enumerate(responses):
        try:
            data = response.json()
        except json.JSONDecodeError:
            logger.error("Error: Response is not valid JSON")
            data = response.text
        final_output["seq" + str(index)] = data

    if hasattr(req, "output") and req.output:
        try:
            with open(req.output, "w") as f:
                json.dump(final_output, f, indent=2)
            logger.info(f"Response written to {req.output}")
        except IOError as e:
            logger.error(f"Failed to write to file {req.output}: {e}")
    else:
        print(json.dumps(final_output, indent=2))

    return
```
Builds a json composing every single response.
If `--output` was specified, it writes the response in that file path. Otherwise the output is printed on the stdout in order for the user to use it in command pipelines or append it to an existing one.

---
## `models/requests.py` Module Documentation {#models-requests}
Defines the requests' models that the server will receive because of the client.

---
## `server/api.py` Module Documentation {#api}
Defines the fastAPI app the include the routers of the endpoints it will be accessible through.

---
## `server/routers.py` Module Documentation {#routers}
Defines the entpoints behaviour.
* fastAPI is by definition asynchronous, but those kind of requests are CPU bounded so it is implementing a multiprocess technique (ProcessPoolExecutor) in order to handles the multiple requests efficiently.

---
### Functions Overview
**cpg_global**
```python
@router.post("/cpg/global")
async def cpg_global(request: CpGRequest) -> dict:
    sequence = extract_sequence(request.sequence, request.format)

    loop = asyncio.get_event_loop()
    result: list[dict] = await loop.run_in_executor(
        executor,
        compute_global_cpg,
        sequence
    )

    return {"cpg_percent": round(result, 2)}
```
Handles the `--req 1` case, computing the global cpg percentage.
**sliding_cpg**
```python
@router.post("/cpg/sliding")
async def sliding_cpg(request: WindowRequest) -> dict:
    sequence = extract_sequence(request.sequence, request.format)

    loop = asyncio.get_event_loop()
    windows: list[dict] = await loop.run_in_executor(
        executor,
        compute_windows_cpg,
        sequence, request.window_size, request.step_size
    )

    return {"windows": windows, "summary": windows_summary(windows)}
```
Handles the `--req 2` case, computing the global cpg percentage.
**cpg_islands**
```python
@router.post("/cpg/islands")
async def cpg_islands(request: IslandsRequest) -> dict:

    loop = asyncio.get_event_loop()
    islands: list[dict] = await loop.run_in_executor(
        executor,
        detect_islands,
        request.sequence, request.format,
        request.window_size, request.step_size
    )

    annotated = await fetch_and_annotate_islands(islands, request)
    return {"islands": annotated}
```
Handles the `--req 4` case, identifying the CpG islands and fetching annotations for them if the location where specified.
* The `fetch_and_annotate_islands` calls some API, this means it is not CPU bound and for this reason it is called asynchronously instead of inside the ProcessPoolExecutor.

---
## `services/cg.py` Module Documentation {#cpg}
Contains the CpG related functions.

---
### Functions Overview
**compute_global_cpg**
```python
def compute_global_cpg(seq: str) -> float:
    return (sum(seq[i:i+2] == "CG" for i in range(len(seq) - 1)) /
            max(len(seq) - 1, 1) * 100)
```
Compute the global CpG content prcentage.
**compute_windows_cpg**
```python
def compute_windows_cpg(seq: str, ws: int, ss: int) -> list[dict]:
    seq_arr = np.frombuffer(seq.encode(), dtype='S1')
    n = len(seq_arr)
    
    # Create boolean arrays for C and G positions
    is_c = (seq_arr == b'C')
    is_g = (seq_arr == b'G')
    
    # Compute CG dinucleotides (C at position i and G at position i+1)
    is_cg = is_c[:-1] & is_g[1:]
    
    # Compute cumulative sums for efficient window calculations
    cumsum_cg = np.cumsum(is_cg, dtype=int)
    cumsum_cg = np.insert(cumsum_cg, 0, 0)
    
    # Calculate start positions for all windows
    starts = np.arange(0, n - ws + 1, ss)
    ends = starts + ws
    
    # Calculate CG counts for all windows
    cg_counts = cumsum_cg[ends-1] - cumsum_cg[starts]
    
    # Calculate percentages
    cg_percentages = cg_counts / (ws - 1) * 100
    
    # Return as list of dictionaries
    return [{"start": int(s), "end": int(e), "cpg_percent": float(p)} 
            for s, e, p in zip(starts, ends, cg_percentages)]
```
Compute the global CpG content prcentage on every window specified by `--ws` and `--ss`.
In order to do so, it exploits the numpy functionalities such as np.cumsum.
* `is_c` and `is_g` are two logical vectors.
* `is_c[:-1] & is_g[1:]` is performing a logical AND operation between `is_c[:-1]` and `is_g[1:]` retrieving a new logical vector which points out the `cg` dinucleotides.
* Compute a cumsum in order to avoid recomputing the sum of `cg` dinucleotides for every window given that part of it is already computed for the preceding one.
* `starts` and `ends` are two **arrays** which contain the start and end positions of every window.
* Avoid iterating over it using numpy structures.

---
## `services/encode_api.py` Module Documentation {#encode-api}
Handles ENCODE requests.

---
### Functions Overview
**fetch_encode_data**
```python
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

    # Check if '@graph' contains data
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

        # Append the annotation to the list
        annotations.append(annotation)

    return annotations
```
Performs the API call to encodeproject and handles the response.
**fetch_and_annotate_islands**
```python
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
```
Handles multiple islands, build the absolute position (relative one + the starting position).
* The requests are performed asynchronously.
* In order to get annotations, the genome has to be hg38.

---
## `services/island_detection.py` Module Documentation {#islands}
This module implements the island detection.

---
### Functions Overview
**recompute_stats**
```python
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
```
Uses numpy cumsum vectors in order to compute gc_content, expected_cg and oe_ratio in an efficiently way.
**detect_islands**
```python
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
```
Implements the `Gardiner-Garden and Frommer CpG island detection algorithm` exployting numpy functionality to convert the original vector to boolean values and then applying the cumsum obtaining vectors that allows me to avoid reiterate over it a huge number of time.
* It firstly compute the stats for the windows and append the island only if it satisfy the actual thresholds.
* Then it starts merging them if they are overlapped.
* Default `gc_threshold` and `oe_threshold` can both be passed as parameters or changed in the config file.
