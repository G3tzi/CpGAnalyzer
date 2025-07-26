import re
import json
import argparse

from config import DEF_HOST, DEF_PORT
from utils import valid_dna, get_logger

logger = get_logger()


def positive_int(value: str) -> int:
    """Validate that the value is a positive integer."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


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


def read_file(file: str) -> str:
    """Read the content of the file and return as a string."""
    with open(file, "r") as f:
        return f.read()


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


def parse_parser() -> argparse.ArgumentParser:
    """Parse command-line arguments and handle validation."""
    parser = argparse.ArgumentParser(description="A simple example script.")
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


def construct_url(host: str, port: int, req: int) -> str:
    """Construct the URL for the request based on the host, port, and request type."""
    endpoint = {
        1: "global",
        2: "sliding",
        3: "islands"
    }[req]
    return f"http://{host}:{port}/cpg/{endpoint}"


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
