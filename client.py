import asyncio

from client.utils import parse_parser, get_sequences, \
    read_file, construct_url, get_logger, handle_responses
from client.requests import send_request

logger = get_logger()


async def main():
    """Main function to orchestrate the processing and sending of requests."""
    parser = parse_parser()
    args = parser.parse_args()

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

    url: str = construct_url(args.host, args.port, args.req)

    tasks = [send_request(d, url) for d in data]
    try:
        responses = await asyncio.gather(*tasks)
        handle_responses(responses, args)
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    asyncio.run(main())
