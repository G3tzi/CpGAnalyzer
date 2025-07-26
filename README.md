# CpG Content Analysis: Server-Client Tool for Human DNA Exploration

A Python-based client-server system for analyzing CpG content in human DNA sequences, supporting raw and FASTA inputs, sliding window profiling, and CpG island detection.

---
## Overview

CpG content plays an important role in gene regulation and epigenetic mechanisms. This tool enables the analysis of CpG distribution across human DNA sequences, helping researchers identify CpG-rich regions (potential islands) and associate them with biologically relevant genomic elements such as promoters or regulatory regions.

---

## Table of Contents

* [Features](#features)
* [Info](#info)
* [Usage](#usage)
  * [Running the Server](#running-the-server)
  * [Using the Client](#using-the-client)
* [Input Formats](#input-formats)
* [Output Example](#output-example)
* [Installation](#installation)
* [Configuration](#configuration)
* [Requirements](#requirements)
* [License](#license)

---

## Features

* Accepts raw or FASTA-formatted DNA sequences
* Accepts both single and multiple sequence inputs in FASTA-format
* Accepts both file input or console parameter input
* Computes:
  * Global CpG percentage
  * CpG profile with sliding windows
* Detects CpG islands with >60% CpG content
* Maps CpG islands to biologically relevant regions (e.g., genes, promoters, cCREs) based on ENCODE API
* Client-server communication over HTTP with JSON
* Customizable host and port (for both server and client)
* Multiprocessing handling for the server to improve performance with multiple requests

---
## Info

* The entire structure and single functions' descriptions are described in STRUCTURE.md
* DNA.txt is an example of FASTA file with multiple sequences.

---

## Usage
### Running the Server

**Terminal command**
```bash
python server.py
```

* Set `host` and `port` in the config file (`config.json`)

#### Server API Endpoints
Server URL: `http://<host>:<port>/cpg/<endpoint>`

| Endpoint         | Method | Description                        |
|------------------|--------|------------------------------------|
| `/cpg/global`    | POST   | Returns global CpG content         |
| `/cpg/sliding`   | POST   | Returns sliding window CpG profile |
| `/cpg/islands`   | POST   | Detects CpG islands and annotates  |

All of them accept a JSON body with the structure [explained later](#json-payload).

---

### Using the Client

**Terminal command**
```bash
python client.py --host 127.0.0.1 --port 8000 --req 1 --seq AACCCGGCTCCGCATTCTTTCCCACACTCGCCCCAGCCAATCGAC -o output.json
```
```bash
python client.py --host 127.0.0.1 --port 8000 --req 2 --file input.txt --ws 200 --ss 50 > output.txt
```
```bash
python client.py --host 127.0.0.1 --port 8000 --req 3 --file input.txt --ws 200 --ss 50 --genome hg38
```

* `--req` can be:
    * `1`: global CpG
    * `2`: Sliding window
    * `3`: Find CpG islands
* The input sequence can be either a string (`--seq`) or a file (`--file`) ([see later](#input-formats))
* The output result can be either saved in a specific file (`-o`) or printed to the console (`>`) ([see later](#output-example))
* `--ws` and `--ss` are required for `--req`=`2` or `3`
* `--chrom` and `--start` can be used to specify the location of a raw dna sequence
* `--genome` can be used to specify the genome assembly (`hg19` or `hg38`)
* `--host` and `--port` have a default value configurable via `config.json`
* `--req` has default value `1` and `--genome` has defaulte value `hg38`
---

## Input Formats

* **Raw DNA string**:
  Example: `ATCGCGTATCGATCGA...`

* **FASTA**:
  ```
  >chr1:1000-1100
  ATCGCGTATCGATCGAGCGCGCG...
  ```

Can be both passed as a string or a file path:
* `--seq` for the string parameter
* `--file` for the file path parameter

(A FASTA-formatted file can containe more than a single sequence, they're all processed together by sending them asynchronously to the server)

#### JSON payload example: {#json-payload}

`--req` = 1:
```json
{
  "sequence": "ATCGCGTATCGATCGAGCGCGCG...",
  "format": "raw",
}
```
`--req` = 2:
```json
{
  "sequence": "ATCGCGTATCGATCGAGCGCGCG...",
  "format": "raw",
  "window_size" = 200,
  "step_size" = 50
}
```
`--req` = 3:
```json
{
  "sequence": "ATCGCGTATCGATCGAGCGCGCG...",
  "format": "raw",
  "window_size" = 200,
  "step_size" = 50,
  "chrom" = "chr1"
  "start" = 100
  "genome" = "hg38"
}
```

---

## Output Example

#### Handle output
* `--output` (`-o`) to specify the output location.
* Otherwise the output will be printed in the terminal: you can use `>` or `>>` to save/append it to a file.

#### Examples
`--req` = 1
```json
{
  "seq0": {
    "cpg_percent": 5.96
  }
}
```

`--req` = 2
```json
{
  "seq0": {
    "windows": [
      {
        "start": 0,
        "end": 200,
        "cpg_percent": 17.08542713567839
      },
      {
        "start": 50,
        "end": 250,
        "cpg_percent": 15.07537688442211
      },
      ...
    ],
    "summary": {
      "mean": 5.8,
      "min": 0.5,
      "max": 17.09,
      "std": 4.49
    }
  }
}
```
`--req` = 3
```json
{
  "seq0": {
    "islands": [
      {
        "start": 3500,
        "end": 3700,
        "length": 200,
        "gc_content": 0.65,
        "expected_cg": 21.12,
        "oe_ratio": 0.615530303030303,
        "ambiguous_content": false,
        "annotations": [
          {
            "accession": "ENCSR800VNX",
            "type": "Annotation",
            "description": "agnostic candidate Cis-Regulatory Elements for GRCh38",
            "lab": "Zhiping Weng, UMass",
            "organism": "Homo sapiens",
            "status": "released",
            "audit": [
              {
                "path": "/files/ENCFF420VPZ/",
                "level": 30,
                "name": "audit_file",
                "detail": "derived_from is a list of files that were used to create a given file; for example, fastq file(s) will appear in the derived_from list of an alignments file. Processed file {ENCFF420VPZ|/files/ENCFF420VPZ/} is missing the requisite file specification in its derived_from list.",
                "category": "missing derived_from"
              },
              ...
            ]
          },
          ...
        ]
      },
      ...
    ]
  }
}
```
`ambiguous_content` is a boolean which specify whether the island has ambiguous nucleotides: `R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`, `N` (allowed nucleotides other than `A`, `C`, `G`, `T`).

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Configuration

Example `config.json` (for the server):

```json
{
  "server_host": "127.0.0.1",
  "server_port": 8000,
  "default_gc_threshold": 0.5,
  "default_oe_threshold": 0.6
}
```
Example `config.json` (for the client):
```json
{
  "default_host": "127.0.0.1",
  "default_port": 8002
}
```

---

## Requirements

* Python >= 3.10.7
* Libraries: `fastAPI`, `uvicorn`, `httpx` etc. (see [`requirements.txt`](#installation))

---

## License

MIT License. See `LICENSE` file for details.
