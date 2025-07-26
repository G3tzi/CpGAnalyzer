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
