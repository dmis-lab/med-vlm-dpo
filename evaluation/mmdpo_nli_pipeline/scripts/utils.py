import yaml
import os
from string import Template
import openai

def load_config():
    with open("./configs/paths.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_paths(config):
    """
    Expand $dataset, $backbone in paths.* entries.
    """
    dataset = config["dataset"]
    backbone = config["backbone"]
    vars = {"dataset": dataset, "backbone": backbone}

    expanded = {}
    for key, path in config["paths"].items():
        expanded[key] = Template(path).safe_substitute(vars)
    return expanded

def apply_api_key(config):
    key = config.get("api", {}).get("key")
    if not key:
        raise ValueError("API key not found in config.api.key")
    openai.api_key = key

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
