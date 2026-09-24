"""CLI entrypoint for a config-driven synthetic TLS demo run."""

import argparse
import json
import os
import sys

# Ensure SyntheticPipeline root is on sys.path when run as a script.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from synthetic_tls.orchestrate import run_tile


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        print("Please provide a valid configuration file using --config.")
        sys.exit(1)
    with open(config_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in config file {config_path}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Lidar Pipeline Runner")
    parser.add_argument("--config", type=str, help="Path to a specific configuration JSON file.")
    parser.add_argument(
        "--config-dir",
        type=str,
        help="Directory containing pipeline_config.json.",
    )

    args = parser.parse_args()

    config_path = None
    if args.config:
        config_path = args.config
    elif args.config_dir:
        config_path = os.path.join(args.config_dir, "pipeline_config.json")
    else:
        default_config = os.path.join(_ROOT, "configs", "pipeline_config.json")
        if os.path.exists(default_config):
            print(f"No config specified, defaulting to: {default_config}")
            config_path = default_config
        else:
            print("Error: No config provided and default configs/pipeline_config.json not found.")
            parser.print_help()
            sys.exit(1)

    config = load_config(config_path)
    run_tile(config)
