import argparse
import os
import sys
import toml
from scheduler.scheduler import Scheduler


def load_config(model_path=None, config_path=None):
    """Load config from an explicit config path, model_path/efmnode.toml, or default config.toml.

    When model_path is provided:
      - Use explicit config_path first when provided
      - Use <model_path>/efmnode.toml if it exists
      - Otherwise fall back to default config.toml with a warning
      - Always override ckpt_dir to point to model_path
    """
    default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")

    if config_path is not None:
        print(f"[INFO] Loading config from: {config_path}")
        config = toml.load(config_path)
    elif model_path is not None:
        model_config_path = os.path.join(model_path, "efmnode.toml")
        if os.path.isfile(model_config_path):
            print(f"[INFO] Loading config from: {model_config_path}")
            config = toml.load(model_config_path)
        else:
            print(f"[WARNING] {model_config_path} not found, falling back to default config.toml", file=sys.stderr)
            config = toml.load(default_config_path)
    else:
        config = toml.load(default_config_path)

    if model_path is not None:
        config.setdefault("model", {})
        config["model"]["ckpt_dir"] = model_path
        print(f"[INFO] Model checkpoint dir: {model_path}")

    return config


def main():
    parser = argparse.ArgumentParser(description="EFMNode inference client")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Absolute path to model directory (overrides ckpt_dir in config)")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to a TOML config file (overrides default config discovery)")
    args = parser.parse_args()

    config = load_config(args.model_path, args.config_path)
    scheduler = Scheduler(config)
    scheduler.run()


if __name__ == "__main__":
    main()
