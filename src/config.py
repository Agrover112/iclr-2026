"""Configuration loading with CLI override support."""

import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str = "local") -> DictConfig:
    """
    Load config from environment-specific YAML file, then override with CLI arguments.

    Config hierarchy:
    1. configs/base.yaml (common settings)
    2. configs/{config_path}.yaml (environment-specific overrides)
    3. CLI arguments (highest priority)

    Args:
        config_path: Config name: 'local', 'slurm', or 'modal' (default: 'local')
                     Or full path to custom config file

    Returns:
        OmegaConf DictConfig with merged settings
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GRaM competition training")

    # Config arguments
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help=f"Config name ('local'|'slurm'|'modal') or path to config file (default: {config_path})"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data directory (overrides config)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        help="Training split ratio (overrides config)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        help="Validation split ratio (overrides config)"
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        help="Use only this fraction of total data (0.0-1.0, default 1.0)"
    )

    # Training arguments
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (overrides config)"
    )

    args = parser.parse_args()

    # Resolve config file path
    config_name = args.config
    if config_name in ['local', 'slurm', 'modal']:
        cfg_path = Path(f"configs/{config_name}.yaml")
    else:
        cfg_path = Path(config_name)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Load base config first
    base_path = Path("configs/base.yaml")
    if base_path.exists():
        cfg = OmegaConf.load(base_path)
        # Merge environment-specific config on top
        env_cfg = OmegaConf.load(cfg_path)
        cfg = OmegaConf.merge(cfg, env_cfg)
    else:
        # Fallback: just load the requested config
        cfg = OmegaConf.load(cfg_path)

    # Override with CLI arguments (only if provided)
    if args.data_path is not None:
        cfg.data.path = args.data_path

    if args.train_split is not None:
        cfg.data.train_split = args.train_split

    if args.val_split is not None:
        cfg.data.val_split = args.val_split

    if args.data_fraction is not None:
        cfg.data.fraction = args.data_fraction

    if args.seed is not None:
        cfg.training.seed = args.seed

    # Auto-detect environment
    _detect_environment(cfg)

    return cfg


def _detect_environment(cfg: DictConfig):
    """
    Auto-detect execution environment and set cfg.environment.

    Checks for SLURM and Modal environment variables.
    """
    if "SLURM_JOB_ID" in os.environ:
        cfg.environment = "slurm"
        print(f"[Config] Detected SLURM environment: job_id={os.environ['SLURM_JOB_ID']}")

    elif "MODAL_TASK_ID" in os.environ:
        cfg.environment = "modal"
        print(f"[Config] Detected Modal environment")

    else:
        cfg.environment = "local"
        print(f"[Config] Detected local environment")


def print_config(cfg: DictConfig, env: str = None):
    """Pretty-print configuration."""
    env_str = f" ({env})" if env else ""
    print("\n" + "=" * 60)
    print(f"Configuration{env_str}")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Example: load config
    cfg = load_config()
    env = cfg.get('environment', 'unknown')
    print_config(cfg, env=env)
