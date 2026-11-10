import os
import yaml
from dotenv import load_dotenv


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config, load .env, and resolve generation env vars."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    load_dotenv()

    gen = config.get("generation", {})
    for key in ("base_url_env", "api_key_env", "model_env"):
        env_key = gen.get(key)
        if env_key:
            gen[key.replace("_env", "")] = os.getenv(env_key)

    return config


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"