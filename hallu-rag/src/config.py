"""
Configuration loader for Hallucination-Aware RAG.
Reads from config.yaml and environment variables.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Configuration manager that loads from YAML and environment variables."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Find config.yaml relative to this file
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / "config.yaml"

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        # Override with environment variables
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to config."""
        # HF_TOKEN
        if os.getenv("HF_TOKEN"):
            self._config.setdefault("huggingface", {})["token"] = os.getenv("HF_TOKEN")

        # OPENAI_API_KEY
        if os.getenv("OPENAI_API_KEY"):
            self._config.setdefault("openai", {})["api_key"] = os.getenv("OPENAI_API_KEY")

        # DEVICE override
        if os.getenv("DEVICE"):
            self._config.setdefault("device", {})["type"] = os.getenv("DEVICE")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., "data.raw_dir")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self._config.get(section, {})

    @property
    def data(self) -> Dict[str, Any]:
        return self.get_section("data")

    @property
    def chunking(self) -> Dict[str, Any]:
        return self.get_section("chunking")

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self.get_section("retrieval")

    @property
    def generation(self) -> Dict[str, Any]:
        return self.get_section("generation")

    @property
    def faithfulness(self) -> Dict[str, Any]:
        return self.get_section("faithfulness")

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.get_section("evaluation")

    @property
    def app(self) -> Dict[str, Any]:
        return self.get_section("app")

    @property
    def device(self) -> Dict[str, Any]:
        return self.get_section("device")

    def get_device(self) -> str:
        """
        Get the device to use for computation.

        Returns:
            Device string: "cuda", "mps", or "cpu"
        """
        device_type = self.get("device.type", "auto")

        if device_type != "auto":
            return device_type

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


if __name__ == "__main__":
    # Smoke test
    cfg = get_config()
    print("Config loaded successfully")
    print(f"Data dir: {cfg.data.get('raw_dir')}")
    print(f"Device: {cfg.get_device()}")
    print(f"Chunking fixed size: {cfg.chunking.get('fixed', {}).get('chunk_size')}")
    print(f"Retrieval dense model: {cfg.retrieval.get('dense', {}).get('model')}")