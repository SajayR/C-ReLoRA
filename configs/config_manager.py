"""
Configuration management for DinoV2 LoRA training experiments.
"""

import os
import yaml
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Main configuration class for experiments."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from dictionary."""
        self._config = OmegaConf.create(config_dict)
    
    def __getattr__(self, name: str) -> Any:
        """Allow dot notation access to config values."""
        return getattr(self._config, name)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access to config values."""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return OmegaConf.select(self._config, key, default=default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return OmegaConf.to_container(self._config)
    
    def save(self, path: str):
        """Save config to file."""
        with open(path, 'w') as f:
            OmegaConf.save(self._config, f)
        logger.info(f"Config saved to {path}")


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = config_dir
    
    def load_config(
        self, 
        config_name: str = "default",
        overrides: Optional[Dict[str, Any]] = None
    ) -> ExperimentConfig:
        """
        Load configuration from YAML file with optional overrides.
        
        Args:
            config_name: Name of config file (without .yaml extension)
            overrides: Dictionary of values to override in config
        
        Returns:
            ExperimentConfig: Loaded and validated configuration
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")
        
        # Load base config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Apply overrides if provided
        if overrides:
            config_dict = self._merge_configs(config_dict, overrides)
        
        # Validate and create config object
        config = ExperimentConfig(config_dict)
        self._validate_config(config)
        
        logger.info(f"Loaded config from {config_path}")
        if overrides:
            logger.info(f"Applied overrides: {overrides}")
        
        return config
    
    def _merge_configs(self, base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override values into base config."""
        base_conf = OmegaConf.create(base_config)
        override_conf = OmegaConf.create(overrides)
        merged = OmegaConf.merge(base_conf, override_conf)
        return OmegaConf.to_container(merged)
    
    def _validate_config(self, config: ExperimentConfig):
        """Validate configuration values."""
        # Validate required fields exist
        required_fields = [
            "data.dataset_path",
            "model.name", 
            "model.num_classes",
            "training.epochs",
            "training.learning_rate"
        ]
        
        for field in required_fields:
            if config.get(field) is None:
                raise ValueError(f"Required config field {field} is missing")
        
        # Validate data path exists
        if not os.path.exists(config.data.dataset_path):
            raise ValueError(f"Dataset path {config.data.dataset_path} does not exist")
        
        # Validate model name
        valid_models = [
            "facebook/dinov2-small",
            "facebook/dinov2-base", 
            "facebook/dinov2-large"
        ]
        if config.model.name not in valid_models:
            logger.warning(f"Model {config.model.name} not in standard list: {valid_models}")
        
        # Validate LoRA parameters
        if config.lora.r <= 0:
            raise ValueError("LoRA rank must be positive")
        
        if config.lora.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        
        # Validate training parameters (convert to float if needed)
        try:
            lr = float(config.training.learning_rate)
            if lr <= 0:
                raise ValueError("Learning rate must be positive")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid learning rate: {config.training.learning_rate}")
        
        try:
            epochs = int(config.training.epochs)
            if epochs <= 0:
                raise ValueError("Number of epochs must be positive")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid number of epochs: {config.training.epochs}")
        
        logger.info("Configuration validation passed")
    
    def create_experiment_config(
        self,
        base_config: str = "default",
        experiment_name: str = None,
        **kwargs
    ) -> ExperimentConfig:
        """
        Create a new experiment configuration with custom overrides.
        
        Args:
            base_config: Base configuration to use
            experiment_name: Name for the experiment
            **kwargs: Additional config overrides
        
        Returns:
            ExperimentConfig: Configured experiment
        """
        overrides = kwargs.copy()
        
        if experiment_name:
            overrides["logging.run_name"] = experiment_name
        
        return self.load_config(base_config, overrides)


def create_config_variants():
    """Create configuration variants for different experiments."""
    base_config_path = "./configs/default.yaml"
    
    # Small model variant
    small_config = {
        "model": {"name": "facebook/dinov2-small"},
        "data": {"batch_size": 64},
        "training": {"learning_rate": 2e-4}
    }
    
    # Large model variant  
    large_config = {
        "model": {"name": "facebook/dinov2-large"},
        "data": {"batch_size": 16},
        "training": {"learning_rate": 5e-5}
    }
    
    # High LoRA rank variant
    high_rank_config = {
        "lora": {"r": 64, "alpha": 128},
        "training": {"learning_rate": 5e-5}
    }
    
    # Different LoRA targets
    full_lora_config = {
        "lora": {
            "target_modules": [
                "query", "key", "value", "dense",
                "intermediate.dense", "output.dense"
            ]
        }
    }
    
    variants = {
        "small": small_config,
        "large": large_config, 
        "high_rank": high_rank_config,
        "full_lora": full_lora_config
    }
    
    # Save variant configs
    os.makedirs("./configs", exist_ok=True)
    
    with open(base_config_path, 'r') as f:
        base = yaml.safe_load(f)
    
    for name, overrides in variants.items():
        variant_config = ConfigManager()._merge_configs(base, overrides)
        
        variant_path = f"./configs/{name}.yaml"
        with open(variant_path, 'w') as f:
            yaml.dump(variant_config, f, default_flow_style=False, indent=2)
        
        print(f"Created variant config: {variant_path}")


if __name__ == "__main__":
    # Test configuration management
    config_manager = ConfigManager()
    
    # Test loading default config
    config = config_manager.load_config("default")
    print("Loaded default config successfully")
    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.data.dataset_path}")
    print(f"LoRA rank: {config.lora.r}")
    
    # Test overrides
    config_with_overrides = config_manager.load_config(
        "default",
        overrides={
            "training.learning_rate": 2e-4,
            "lora.r": 32,
            "logging.run_name": "test_run"
        }
    )
    print(f"Override LR: {config_with_overrides.training.learning_rate}")
    print(f"Override LoRA rank: {config_with_overrides.lora.r}")
    
    # Create variant configs
    create_config_variants()
