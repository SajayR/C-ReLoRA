"""
DinoV2 model with LoRA (Low-Rank Adaptation) integration for efficient fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2Config
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DinoV2WithClassificationHead(nn.Module):
    """
    DinoV2 model with a classification head for ImageNet classification.
    """
    
    def __init__(
        self, 
        model_name: str = "facebook/dinov2-base",
        num_classes: int = 1000,
        dropout: float = 0.1,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load DinoV2 backbone
        self.backbone = Dinov2Model.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("DinoV2 backbone frozen")
        
        # Get hidden size from config
        self.hidden_size = self.backbone.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes)
        )
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            pixel_values: Input images [batch_size, 3, height, width]
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Get features from DinoV2 backbone
        outputs = self.backbone(pixel_values=pixel_values)
        
        # Use CLS token representation
        cls_token_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(cls_token_output)
        
        return logits


class DinoV2LoRAClassifier(nn.Module):
    """
    DinoV2 model with LoRA adaptation for efficient fine-tuning.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        num_classes: int = 1000,
        lora_config: Optional[Dict[str, Any]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Create base model
        self.base_model = DinoV2WithClassificationHead(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=False  # We'll handle freezing with LoRA
        )
        
        # Default LoRA configuration
        if lora_config is None:
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": [
                    "query", "key", "value", "dense"
                ],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.FEATURE_EXTRACTION
            }
        
        # Apply LoRA to the backbone (ensure config is a regular dict)
        if hasattr(lora_config, 'to_container'):
            # Handle OmegaConf objects
            lora_config_dict = lora_config.to_container()
        elif hasattr(lora_config, '__dict__'):
            # Handle other object types
            lora_config_dict = lora_config.__dict__.copy()
        else:
            # Handle regular dicts and dict-like objects
            lora_config_dict = dict(lora_config)
        
        # Ensure target_modules is a regular list
        if 'target_modules' in lora_config_dict:
            target_modules = lora_config_dict['target_modules']
            if hasattr(target_modules, 'to_container'):
                lora_config_dict['target_modules'] = target_modules.to_container()
            elif not isinstance(target_modules, list):
                lora_config_dict['target_modules'] = list(target_modules)
        
        lora_config_obj = LoraConfig(**lora_config_dict)
        
        # Only apply LoRA to the backbone, not the classifier
        self.backbone_with_lora = get_peft_model(self.base_model.backbone, lora_config_obj)
        
        # Keep the classifier separate
        self.classifier = self.base_model.classifier
        
        # Freeze all non-LoRA parameters in backbone
        for name, param in self.backbone_with_lora.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
        
        logger.info(f"Applied LoRA with config: {lora_config}")
        self._print_trainable_parameters()
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LoRA-adapted model.
        
        Args:
            pixel_values: Input images [batch_size, 3, height, width]
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Get features from LoRA-adapted backbone
        outputs = self.backbone_with_lora(pixel_values=pixel_values)
        
        # Use CLS token representation
        cls_token_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(cls_token_output)
        
        return logits
    
    def _print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )
    
    def save_lora_weights(self, save_path: str):
        """Save only the LoRA weights."""
        self.backbone_with_lora.save_pretrained(save_path)
        logger.info(f"LoRA weights saved to {save_path}")
    
    def load_lora_weights(self, load_path: str):
        """Load LoRA weights."""
        # This would typically be done during initialization
        # by loading a pre-trained LoRA model
        logger.info(f"Loading LoRA weights from {load_path}")


def create_dinov2_lora_model(
    model_name: str = "facebook/dinov2-base",
    num_classes: int = 1000,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None
) -> DinoV2LoRAClassifier:
    """
    Factory function to create a DinoV2 model with LoRA adaptation.
    
    Args:
        model_name: HuggingFace model name for DinoV2
        num_classes: Number of output classes
        lora_r: LoRA rank
        lora_alpha: LoRA scaling parameter
        lora_dropout: LoRA dropout rate
        target_modules: List of modules to apply LoRA to
    
    Returns:
        DinoV2LoRAClassifier: Model ready for training
    """
    if target_modules is None:
        target_modules = ["query", "key", "value", "dense"]
    
    lora_config = {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": list(target_modules),  # Ensure it's a regular list
        "lora_dropout": lora_dropout,
        "bias": "none",
        "task_type": TaskType.FEATURE_EXTRACTION
    }
    
    model = DinoV2LoRAClassifier(
        model_name=model_name,
        num_classes=num_classes,
        lora_config=lora_config
    )
    
    return model


def get_model_size_info(model: nn.Module) -> Dict[str, int]:
    """Get information about model size and parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0
    }


if __name__ == "__main__":
    # Test model creation
    print("Testing DinoV2 LoRA model creation...")
    
    # Create model
    model = create_dinov2_lora_model(
        model_name="facebook/dinov2-base",
        num_classes=1000,
        lora_r=16,
        lora_alpha=32
    )
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # Print model info
    info = get_model_size_info(model)
    print(f"Model info: {info}")
