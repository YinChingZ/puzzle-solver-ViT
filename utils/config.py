import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理器，负责加载和处理配置文件"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        
    def load_config(self, config_type: str, name: str = "default") -> Dict[str, Any]:
        """加载指定类型和名称的配置文件"""
        config_path = os.path.join(self.config_dir, f"{config_type}_configs", f"{name}.json")
        
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        logger.info(f"Loaded {config_type} config '{name}' from {config_path}")
        return config
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并两个配置，override_config会覆盖base_config的值"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def get_experiment_config(self, model_name: str = "base", 
                             data_name: str = "default",
                             training_name: str = "default") -> Dict[str, Any]:
        """获取完整实验配置，合并模型、数据和训练配置"""
        model_config = self.load_config("model", model_name)
        data_config = self.load_config("data", data_name)
        training_config = self.load_config("training", training_name)
        
        # 合并所有配置
        config = {}
        config.update(model_config)
        config.update(data_config)
        config.update(training_config)
        
        return config
