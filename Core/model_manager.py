# Coser/Core/model_manager.py

import os
import json
import torch
import time
from typing import Dict, Any, Optional
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from Core import user  # 用于获取配置数据
from Core.schemas import ChatRequest


# --- 全局单例状态 ---
class ModelManager:
    """
    单例模型管理器，负责基础模型加载和 LoRA 权重切换。
    """
    _instance = None
    _model = None
    _tokenizer = None
    _current_config_id = None

    # 固定推理参数 (与 infer.py 保持一致)
    FIXED_TEMPERATURE = 0.4
    FIXED_MAX_LENGTH = 50
    FIXED_TOP_P = 0.9

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def _load_base_model(self, model_path: str):
        """仅在首次使用时加载基础模型和分词器。"""
        if self._model is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 首次加载基础模型：{model_path}")

            # 使用 torch.bfloat16 加载基础模型，确保在 GPU 上
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
            )

            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # 基础模型不需要 eval()，因为它将被 PeftModel 包装
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 基础模型加载完成。")

    def _load_lora_weights(self, username: str, config_id: str, config: Dict[str, Any]):
        """加载或切换到指定配置的 LoRA 权重。"""

        # 1. 检查是否已经加载
        if self._current_config_id == config_id:
            return

            # 2. 构造 LoRA 路径
        WEIGHT_SAVE_DIR_BASE = config['WEIGHT_PATH']
        LORA_WEIGHT_PATH = os.path.join(WEIGHT_SAVE_DIR_BASE, username, config_id, "final_lora_weights")

        if not os.path.isdir(LORA_WEIGHT_PATH):
            raise FileNotFoundError(f"LoRA 权重路径不存在：{LORA_WEIGHT_PATH}")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 切换/加载 LoRA 权重：{config_id}")

        # 3. 卸载旧权重 (如果是 PeftModel)
        if isinstance(self._model, PeftModel):
            # 使用 unwrap_model 卸载 PEFT 包装
            self._model = self._model.unload_and_uncache()

        # 4. 加载新 LoRA 权重
        self._model = PeftModel.from_pretrained(self._model, LORA_WEIGHT_PATH)
        self._model.eval()
        self._current_config_id = config_id
        print(f"[{datetime.now().strftime('%H:%M:%S')}] LoRA 权重切换完成到 {config_id}。")

    def generate_response(self, username: str, request: ChatRequest, config_data: Dict[str, Any]) -> str:
        """执行推理并返回结果。"""

        start_time = time.time()

        # 1. 确保模型和权重已加载/切换
        # 加载基础模型 (如果未加载)
        self._load_base_model(config_data['MODEL_PATH'])
        # 切换 LoRA 权重
        self._load_lora_weights(username, request.config_id, config_data)

        # 2. 格式化输入 (使用 infer.py 的逻辑)
        prompt = f"### Instruction:\n{request.prompt}\n\n### Response:\n"

        # 3. 分词
        tokenizer = self._tokenizer
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        # 移动到 GPU
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # 4. 合并生成参数
        max_new_tokens = request.max_length if request.max_length is not None else self.FIXED_MAX_LENGTH
        temperature = request.temperature if request.temperature is not None else self.FIXED_TEMPERATURE

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True if temperature > 0.0 else False,
            "top_p": self.FIXED_TOP_P,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # 5. 生成响应
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **generation_config
            )

        # 6. 解码并清理
        output_text = tokenizer.decode(
            output_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        end_time = time.time()
        return output_text.strip(), end_time - start_time


# 在 API 中初始化管理器实例
model_manager = ModelManager()