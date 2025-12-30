# Core/infer.py

import argparse
import json
import os
import sys
import torch
from datetime import datetime

# 引入 Hugging Face 库
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel


# --- 辅助函数：格式化对话输入 ---
def format_chat_input(instruction: str, history: list = None) -> str:
    """
    将用户指令格式化为模型训练时接受的 Alpaca 风格的指令模板。
    """
    # 假设推理时只处理单轮对话，不考虑 history，与训练数据保持一致
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt


# --- 核心推理函数 ---
def chat_with_model(config_path: str):
    """
    加载微调后的 LoRA 权重，并启动交互式对话。

    Args:
        config_path: 用户配置文件的绝对路径。
    """

    # --- 固定推理生成参数 ---
    FIXED_TEMPERATURE = 0.4  # 较低的温度，使输出更保守、更连贯
    FIXED_MAX_LENGTH = 50  # 较短的输出长度，避免模型跑飞
    FIXED_TOP_P = 0.9  # 保持 Top-P 采样
    # ---------------------------

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始加载配置：{config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"错误：无法加载配置文件 {config_path}. 详细信息: {e}")
        return

    # --- 1. 从配置中提取路径 ---
    MODEL_PATH = config['MODEL_PATH']
    WEIGHT_SAVE_DIR_BASE = config['WEIGHT_PATH']
    USERNAME = config['USERNAME']
    ID_NAME = config['ID']
    LORA_WEIGHT_PATH = os.path.join(WEIGHT_SAVE_DIR_BASE, USERNAME, ID_NAME, "final_lora_weights")

    # --- 2. 权重检查 ---
    if not os.path.isdir(LORA_WEIGHT_PATH):
        print(f"\n错误：角色权重 '{ID_NAME}' 不存在！")
        print(f"预期权重路径: {LORA_WEIGHT_PATH}")
        print("请先运行微调任务 (action=train) 才能进行对话。")
        return

    print(f"找到权重：{LORA_WEIGHT_PATH}")
    print(f"使用基础模型：{MODEL_PATH}")

    # --- 3. 配置模型加载 (Base Model + LoRA) ---
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在加载基础模型和分词器...")

    # 使用 torch.bfloat16 加载基础模型，确保在 GPU 上
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4. 加载 LoRA 权重 ---
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在加载 LoRA 权重...")
    model = PeftModel.from_pretrained(model, LORA_WEIGHT_PATH)
    model.eval()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 模型准备就绪。")

    # --- 5. 启动对话循环 ---
    print("\n=========================================")
    print(f"     正在与角色 [{ID_NAME}] 对话")
    print("     输入 'exit' 或 'quit' 退出")
    print(f"     [固定参数: Temp={FIXED_TEMPERATURE}, MaxLen={FIXED_MAX_LENGTH}]")
    print("=========================================")

    while True:
        try:
            user_input = input(f"[{ID_NAME} 提问] > ")

            if user_input.lower() in ['exit', 'quit']:
                print(f"[{ID_NAME}] > 期待下次再见！")
                break

            if not user_input.strip():
                continue

            # 格式化输入
            prompt = format_chat_input(user_input)

            # 分词
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 生成配置：使用固定的优化参数
            generation_config = {
                "max_new_tokens": FIXED_MAX_LENGTH,
                "temperature": FIXED_TEMPERATURE,
                "do_sample": True if FIXED_TEMPERATURE > 0.0 else False,
                "top_p": FIXED_TOP_P,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            # 生成响应
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    **generation_config
                )

            # 解码并清洗输出
            output_text = tokenizer.decode(output_ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            print(f"[{ID_NAME} 回复] > {output_text.strip()}")

        except Exception as e:
            print(f"\n生成响应时发生错误: {e}")
            break


# --- 命令行入口点 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Learning Model Inference Script.")
    # 只需要配置路径参数
    parser.add_argument('--config', type=str, required=True, help="推理配置文件的绝对路径。")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("错误：未检测到 CUDA GPU！模型加载需要 GPU 资源。")
        sys.exit(1)

    # 直接调用 chat_with_model，不再传递额外的参数
    chat_with_model(args.config)