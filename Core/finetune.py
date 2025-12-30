import os
import sys

# --- 解决 ModuleNotFoundError 的patch -----------
# 1. 获取当前脚本 Core/ 目录的绝对路径
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 获取项目根目录 Coser/ 的绝对路径
PROJECT_ROOT = os.path.dirname(CORE_DIR)
# 3. 将项目根目录添加到 Python 模块搜索路径
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --------------------------------------------------

import argparse
import json
from datetime import datetime
from datasets import Dataset

# 引入必要的Hugging Face库
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling  # 用于处理数据批次
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from Core.utils import preprocess_dataset


# --- 辅助函数：将JSON数据格式化为模型可训练的文本格式 ---
def format_data(example):
    """
    将单条数据格式化为模型训练所需的输入文本。
    采用标准的指令微调模板，适应 DeepSeek-R1-8B Distill Llama。
    """
    # 构造 Alpaca/ChatML 风格的指令模板
    prompt = f"### Instruction:\n{example['instruction']}\n\n"
    if example['input']:
        prompt += f"### Input:\n{example['input']}\n\n"
    prompt += f"### Response:\n{example['output']}"

    # 确保没有多余的换行或空格
    return {"text": prompt.strip()}


# --- 核心微调函数 ---
def finetune_model(config_path: str):
    """
    根据配置路径加载信息，执行 QLoRA 微调，并保存 LoRA 权重。

    Args:
        config_path: 用户临时配置文件的绝对路径 (e.g., /tmp/train_conf_Trailblazer_firefly_1234.json)
    """

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始加载配置：{config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"错误：无法加载配置文件 {config_path}. 详细信息: {e}")
        return

    # --- 1. 从配置中提取路径和参数 ---
    MODEL_PATH = config['MODEL_PATH']
    DATASET_FILE = config['DATA_PATH']  # 直接指向 .json 文件
    WEIGHT_SAVE_DIR_BASE = config['WEIGHT_PATH']
    USERNAME = config['USERNAME']
    ID_NAME = config['ID']
    LORA_PARAMS = config['DEFAULT_LORA_PARAMS']

    # 最终权重保存路径: WEIGHT_PATH/ID/final_lora_weights
    WEIGHT_SAVE_DIR = os.path.join(WEIGHT_SAVE_DIR_BASE, USERNAME, ID_NAME)

    # 检查数据集文件是否存在
    if not os.path.exists(DATASET_FILE):
        print(f"错误：数据集文件 {DATASET_FILE} 不存在。请检查 DATA_PATH。")
        return

    # --- 2. 加载数据集和分词器 (Tokenizer) ---
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📚 正在加载数据集...")

    # 检查数据集文件是否存在
    if not os.path.exists(DATASET_FILE):
        print(f"错误：数据集文件 {DATASET_FILE} 不存在。请检查 DATA_PATH。")
        return

    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"错误：无法加载或解析 JSON 数据文件 {DATASET_FILE}. 详细信息: {e}")
        return

    cleaned_data = preprocess_dataset(raw_data)

    if not cleaned_data:
        print("错误：预处理后数据集为空。训练终止。")
        return

    dataset = Dataset.from_list(cleaned_data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 应用格式化函数，将JSON数据转换为训练所需的文本
    # 注意：这里我们不移除列，因为我们稍后需要 DataCollator 来处理
    dataset = dataset.map(format_data, remove_columns=dataset.column_names)

    # 分词处理
    def tokenize_function(examples):
        # 截断到最大序列长度
        tokenized = tokenizer(
            examples["text"],
            max_length=LORA_PARAMS['max_seq_len'],
            truncation=True,
            # 不进行 padding，交给 DataCollator 负责
        )
        # 将输入ID和标签ID分开，DataCollatorForLanguageModeling 只需要 input_ids
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 划分训练集
    train_dataset = tokenized_dataset

    print(f"数据加载完成。训练样本数: {len(train_dataset)}")

    # --- 3. 配置 QLoRA (4bit Quantization) ---
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 配置 QLoRA 和 4bit 量化...")

    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16  # A5000 支持 bfloat16，推荐使用
    )

    # LoRA 配置
    lora_config = LoraConfig(
        r=LORA_PARAMS['lora_r'],
        lora_alpha=LORA_PARAMS['lora_alpha'],
        lora_dropout=LORA_PARAMS['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        # 增加更多 Llama/DeepSeek 的关键层
    )

    # --- 4. 加载模型 ---
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # 预处理模型并注入 LoRA 配置
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    #

    # --- 5. 配置训练参数 (TrainingArguments) ---
    # checkpoint_dir 是最终权重保存的父目录
    checkpoint_dir = os.path.join(WEIGHT_SAVE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=LORA_PARAMS['epochs'],
        per_device_train_batch_size=LORA_PARAMS['batch_size'],
        learning_rate=LORA_PARAMS['learning_rate'],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        fp16=False,
        bf16=True,
        report_to="none",
        disable_tqdm=False,  # 启用进度条
    )

    # --- 6. 启动训练 ---
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  开始微调模型...")

    # 使用 Data Collator，它可以动态填充 batch 中的序列到最长长度
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal Language Modeling (CLM) 设置为 False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 训练主循环
    trainer.train()

    # --- 7. 保存最终 LoRA 权重 ---
    final_save_path = os.path.join(WEIGHT_SAVE_DIR, "final_lora_weights")
    os.makedirs(final_save_path, exist_ok=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 微调完成，正在保存最终 LoRA 权重到：{final_save_path}")

    # 必须保存模型的 LoRA 适配器和分词器
    trainer.model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    # --- 8. 清理临时配置文件 ---
    try:
        if os.path.exists(config_path):
            os.remove(config_path)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已清理临时配置文件: {config_path}")
    except Exception as e:
        # 清理失败不影响训练结果，仅打印警告
        print(f"警告：清理临时配置文件失败 {config_path}. 详细信息: {e}")

    print("整个微调任务成功完成！")


# --- 命令行入口点 ---
if __name__ == '__main__':
    # **关键修改：使脚本能够通过命令行接收配置路径**
    parser = argparse.ArgumentParser(description="Deep Learning Model Finetuning Script.")
    parser.add_argument('--config', type=str, required=True, help="微调配置文件的绝对路径。")

    args = parser.parse_args()

    # 调用核心函数
    finetune_model(args.config)