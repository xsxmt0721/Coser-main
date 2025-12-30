# scripts/simple_test_finetune.py

import os
import sys
import subprocess
import json
from datetime import datetime

# --- 设定路径 ---
# 假设脚本在 scripts/ 下，Core 在其上一级目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(PROJECT_ROOT, 'Core')
FINETUNE_SCRIPT = os.path.join(CORE_DIR, 'finetune.py')

# --- 设定您真实的配置和预期路径 ---
# 请根据您的实际环境调整
CONFIG_PATH = os.path.join(PROJECT_ROOT, "user_configs/example/firefly.json")

def get_expected_paths(config_path):
    """从配置文件中解析预期的保存路径"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        weight_base = config['WEIGHT_PATH']
        id_name = config['ID']

        # 预期的最终权重保存路径: WEIGHT_PATH/ID/final_lora_weights
        expected_weight_path = os.path.join(weight_base, id_name, "final_lora_weights")
        return expected_weight_path

    except FileNotFoundError:
        print(f"🚨 错误：找不到配置文件: {config_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"🚨 错误：配置文件缺少关键字段: {e}")
        sys.exit(1)


# --- 执行微调测试 ---
def run_finetune_test():
    """以命令行方式调用 Core/finetune.py"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 开始执行微调脚本...")
    print(f"配置文件: {CONFIG_PATH}")

    # 构造命令行调用命令
    command = [
        sys.executable,  # 使用当前的 python 解释器
        FINETUNE_SCRIPT,
        "--config",
        CONFIG_PATH
    ]

    try:
        # 使用 subprocess.run 执行脚本。
        # check=True: 如果返回码非零，则抛出异常
        # capture_output=False: 允许训练进度信息直接流向终端
        result = subprocess.run(command, check=True, capture_output=False)

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🎉 Core/finetune.py 执行完成，返回码: {result.returncode}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"🚨 严重错误：Core/finetune.py 执行失败。")
        print(f"错误信息请查看终端输出的 Traceback。")
        return False
    except FileNotFoundError:
        print(f"🚨 错误：找不到微调脚本文件: {FINETUNE_SCRIPT}。")
        return False


# --- 验证结果 ---
def verify_results(expected_path):
    """验证 LoRA 权重是否成功保存"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔍 验证结果...")

    if os.path.isdir(expected_path) and len(os.listdir(expected_path)) > 0:
        # 检查关键文件是否存在
        files_exist = (
                os.path.exists(os.path.join(expected_path, 'adapter_config.json')) and
                any(f.startswith('adapter_model') for f in os.listdir(expected_path))
        )

        if files_exist:
            print(f"✅ 权重验证成功！LoRA 权重已保存到配置指定的路径:")
            print(f"   --> {expected_path}")
            return True
        else:
            print(f"⚠️ 权重目录存在 ({expected_path})，但关键文件缺失。请检查 finetune.py 中的保存逻辑。")
            return False
    else:
        print(f"❌ 权重目录不存在或为空：{expected_path}")
        print("   --> 请检查 finetune.py 是否成功运行到保存步骤。")
        return False


# --- 主函数 ---
if __name__ == '__main__':
    print("=========================================")
    print("      🚀 LoRA 微调配置和保存功能测试     ")
    print("=========================================")

    # 0. 预检查
    if not os.path.exists(CONFIG_PATH):
        print(f"🚨 错误：配置文件不存在！请检查路径：{CONFIG_PATH}")
        sys.exit(1)

    # 获取预期路径
    expected_path = get_expected_paths(CONFIG_PATH)

    # 1. 执行测试
    if run_finetune_test():
        # 2. 验证结果
        if verify_results(expected_path):
            print("\n=========================================")
            print("     🎉 微调核心功能 (配置读取/保存) 测试通过！")
            print("=========================================")
        else:
            print("\n=========================================")
            print("     ❌ 微调功能失败。请检查错误信息。")
            print("=========================================")
    else:
        print("\n=========================================")
        print("     ❌ 微调脚本执行失败。请检查错误信息。")
        print("=========================================")