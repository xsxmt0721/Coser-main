# Coser/Core/scheduler.py

import os
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

# 导入配置和路径信息
from Core import user

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CORE_DIR)

with open(os.path.join(PROJECT_ROOT, "settings.json"), 'r') as f:
    SETTINGS = json.load(f)
TRAIN_SCRIPT_PATH = os.path.join(user.CORE_DIR, "finetune.py")
LOG_DIR = user.LOGS_DIR  # 训练日志存放的基础目录
PYTHON_BIN = SETTINGS["python_bin"]  # Python 可执行文件的绝对路径

# 内存中的任务状态追踪器
# 结构: {'hutao': {'pid': 12345, 'status': 'RUNNING', 'start_time': '...'}}
ACTIVE_TRAINING_TASKS: Dict[str, Dict[str, Any]] = {}


def get_config_with_overrides(config_id: str, username: str, overrides: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    根据配置ID加载文件，并应用临时的LoRA参数覆盖。
    """
    all_configs = user.get_user_config_data(username)
    if config_id not in all_configs:
        return None

    config = all_configs[config_id]

    # 深度拷贝，避免修改原始配置对象
    merged_config = config.copy()
    lora_params = merged_config['DEFAULT_LORA_PARAMS'].copy()

    # 应用参数覆盖 (只覆盖非 None 的值)
    for key, value in overrides.items():
        if value is not None and key in lora_params:
            lora_params[key] = value

    merged_config['DEFAULT_LORA_PARAMS'] = lora_params

    # 增加任务ID和时间戳
    task_id = f"{config_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    merged_config['TASK_ID'] = task_id

    return merged_config


def is_process_running(pid: int) -> bool:
    """检查进程 ID (PID) 是否仍在运行。"""
    if pid is None:
        return False
    try:
        # 向进程发送一个信号 0，用于检查进程是否存在
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_training_task(username: str, config_id: str, merged_config: Dict[str, Any]) -> str:
    """
    【新】使用 subprocess.Popen 启动后台训练进程，更安全地处理 I/O 重定向。
    返回 task_id。
    """
    task_id = merged_config['TASK_ID']

    # 1. 准备日志文件路径
    user_log_dir = os.path.join(LOG_DIR, username)
    os.makedirs(user_log_dir, exist_ok=True)
    log_file_path = os.path.join(user_log_dir, f"train_{task_id}.log")

    # 2. 将最终配置写入临时文件
    temp_config_path = os.path.join("/tmp", f"train_config_{task_id}.json")
    with open(temp_config_path, 'w') as f:
        json.dump(merged_config, f)

    # 3. 构造 Popen 命令列表
    cmd = [
        PYTHON_BIN,  # 绝对路径 Python
        TRAIN_SCRIPT_PATH,
        "--config",
        temp_config_path
    ]

    # 4. 执行命令 (使用 Popen 启动)
    try:
        # 以写模式打开日志文件，用于重定向 stdout/stderr
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # 将标准错误也重定向到日志文件
                cwd=user.CORE_DIR,  # 设置工作目录为 Core
                start_new_session=True  # 启动新会话，防止与 API 进程关联
            )
            pid = process.pid

    except Exception as e:
        # 如果启动失败，清除临时配置文件
        os.remove(temp_config_path)
        raise Exception(f"训练进程启动失败: {e}")

    # 5. 更新内存中的状态
    ACTIVE_TRAINING_TASKS[config_id] = {
        'task_id': task_id,
        'status': 'RUNNING',
        'log_path': log_file_path,
        'start_time': datetime.now().isoformat(),
        'pid': pid,  # 记录 PID，用于状态检查
    }

    return task_id


def get_training_task_status(config_id: str) -> Dict[str, Any]:
    """
    【新】获取指定配置的最新任务状态和日志内容 (基于 PID 检查)。
    """
    if config_id not in ACTIVE_TRAINING_TASKS:
        return {'status': 'NOT_FOUND', 'log_content': '无活动训练任务记录。'}

    task_info = ACTIVE_TRAINING_TASKS[config_id]
    log_path = task_info['log_path']
    task_info['log_content'] = ""
    pid = task_info.get('pid')

    # 1. 更新进程状态
    if task_info['status'] == 'RUNNING' and not is_process_running(pid):
        # 如果之前是 RUNNING 但现在进程消失了，说明训练已完成或崩溃
        # 我们需要检查日志来确认 COMPLETED 或 FAILED
        task_info['status'] = 'COMPLETED_CHECK'  # 标记需要检查日志

    # 2. 读取日志文件的内容
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.readlines()
                task_info['log_content'] = "".join(content[-100:])

                # 3. 检查日志内容以确定最终状态
            if task_info['status'] == 'COMPLETED_CHECK':
                if "整个微调任务成功完成！" in task_info['log_content']:
                    task_info['status'] = 'COMPLETED'
                    task_info['completion_time'] = datetime.now().isoformat()
                elif "错误" in task_info['log_content'] or "exception" in task_info['log_content'].lower():
                    task_info['status'] = 'FAILED'
                else:
                    # 进程意外终止，但日志中没有明确的完成或错误信息
                    task_info['status'] = 'FAILED (Crash)'

        except Exception as e:
            task_info['log_content'] = f"读取日志文件失败: {e}"
            task_info['status'] = 'ERROR'
    else:
        task_info['log_content'] = "日志文件尚未生成或训练已结束。"

    return task_info