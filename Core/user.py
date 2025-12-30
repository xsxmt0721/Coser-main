# Core/user.py

import json
import os
import shutil
from datetime import datetime
import sys
from typing import Optional, Dict, List, Any

# 假设 Core 目录在项目根目录 /home/user/Coser 下
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CORE_DIR)

# --- 存储路径定义 (必须与 service_cli.py 保持一致) ---
USERS_JSON_PATH = os.path.join(PROJECT_ROOT, "user_configs", "users.json")
BASE_CONFIG_DIR = os.path.join(PROJECT_ROOT, "user_configs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
CACHE_DIR = os.path.join(DATA_DIR, "Cache")

with open(os.path.join(PROJECT_ROOT, "settings.json"), 'r') as f:
    SETTINGS = json.load(f)

# --- 默认配置文件模板 ---
# 用户新建配置时使用此模板
DEFAULT_CONFIG_TEMPLATE = {
    "USERNAME": "",
    "PWD": "",
    "ID": "",
    "MODEL_PATH": SETTINGS["model_path"],
    "DATA_PATH": "",  # 将来指向 DATA_DIR/USERNAME/dataset.json
    "WEIGHT_PATH": WEIGHTS_DIR,
    "DEFAULT_LORA_PARAMS": {
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "batch_size": 4,
        "learning_rate": 0.0002,
        "epochs": 3,
        "max_seq_len": 1024
    }
}


def _load_user_data() -> List[Dict[str, str]]:
    """加载 users.json 文件内容"""
    if not os.path.exists(USERS_JSON_PATH):
        return []
    try:
        with open(USERS_JSON_PATH, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"警告：users.json 文件损坏或为空，已重置。")
        return []


def _save_user_data(data: List[Dict[str, str]]):
    """保存用户数据到 users.json"""
    os.makedirs(os.path.dirname(USERS_JSON_PATH), exist_ok=True)
    with open(USERS_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=4)


def check_username_exists(username: str) -> bool:
    """检查用户名是否已注册"""
    users = _load_user_data()
    return any(u["USERNAME"].lower() == username.lower() for u in users)


def register_user(username: str, pwd: str) -> Optional[str]:
    """
    注册用户，创建目录结构，并写入 users.json。
    成功返回 None，失败返回错误信息。
    """
    if check_username_exists(username):
        return "用户名已存在。"

    # 1. 创建目录结构
    dirs_to_create = [
        os.path.join(DATA_DIR, username),
        os.path.join(WEIGHTS_DIR, username),
        os.path.join(LOGS_DIR, username),
        os.path.join(BASE_CONFIG_DIR, username),
    ]
    try:
        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)

        # 2. 写入 users.json
        users = _load_user_data()
        users.append({"USERNAME": username, "PWD": pwd})
        _save_user_data(users)

        return None  # 注册成功

    except Exception as e:
        # 清理已创建的目录
        for d in dirs_to_create:
            if os.path.exists(d):
                # 确保只删除空目录或安全删除
                if os.path.isdir(d) and not os.listdir(d):
                    os.rmdir(d)
        return f"注册失败：系统错误 {e}"


def login_user(username: str, pwd: str) -> Optional[Dict[str, Any]]:
    """
    用户登录：检查用户名和密码。
    成功返回用户配置信息（Dict），失败返回 None。
    """
    users = _load_user_data()
    user_info = next((u for u in users if u["USERNAME"] == username and u["PWD"] == pwd), None)

    if user_info:
        # 登录成功，读取并返回用户的配置列表
        return get_user_config_data(username)
    else:
        return None


def get_user_config_data(username: str) -> Dict[str, Any]:
    """读取用户配置目录下的所有配置，并返回一个字典"""
    user_config_path = os.path.join(BASE_CONFIG_DIR, username)
    config_data = {}

    if not os.path.isdir(user_config_path):
        # 理论上注册时已创建，但以防万一
        os.makedirs(user_config_path, exist_ok=True)

    for filename in os.listdir(user_config_path):
        if filename.endswith('.json'):
            config_id = filename[:-5]
            file_path = os.path.join(user_config_path, filename)
            try:
                with open(file_path, 'r') as f:
                    config_data[config_id] = json.load(f)
            except Exception as e:
                print(f"警告：加载配置 {filename} 失败: {e}")

    return config_data


def save_config_file(username: str, config_id: str, new_config: Dict[str, Any]) -> None:
    """保存修改后的配置到用户的配置文件中"""
    file_path = os.path.join(BASE_CONFIG_DIR, username, f"{config_id}.json")
    with open(file_path, 'w') as f:
        json.dump(new_config, f, indent=4)


def create_new_config(username: str, config_id: str) -> Optional[str]:
    """根据模板创建新的配置文件"""
    if not config_id.isalnum():
        return "配置ID必须是字母和数字。"

    config_data = get_user_config_data(username)
    if config_id in config_data:
        return f"配置ID '{config_id}' 已存在。"

    new_config = DEFAULT_CONFIG_TEMPLATE.copy()
    new_config["USERNAME"] = username
    # 将用户密码写入配置（用于 finetune.py 内部校验，虽然现在没用，但符合需求）
    user_info = next((u for u in _load_user_data() if u["USERNAME"] == username), None)
    if user_info:
        new_config["PWD"] = user_info["PWD"]

    new_config["ID"] = config_id
    new_config["DATA_PATH"] = os.path.join(DATA_DIR, username, f"{config_id}_dataset.json")

    try:
        save_config_file(username, config_id, new_config)
        return None
    except Exception as e:
        return f"创建配置失败: {e}"


def upload_dataset(username: str, dataset_name: str) -> Optional[str]:
    """
    【新逻辑】
    从 Cache 目录中查找以 {username}.{dataset_name}.json 命名的文件，
    将其导入到用户个人目录，并删除 Cache 中的文件。
    """
    # 1. 构造 Cache 中期望的文件名
    cache_filename = f"{username}.{dataset_name}.json"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # 2. 构造用户个人目录的目标路径 (移除前缀)
    destination_path = os.path.join(DATA_DIR, username, f"{dataset_name}.json")

    # 3. 检查 Cache 文件是否存在
    if not os.path.exists(cache_path):
        return f"Cache 中找不到匹配文件：'{cache_filename}'。请确保文件已到位。"
    if not cache_path.lower().endswith('.json'):
        return "Cache 文件格式错误，只支持 JSON 文件。"

    try:
        # 4. 执行文件移动 (os.rename 或 shutil.move)
        # 使用 shutil.move 确保跨设备操作的健壮性
        shutil.move(cache_path, destination_path)

        return None

    except Exception as e:
        # 如果移动失败，可能 Cache 文件没有被删除，但此处不处理回滚
        return f"数据集导入失败：系统错误 {e}"


def list_user_datasets(username: str) -> List[str]:
    """列出用户 data/USERNAME/ 目录下的所有数据集"""
    user_data_path = os.path.join(DATA_DIR, username)
    if not os.path.isdir(user_data_path):
        return []
    return [f for f in os.listdir(user_data_path) if f.endswith('.json')]


def delete_user_dataset(username: str, dataset_name: str) -> Optional[str]:
    """删除用户 data/USERNAME/ 目录下的指定数据集"""
    file_path = os.path.join(DATA_DIR, username, f"{dataset_name}.json")

    if not os.path.exists(file_path):
        return f"数据集 '{dataset_name}.json' 不存在。"

    try:
        os.remove(file_path)
        return None
    except Exception as e:
        return f"删除失败: {e}"


def delete_user_config(username: str, config_id: str) -> Optional[str]:
    """
    删除用户的指定配置文件及其关联的模型权重目录。
    """

    # 1. 构造配置文件的路径
    config_file_path = os.path.join(BASE_CONFIG_DIR, username, f"{config_id}.json")

    # 2. 构造模型权重的目录路径
    weight_dir_path = os.path.join(WEIGHTS_DIR, username, config_id)

    # --- A. 检查并删除配置文件 ---
    if not os.path.exists(config_file_path):
        return f"配置ID '{config_id}' 不存在。"

    try:
        os.remove(config_file_path)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 成功删除配置文件: {config_file_path}")
    except Exception as e:
        return f"删除配置文件失败: {e}"

    # --- B. 检查并删除模型权重目录 ---
    if os.path.isdir(weight_dir_path):
        try:
            # 使用 shutil.rmtree 递归删除目录及其内容
            shutil.rmtree(weight_dir_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 成功删除模型权重目录: {weight_dir_path}")
        except Exception as e:
            # 如果配置已删除，但权重删除失败，仍然返回错误信息
            return f"配置文件已删除，但模型权重目录删除失败: {e}"

    return None  # 删除成功


if __name__ == '__main__':
    print("Core/user.py 模块，用于提供用户和配置管理函数。")