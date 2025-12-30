# Core/service_cli.py

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

# 导入 Core/user.py 模块
# 解决 ModuleNotFoundError: No module named 'Core' 的问题
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
# ----------------------------------------------------
from Core import user  # 导入 user.py 模块
from Core.user import (
    BASE_CONFIG_DIR,
    LOGS_DIR,
    get_user_config_data,
    save_config_file
)

# ----------------------------------------------------

# 核心功能脚本路径
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNE_SCRIPT = os.path.join(CORE_DIR, 'finetune.py')
INFER_SCRIPT = os.path.join(CORE_DIR, 'infer.py')

# --- 全局状态变量 ---
CURRENT_USER: Optional[str] = None
CURRENT_CONFIG_ID: Optional[str] = None
CURRENT_CONFIG_DATA: Dict[str, Any] = {}  # 存储当前用户的全部配置


# ====================================================================
# === 任务调度函数（与之前 finetune/infer 逻辑相同，但使用 user 模块的路径）
# ====================================================================

def dispatch_training_task(username: str, id_name: str, config: Dict[str, Any]):
    """启动一个微调任务，并在后台使用 nohup 运行。"""

    # 1. 写入临时配置（用于覆盖参数和确保 finetune.py 使用最新配置）
    temp_config_path = os.path.join("/tmp", f"train_conf_{username}_{id_name}_{os.getpid()}.json")
    try:
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"错误：无法创建临时配置: {e}")
        return

    # 2. 准备后台运行的命令
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, username, f"train_{id_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 使用 sh -c "export PYTHONPATH=... && python ..." 结构解决 ModuleNotFoundError
    PYTHONPATH_VAR = f"export PYTHONPATH={PROJECT_ROOT}"
    internal_command = f"{PYTHONPATH_VAR} && {sys.executable} {FINETUNE_SCRIPT} --config {temp_config_path}"
    command = f"nohup sh -c \"{internal_command}\" > {log_file} 2>&1 &"

    try:
        subprocess.Popen(command, shell=True, close_fds=True)
        print(f"\n训练任务已成功在后台启动！")
        print(f"   日志文件路径: {log_file}")
        print(f"   使用 'log' 命令查看实时进度。")
        # 自动调用 log 指令
        view_training_log(username, id_name, log_file)
    except Exception as e:
        print(f"启动训练任务失败: {e}")


def dispatch_inference_task(config_path: str):
    """启动模型对话任务（前台运行）。"""
    command = [
        sys.executable,
        INFER_SCRIPT,
        "--config",
        config_path
    ]

    try:
        # 对话过程中使用 quit/exit 返回后，回到当前配置控制界面
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"对话服务意外终止 (返回码: {e.returncode})。")
    except Exception as e:
        print(f"对话服务启动失败或意外终止: {e}")


def view_training_log(username: str, id_name: str, log_file: Optional[str] = None):
    """查找并显示最新的训练日志路径，指导用户使用 tail -f"""
    if not log_file:
        # 查找最新的日志文件
        user_log_dir = os.path.join(LOGS_DIR, username)
        log_files = sorted([
            f for f in os.listdir(user_log_dir) if f.startswith(f"train_{id_name}")
        ], reverse=True)

        if log_files:
            log_file = os.path.join(user_log_dir, log_files[0])
        else:
            print(f"⚠找不到角色 '{id_name}' 的训练日志。")
            return

    print(f"=========================================")
    print(f"     角色ID '{id_name}' 的训练日志     ")
    print(f"   路径: {log_file}")
    print("-----------------------------------------")
    print("请在另一个终端或退出当前界面后执行以下命令查看实时进度：")
    print(f"   tail -f {log_file}")
    print("-----------------------------------------")
    print("按 Enter 键返回当前配置界面...")
    input()  # 阻塞，等待用户确认


# ====================================================================
# === 菜单和状态控制函数
# ====================================================================

def menu_main():
    """主菜单：未登录状态"""
    global CURRENT_USER

    while True:
        print("\n=========================================")
        print("     Deep Learning 云服务控制台     ")
        print("=========================================")
        print("1. 登录 (login)")
        print("2. 注册 (register)")
        print("3. 退出程序 (exit)")

        choice = input("请选择操作 (输入数字或括号内指令): ").strip().lower()

        if choice in ('1', 'login'):
            handle_login()
        elif choice in ('2', 'register'):
            handle_register()
        elif choice in ('3', 'exit'):
            print("再见！")
            break
        else:
            print("无效的选择，请重试。")


def handle_login():
    """处理用户登录流程"""
    global CURRENT_USER, CURRENT_CONFIG_DATA

    print("\n--- 用户登录 ---")
    username = input("用户名: ").strip()
    pwd = input("密码 (明文): ").strip()

    config_data = user.login_user(username, pwd)

    if config_data is not None:
        CURRENT_USER = username
        CURRENT_CONFIG_DATA = config_data
        print(f"登录成功！欢迎, {username}。")
        menu_config_select()  # 进入配置选择菜单
    else:
        print("登录失败：用户名或密码错误。")


def handle_register():
    """处理用户注册流程"""
    print("\n--- 用户注册 ---")
    username = input("输入用户名: ").strip()
    pwd = input("输入密码 (明文): ").strip()

    if not username or not pwd:
        print("用户名和密码不能为空。")
        return

    error = user.register_user(username, pwd)

    if error is None:
        print(f"注册成功！用户 '{username}' 的环境已为您创建。")
        # 注册成功后自动登录
        handle_login()
    else:
        print(f"注册失败: {error}")


def menu_config_select():
    """二级菜单：配置选择界面（已登录状态）"""
    global CURRENT_USER, CURRENT_CONFIG_ID, CURRENT_CONFIG_DATA

    while CURRENT_USER and not CURRENT_CONFIG_ID:

        CURRENT_CONFIG_DATA = user.get_user_config_data(CURRENT_USER)  # 刷新配置列表

        print("\n=========================================")
        print(f"     {CURRENT_USER} 的配置仓库     ")
        print("-----------------------------------------")
        if CURRENT_CONFIG_DATA:
            for i, config_id in enumerate(CURRENT_CONFIG_DATA.keys()):
                print(f"{i + 1}. 进入配置 [{config_id}]")
        else:
            print("尚无可用配置。")
        print("-----------------------------------------")
        print("N. 新建配置 (new)")
        print("D. 数据集管理 (data)")
        print("L. 退出登录 (logout)")

        choice = input("请选择操作 (输入数字/指令): ").strip().lower()

        if choice.isdigit() and 1 <= int(choice) <= len(CURRENT_CONFIG_DATA):
            config_id = list(CURRENT_CONFIG_DATA.keys())[int(choice) - 1]
            CURRENT_CONFIG_ID = config_id
            print(f"进入配置: {config_id}")
            menu_config_control()  # 进入配置控制菜单

        elif choice in ('n', 'new'):
            new_id = input("输入新配置ID (字母数字): ").strip()
            error = user.create_new_config(CURRENT_USER, new_id)
            if error:
                print(f"创建失败: {error}")
            else:
                print(f"配置 '{new_id}' 创建成功！")

        elif choice in ('d', 'data'):
            menu_data_management()

        elif choice in ('l', 'logout'):
            CURRENT_USER = None
            CURRENT_CONFIG_DATA = {}
            print("退出登录。")
            break
        else:
            print("无效的选择，请重试。")


def menu_config_control():
    """三级菜单：当前配置控制界面"""
    global CURRENT_CONFIG_ID, CURRENT_CONFIG_DATA

    config_id = CURRENT_CONFIG_ID
    config = CURRENT_CONFIG_DATA[config_id]

    while CURRENT_CONFIG_ID == config_id:
        print("\n=========================================")
        print(f"     配置控制: {config_id}     ")
        print("-----------------------------------------")
        print(f"当前数据集: {os.path.basename(config.get('DATA_PATH', '未设置'))}")
        print("1. 查看配置参数 (view)")
        print("2. 修改配置参数 (edit)")
        print("3. 启动模型训练 (train)")
        print("4. 启动模型对话 (chat)")
        print("B. 返回配置选择 (back)")

        choice = input("请选择操作 (输入数字/指令): ").strip().lower()

        if choice in ('1', 'view'):
            handle_view_config(config)
        elif choice in ('2', 'edit'):
            handle_edit_config(config_id, config)
        elif choice in ('3', 'train'):
            handle_train_model(config_id, config)
        elif choice in ('4', 'chat'):
            handle_chat_model(config_id, config)
        elif choice in ('b', 'back'):
            CURRENT_CONFIG_ID = None
            print("退出当前配置。")
            break
        else:
            print("无效的选择，请重试。")


def handle_view_config(config: Dict[str, Any]):
    """查看当前配置参数"""
    print("\n--- 当前配置参数 ---")
    print(json.dumps(config, indent=4))
    print("--------------------")


def handle_edit_config(config_id: str, config: Dict[str, Any]):
    """修改当前配置参数"""
    print("\n--- 修改配置参数 ---")
    print("可修改键值 (默认LORA参数): DATA_PATH, learning_rate, epochs, batch_size, gradient_accumulation_steps...")
    key = input("输入要修改的键 (例如 epochs): ").strip()

    if key == 'DATA_PATH':
        new_val = input(f"输入新的数据集文件路径 (位于 data/{CURRENT_USER}/ 的文件): ").strip()
        # 校验数据集是否存在
        data_path_check = os.path.join(user.DATA_DIR, CURRENT_USER, new_val)
        if not os.path.exists(data_path_check):
            print(f"错误：数据集文件 {data_path_check} 不存在。")
            return
        config['DATA_PATH'] = data_path_check
        print(f"{key} 更新成功。")

    elif key in config:
        new_val = input(f"输入 {key} 的新值 (当前: {config[key]}): ").strip()
        try:
            # 尝试类型转换
            if isinstance(config[key], int):
                config[key] = int(new_val)
            elif isinstance(config[key], float):
                config[key] = float(new_val)
            else:
                config[key] = new_val
            print(f"{key} 更新成功。")
        except ValueError:
            print("错误：输入值类型不匹配。")
            return

    elif key in config.get('DEFAULT_LORA_PARAMS', {}):
        new_val = input(f"输入 {key} 的新值 (当前: {config['DEFAULT_LORA_PARAMS'][key]}): ").strip()
        try:
            # 尝试类型转换
            param_type = type(config['DEFAULT_LORA_PARAMS'][key])
            config['DEFAULT_LORA_PARAMS'][key] = param_type(new_val)
            print(f"{key} 更新成功。")
        except ValueError:
            print("错误：输入值类型不匹配。")
            return
    else:
        print("错误：键不存在或不可修改。")
        return

    # 保存修改
    save_config_file(CURRENT_USER, config_id, config)


def handle_train_model(config_id: str, config: Dict[str, Any]):
    """启动模型训练任务"""
    if not os.path.exists(config.get('DATA_PATH', '')):
        print("错误：DATA_PATH 未设置或数据集文件不存在，请先修改配置并上传数据集。")
        return

    print("\n--- 启动训练任务 ---")
    print(f"模型: {config['MODEL_PATH']}")
    print(f"数据集: {config['DATA_PATH']}")
    print(f"轮数: {config['DEFAULT_LORA_PARAMS']['epochs']}")

    dispatch_training_task(CURRENT_USER, config_id, config)


def handle_chat_model(config_id: str, config: Dict[str, Any]):
    """启动模型对话任务"""
    # 构造当前配置文件的绝对路径
    config_path = os.path.join(BASE_CONFIG_DIR, CURRENT_USER, f"{config_id}.json")

    lora_weight_path = os.path.join(config['WEIGHT_PATH'], config['USERNAME'], config_id, "final_lora_weights")
    if not os.path.isdir(lora_weight_path):
        print("错误：当前配置尚未训练出权重，请先运行训练任务。")
        return

    print("\n--- 启动对话任务 (Ctrl+C 或 输入 'quit' 退出) ---")
    # 调用推理函数 (它在前台运行)
    dispatch_inference_task(config_path)
    print("\n退出对话，返回配置控制界面。")


def menu_data_management():
    """数据集管理界面"""
    global CURRENT_USER

    while True:
        datasets = user.list_user_datasets(CURRENT_USER)

        print("\n=========================================")
        print(f"     {CURRENT_USER} 的数据集仓库     ")
        print("-----------------------------------------")
        if datasets:
            for i, ds_name in enumerate(datasets):
                print(f"{i + 1}. {ds_name}")
        else:
            print("数据仓库为空。")
        print("-----------------------------------------")
        print("U. 上传数据集 (upload)")
        print("D. 删除数据集 (delete)")
        print("B. 返回配置选择 (back)")

        choice = input("请选择操作 (输入指令): ").strip().lower()

        if choice in ('u', 'upload'):
            local_path = input("输入您本地 JSON 文件的完整路径: ").strip()
            ds_name = input("输入数据集在服务器上的存储名称 (不带.json): ").strip()
            error = user.upload_dataset(CURRENT_USER, local_path, ds_name)
            if error:
                print(f"上传失败: {error}")
            else:
                print(f"数据集 '{ds_name}.json' 上传成功！")

        elif choice in ('d', 'delete'):
            ds_name = input("输入要删除的数据集名称 (不带.json): ").strip()
            error = user.delete_user_dataset(CURRENT_USER, ds_name)
            if error:
                print(f"删除失败: {error}")
            else:
                print(f"数据集 '{ds_name}.json' 删除成功！")

        elif choice in ('b', 'back'):
            break
        else:
            print("无效的选择，请重试。")


def cli_entry():
    """程序的命令行入口点，用于启动主菜单"""
    # 强制不使用 argparse，直接启动主菜单，因为我们需要交互式控制
    menu_main()


if __name__ == '__main__':
    # 确保 users.json 所在的目录存在
    os.makedirs(os.path.dirname(user.USERS_JSON_PATH), exist_ok=True)
    cli_entry()