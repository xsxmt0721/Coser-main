# cli.py
"""
Coser CLI - LLM 微调云服务命令行工具

用法:
    python cli.py auth login --username <user> --password <pwd>
    python cli.py auth register --username <user> --password <pwd>
    python cli.py auth logout
    python cli.py config list
    python cli.py config create --id <config_id>
    python cli.py config update --id <config_id> --data-path <path>
    python cli.py config delete --id <config_id>
    python cli.py dataset list
    python cli.py dataset upload --name <dataset_name>
    python cli.py dataset delete --name <dataset_name>
    python cli.py task train --config-id <id> [--epochs N] [--lr 0.0001]
    python cli.py task log --config-id <id> [--full]
    python cli.py chat --config-id <id> --message "你好"
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlencode

import click
import requests

# --- 配置 ---
DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
SESSION_FILE = Path.home() / ".coser_session.json"

# 显式禁用代理
NO_PROXY = {
    "http": None,
    "https": None,
}


# --- 会话管理 ---
def load_session() -> Dict[str, str]:
    """加载本地保存的会话信息"""
    if SESSION_FILE.exists():
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_session(data: Dict[str, str]) -> None:
    """保存会话信息到本地"""
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clear_session() -> None:
    """清除本地会话"""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


def get_server_url() -> str:
    """获取服务器 URL"""
    session = load_session()
    return session.get("server_url", DEFAULT_SERVER_URL)


def get_session_token() -> Optional[str]:
    """获取当前会话 token"""
    session = load_session()
    return session.get("session_token")


def require_auth() -> str:
    """要求用户已登录，返回 session_token"""
    token = get_session_token()
    if not token:
        raise click.ClickException("未登录，请先执行 `python cli.py auth login`")
    return token


# --- HTTP 客户端 ---
class APIClient:
    """封装 API 请求的客户端（使用 requests 库）"""

    def __init__(self, base_url: str, session_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session_token = session_token
        self.session = requests.Session()
        self.session.proxies = NO_PROXY
        self.session.trust_env = False

    def _build_url(self, endpoint: str, extra_params: Optional[Dict] = None) -> str:
        """构建完整 URL（包含认证参数）"""
        url = f"{self.base_url}{endpoint}"
        params = {}
        if self.session_token:
            params["session_token"] = self.session_token
        if extra_params:
            params.update(extra_params)
        if params:
            url = f"{url}?{urlencode(params)}"
        return url

    def get(self, endpoint: str, params: Optional[Dict] = None) -> requests.Response:
        """发送 GET 请求"""
        url = self._build_url(endpoint, params)
        return self.session.get(url, timeout=30)

    def post(self, endpoint: str, json_data: Optional[Dict] = None) -> requests.Response:
        """发送 POST 请求"""
        url = self._build_url(endpoint)
        return self.session.post(url, json=json_data, timeout=120)

    def put(self, endpoint: str, json_data: Optional[Dict] = None) -> requests.Response:
        """发送 PUT 请求"""
        url = self._build_url(endpoint)
        return self.session.put(url, json=json_data, timeout=30)

    def delete(self, endpoint: str, json_data: Optional[Dict] = None) -> requests.Response:
        """发送 DELETE 请求"""
        url = self._build_url(endpoint)
        return self.session.delete(url, json=json_data, timeout=30)

    def close(self):
        """关闭客户端连接"""
        self.session.close()


def get_client(require_login: bool = False) -> APIClient:
    """获取 API 客户端实例"""
    server_url = get_server_url()
    token = require_auth() if require_login else get_session_token()
    return APIClient(server_url, token)


def handle_response(resp: requests.Response, success_msg: Optional[str] = None) -> Dict[str, Any]:
    """统一处理 API 响应"""
    try:
        data = resp.json()
    except json.JSONDecodeError:
        data = {"raw": resp.text}

    if resp.status_code >= 400:
        error_detail = data.get("detail", str(data))
        raise click.ClickException(f"API 错误 [{resp.status_code}]: {error_detail}")

    if success_msg:
        click.echo(click.style(success_msg, fg="green"))

    return data


# --- 主命令组 ---
@click.group()
@click.option("--server", "-s", default=None, help="指定服务器 URL")
@click.pass_context
def cli(ctx, server: Optional[str]):
    """Coser CLI - LLM 微调云服务命令行工具"""
    ctx.ensure_object(dict)
    if server:
        session = load_session()
        session["server_url"] = server
        save_session(session)
        click.echo(f"服务器地址已设置为: {server}")


# ====================================================================
# === 认证命令组 (auth) ===
# ====================================================================
@cli.group()
def auth():
    """用户认证相关命令"""
    pass


@auth.command("register")
@click.option("--username", "-u", required=True, prompt="用户名", help="注册用户名")
@click.option("--password", "-p", required=True, prompt="密码", hide_input=True, help="注册密码")
def auth_register(username: str, password: str):
    """注册新用户"""
    client = get_client(require_login=False)
    try:
        # 后端 UserRegister 模型字段: username, pwd
        resp = client.post("/auth/register", {"username": username, "pwd": password})
        data = handle_response(resp, f"✓ 用户 '{username}' 注册成功！")
        click.echo(data.get("message", ""))
    finally:
        client.close()


@auth.command("login")
@click.option("--username", "-u", required=True, prompt="用户名", help="登录用户名")
@click.option("--password", "-p", required=True, prompt="密码", hide_input=True, help="登录密码")
def auth_login(username: str, password: str):
    """用户登录"""
    client = get_client(require_login=False)
    try:
        # 后端 UserLogin 模型字段: username, pwd
        resp = client.post("/auth/login", {"username": username, "pwd": password})
        data = handle_response(resp)

        session = load_session()
        session["session_token"] = data.get("session_token")
        session["username"] = username
        save_session(session)

        click.echo(click.style(f"✓ 登录成功！欢迎 {username}", fg="green"))

        config_data = data.get("config_data", {})
        if config_data:
            click.echo("\n您的配置列表:")
            for config_id in config_data.keys():
                click.echo(f"  - {config_id}")
    finally:
        client.close()


@auth.command("logout")
def auth_logout():
    """用户注销"""
    client = get_client(require_login=True)
    try:
        resp = client.post("/auth/logout")
        handle_response(resp, "✓ 注销成功！")
        clear_session()
    finally:
        client.close()


@auth.command("status")
def auth_status():
    """查看当前登录状态"""
    session = load_session()
    token = session.get("session_token")
    username = session.get("username")
    server = session.get("server_url", DEFAULT_SERVER_URL)

    click.echo(f"服务器: {server}")
    if token and username:
        click.echo(click.style(f"已登录用户: {username}", fg="green"))
    else:
        click.echo(click.style("未登录", fg="yellow"))


# ====================================================================
# === 配置管理命令组 (config) ===
# ====================================================================
@cli.group()
def config():
    """训练配置管理命令"""
    pass


@config.command("list")
def config_list():
    """列出所有训练配置"""
    client = get_client(require_login=True)
    try:
        resp = client.get("/config/all")
        data = handle_response(resp)

        if not data:
            click.echo("暂无配置")
            return

        click.echo("\n训练配置列表:")
        click.echo("-" * 50)
        for config_id, config_detail in data.items():
            click.echo(f"\n配置ID: {click.style(config_id, fg='cyan', bold=True)}")
            if isinstance(config_detail, dict):
                data_path = config_detail.get("DATA_PATH", "未设置")
                click.echo(f"  数据集路径: {data_path}")
                lora_params = config_detail.get("DEFAULT_LORA_PARAMS", {})
                if lora_params:
                    click.echo("  LoRA 参数:")
                    for key, value in lora_params.items():
                        click.echo(f"    {key}: {value}")
    finally:
        client.close()


@config.command("create")
@click.option("--id", "config_id", required=True, prompt="配置ID", help="新配置的唯一标识")
def config_create(config_id: str):
    """创建新的训练配置"""
    client = get_client(require_login=True)
    try:
        # 后端 ConfigCreate 模型字段: config_id
        resp = client.post("/config/new", {"config_id": config_id})
        handle_response(resp, f"✓ 配置 '{config_id}' 创建成功！")
    finally:
        client.close()


@config.command("update")
@click.option("--id", "config_id", required=True, help="配置ID")
@click.option("--data-path", default=None, help="数据集文件路径")
@click.option("--lora-r", type=int, default=None, help="LoRA rank")
@click.option("--lora-alpha", type=int, default=None, help="LoRA alpha")
@click.option("--epochs", type=int, default=None, help="训练轮数")
@click.option("--lr", type=float, default=None, help="学习率")
@click.option("--batch-size", type=int, default=None, help="批次大小")
def config_update(config_id: str, data_path: Optional[str], lora_r: Optional[int],
                  lora_alpha: Optional[int], epochs: Optional[int], lr: Optional[float],
                  batch_size: Optional[int]):
    """修改训练配置参数"""
    # 后端 ConfigUpdate 模型字段: data_path, lora_params
    update_data: Dict[str, Any] = {}

    if data_path is not None:
        update_data["data_path"] = data_path

    lora_params = {}
    if lora_r is not None:
        lora_params["r"] = lora_r
    if lora_alpha is not None:
        lora_params["lora_alpha"] = lora_alpha
    if epochs is not None:
        lora_params["num_train_epochs"] = epochs
    if lr is not None:
        lora_params["learning_rate"] = lr
    if batch_size is not None:
        lora_params["per_device_train_batch_size"] = batch_size

    if lora_params:
        update_data["lora_params"] = lora_params

    if not update_data:
        raise click.ClickException("请至少指定一个要修改的参数")

    client = get_client(require_login=True)
    try:
        resp = client.put(f"/config/{config_id}", update_data)
        handle_response(resp, f"✓ 配置 '{config_id}' 更新成功！")
    finally:
        client.close()


@config.command("delete")
@click.option("--id", "config_id", required=True, help="配置ID")
@click.confirmation_option(prompt="确定要删除此配置及其关联的模型权重吗?")
def config_delete(config_id: str):
    """删除训练配置及关联的模型权重"""
    client = get_client(require_login=True)
    try:
        resp = client.delete(f"/config/{config_id}")
        handle_response(resp, f"✓ 配置 '{config_id}' 及其权重已删除！")
    finally:
        client.close()


# ====================================================================
# === 数据集管理命令组 (dataset) ===
# ====================================================================
@cli.group()
def dataset():
    """数据集管理命令"""
    pass


@dataset.command("list")
def dataset_list():
    """列出所有数据集"""
    client = get_client(require_login=True)
    try:
        resp = client.get("/dataset/list")
        data = handle_response(resp)

        if not data:
            click.echo("暂无数据集")
            return

        click.echo("\n数据集列表:")
        for ds in data:
            click.echo(f"  - {ds}")
    finally:
        client.close()


@dataset.command("upload")
@click.option("--name", required=True, prompt="数据集名称", help="数据集名称（不含扩展名）")
def dataset_upload(name: str):
    """从 Cache 导入数据集到用户目录"""
    client = get_client(require_login=True)
    try:
        # 后端 DatasetUpload 模型字段: dataset_name
        resp = client.post("/dataset/upload", {"dataset_name": name})
        handle_response(resp, f"✓ 数据集 '{name}' 导入成功！")
    finally:
        client.close()


@dataset.command("delete")
@click.option("--name", required=True, help="数据集名称")
@click.confirmation_option(prompt="确定要删除此数据集吗?")
def dataset_delete(name: str):
    """删除数据集"""
    client = get_client(require_login=True)
    try:
        # 后端 DatasetDelete 模型字段: dataset_name
        resp = client.delete("/dataset/delete", {"dataset_name": name})
        handle_response(resp, f"✓ 数据集 '{name}' 已删除！")
    finally:
        client.close()


# ====================================================================
# === 任务管理命令组 (task) ===
# ====================================================================
@cli.group()
def task():
    """训练任务管理命令"""
    pass


@task.command("train")
@click.option("--config-id", required=True, help="配置ID")
@click.option("--epochs", type=int, default=None, help="训练轮数（临时覆盖）")
@click.option("--lr", type=float, default=None, help="学习率（临时覆盖）")
@click.option("--batch-size", type=int, default=None, help="批次大小（临时覆盖）")
@click.option("--lora-r", type=int, default=None, help="LoRA rank（临时覆盖）")
@click.option("--lora-alpha", type=int, default=None, help="LoRA alpha（临时覆盖）")
def task_train(config_id: str, epochs: Optional[int], lr: Optional[float],
               batch_size: Optional[int], lora_r: Optional[int], lora_alpha: Optional[int]):
    """启动模型训练任务"""
    # 后端 TrainRequest 模型字段: config_id, r, lora_alpha, num_train_epochs, learning_rate, per_device_train_batch_size
    request_data: Dict[str, Any] = {"config_id": config_id}

    if epochs is not None:
        request_data["num_train_epochs"] = epochs
    if lr is not None:
        request_data["learning_rate"] = lr
    if batch_size is not None:
        request_data["per_device_train_batch_size"] = batch_size
    if lora_r is not None:
        request_data["r"] = lora_r
    if lora_alpha is not None:
        request_data["lora_alpha"] = lora_alpha

    client = get_client(require_login=True)
    try:
        click.echo("正在启动训练任务...")
        resp = client.post("/task/train", request_data)
        data = handle_response(resp)

        click.echo(click.style("\n✓ 训练任务已启动！", fg="green"))
        click.echo(f"  任务ID: {data.get('task_id', 'N/A')}")
        click.echo(f"  配置ID: {data.get('config_id', 'N/A')}")
        click.echo(f"  日志路径: {data.get('log_file_path', 'N/A')}")
        click.echo(f"\n使用 `python cli.py task log --config-id {config_id}` 查看训练日志")
    finally:
        client.close()


@task.command("log")
@click.option("--config-id", required=True, help="配置ID")
@click.option("--follow", "-f", is_flag=True, help="持续刷新日志（每5秒）")
@click.option("--last-line", "-l", is_flag=True, default=True, help="只显示最后一行日志（默认开启）")
@click.option("--full", is_flag=True, help="显示完整日志")
def task_log(config_id: str, follow: bool, last_line: bool, full: bool):
    """查看训练任务日志"""

    def fetch_and_display() -> str:
        client = get_client(require_login=True)
        try:
            resp = client.get(f"/task/log/{config_id}")
            data = handle_response(resp)

            status = data.get("status", "UNKNOWN")
            log_content = data.get("log_content", "")

            if follow:
                click.clear()

            status_color = {"RUNNING": "yellow", "COMPLETED": "green", "FAILED": "red"}.get(status, "white")
            click.echo(f"配置ID: {config_id}")
            click.echo(f"状态: {click.style(status, fg=status_color, bold=True)}")
            click.echo(f"日志路径: {data.get('log_file_path', 'N/A')}")
            click.echo("-" * 50)

            if log_content:
                if full:
                    # 显示完整日志
                    click.echo(log_content)
                else:
                    # 只显示最后一行（过滤空行）
                    lines = [line for line in log_content.strip().split('\n') if line.strip()]
                    if lines:
                        click.echo(lines[-1])
                    else:
                        click.echo("(暂无日志)")
            else:
                click.echo("(暂无日志)")

            return status
        finally:
            client.close()

    if follow:
        click.echo("持续监控日志中... (Ctrl+C 退出)")
        try:
            while True:
                status = fetch_and_display()
                if status in ("COMPLETED", "FAILED"):
                    click.echo(f"\n任务已结束: {status}")
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            click.echo("\n已停止监控")
    else:
        fetch_and_display()



# ====================================================================
# === 推理对话命令 (chat) ===
# ====================================================================
@cli.command("chat")
@click.option("--config-id", required=True, help="使用的模型配置ID")
@click.option("--message", "-m", default=None, help="发送的消息")
@click.option("--interactive", "-i", is_flag=True, help="进入交互式对话模式")
@click.option("--max-tokens", type=int, default=512, help="最大生成 token 数")
@click.option("--temperature", type=float, default=0.7, help="生成温度")
def chat(config_id: str, message: Optional[str], interactive: bool,
         max_tokens: int, temperature: float):
    """使用微调后的模型进行对话"""

    def send_message(msg: str, history: list) -> Tuple[str, float]:
        """发送消息并获取回复"""
        client = get_client(require_login=True)
        try:
            # 后端 ChatRequest 模型字段: config_id, prompt, history, max_new_tokens, temperature
            request_data = {
                "config_id": config_id,
                "prompt": msg,  # 关键修复：使用 prompt 而非 message
                "history": history,
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }
            resp = client.post("/inference/chat", request_data)
            data = handle_response(resp)
            return data.get("response", ""), data.get("time_taken", 0)
        finally:
            client.close()

    if interactive:
        click.echo(click.style(f"进入交互式对话 (配置: {config_id})", fg="cyan"))
        click.echo("输入 'exit' 或 'quit' 退出，输入 'clear' 清除对话历史\n")

        history = []
        while True:
            try:
                user_input = click.prompt("你", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                click.echo("\n再见！")
                break

            if user_input.lower() in ("exit", "quit"):
                click.echo("再见！")
                break

            if user_input.lower() == "clear":
                history = []
                click.echo("对话历史已清除\n")
                continue

            try:
                response, time_taken = send_message(user_input, history)
                click.echo(f"{click.style('AI', fg='green')}> {response}")
                click.echo(click.style(f"  (耗时: {time_taken:.2f}s)\n", fg="bright_black"))

                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})

            except click.ClickException as e:
                click.echo(click.style(f"错误: {e.message}", fg="red"))

    elif message:
        response, time_taken = send_message(message, [])
        click.echo(f"\n{click.style('AI', fg='green')}: {response}")
        click.echo(click.style(f"\n(耗时: {time_taken:.2f}s)", fg="bright_black"))

    else:
        raise click.ClickException("请使用 --message 发送消息，或使用 --interactive 进入交互模式")


# ====================================================================
# === 入口点 ===
# ====================================================================
if __name__ == "__main__":
    cli()
