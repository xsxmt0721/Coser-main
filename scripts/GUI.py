# GUI.py
"""
Coser GUI - LLM 微调云服务图形界面客户端

独立运行的 GUI 客户端，通过 HTTP API 与后端服务通信。
可放置在任意位置运行，无需依赖项目其他模块。

使用方法：
    python GUI.py
"""

import json
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlencode

import requests

# ====================================================================
# === 配置常量 ===
# ====================================================================
DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
SESSION_FILE = Path.home() / ".coser_session.json"

NO_PROXY = {"http": None, "https": None}


# ====================================================================
# === 会话管理 ===
# ====================================================================
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


# ====================================================================
# === API 客户端 ===
# ====================================================================
class APIClient:
    """封装 API 请求的客户端"""

    def __init__(self, base_url: str, session_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session_token = session_token
        self.session = requests.Session()
        self.session.proxies = NO_PROXY
        self.session.trust_env = False

    def _build_url(self, endpoint: str, extra_params: Optional[Dict] = None) -> str:
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
        url = self._build_url(endpoint, params)
        return self.session.get(url, timeout=30)

    def post(self, endpoint: str, json_data: Optional[Dict] = None) -> requests.Response:
        url = self._build_url(endpoint)
        return self.session.post(url, json=json_data, timeout=120)

    def put(self, endpoint: str, json_data: Optional[Dict] = None) -> requests.Response:
        url = self._build_url(endpoint)
        return self.session.put(url, json=json_data, timeout=30)

    def delete(self, endpoint: str, json_data: Optional[Dict] = None) -> requests.Response:
        url = self._build_url(endpoint)
        return self.session.delete(url, json=json_data, timeout=30)

    def close(self):
        self.session.close()


# ====================================================================
# === 主应用类 ===
# ====================================================================
class CoserGUI:
    """Coser GUI 主应用"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Coser - LLM 微调云服务客户端")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        # 状态变量
        self.server_url = tk.StringVar(value=DEFAULT_SERVER_URL)
        self.username = tk.StringVar()
        self.session_token: Optional[str] = None
        self.is_logged_in = False

        # 日志监控相关
        self.log_monitor_running = False
        self.log_monitor_thread: Optional[threading.Thread] = None

        # 加载已保存的会话
        self._load_saved_session()

        # 构建界面
        self._build_ui()

        # 更新登录状态显示
        self._update_login_status()

    def _load_saved_session(self):
        """加载已保存的会话"""
        session = load_session()
        if session.get("session_token") and session.get("username"):
            self.session_token = session["session_token"]
            self.username.set(session["username"])
            self.is_logged_in = True
        if session.get("server_url"):
            self.server_url.set(session["server_url"])

    def _get_client(self) -> APIClient:
        """获取 API 客户端"""
        return APIClient(self.server_url.get(), self.session_token)

    def _build_ui(self):
        """构建用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 顶部：服务器配置和登录状态 ---
        top_frame = ttk.LabelFrame(main_frame, text="连接设置", padding="10")
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # 服务器 URL
        ttk.Label(top_frame, text="服务器地址:").grid(row=0, column=0, sticky=tk.W)
        server_entry = ttk.Entry(top_frame, textvariable=self.server_url, width=40)
        server_entry.grid(row=0, column=1, padx=5)

        # 登录状态标签
        self.status_label = ttk.Label(top_frame, text="未登录", foreground="red")
        self.status_label.grid(row=0, column=2, padx=20)

        # 登录/注销按钮
        self.login_btn = ttk.Button(top_frame, text="登录", command=self._show_login_dialog)
        self.login_btn.grid(row=0, column=3, padx=5)

        self.logout_btn = ttk.Button(top_frame, text="注销", command=self._logout, state=tk.DISABLED)
        self.logout_btn.grid(row=0, column=4, padx=5)

        ttk.Button(top_frame, text="注册", command=self._show_register_dialog).grid(row=0, column=5, padx=5)

        # --- 中部：功能选项卡 ---
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # 配置管理选项卡
        self.config_tab = self._build_config_tab(notebook)
        notebook.add(self.config_tab, text="配置管理")

        # 数据集管理选项卡
        self.dataset_tab = self._build_dataset_tab(notebook)
        notebook.add(self.dataset_tab, text="数据集管理")

        # 训练任务选项卡
        self.task_tab = self._build_task_tab(notebook)
        notebook.add(self.task_tab, text="训练任务")

        # 对话推理选项卡
        self.chat_tab = self._build_chat_tab(notebook)
        notebook.add(self.chat_tab, text="对话推理")

        # --- 底部：日志输出 ---
        log_frame = ttk.LabelFrame(main_frame, text="操作日志", padding="10")
        log_frame.pack(fill=tk.X, pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, state=tk.DISABLED)
        self.log_text.pack(fill=tk.X)

        ttk.Button(log_frame, text="清空日志", command=self._clear_log).pack(anchor=tk.E, pady=(5, 0))

    def _build_config_tab(self, parent) -> ttk.Frame:
        """构建配置管理选项卡"""
        frame = ttk.Frame(parent, padding="10")

        # 左侧：配置列表
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        ttk.Label(left_frame, text="配置列表:").pack(anchor=tk.W)

        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.config_listbox = tk.Listbox(list_frame, width=25, height=15)
        self.config_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.config_listbox.bind("<<ListboxSelect>>", self._on_config_select)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.config_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.config_listbox.config(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="刷新", command=self._refresh_configs).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="新建", command=self._create_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="删除", command=self._delete_config).pack(side=tk.LEFT, padx=2)

        # 右侧：配置详情和修改
        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 配置详情显示
        detail_frame = ttk.LabelFrame(right_frame, text="配置详情", padding="5")
        detail_frame.pack(fill=tk.BOTH, expand=True)

        self.config_detail_text = scrolledtext.ScrolledText(detail_frame, height=8, state=tk.DISABLED)
        self.config_detail_text.pack(fill=tk.BOTH, expand=True)

        # 修改配置区域
        update_frame = ttk.LabelFrame(right_frame, text="修改配置", padding="5")
        update_frame.pack(fill=tk.X, pady=(10, 0))

        # 数据集路径
        ttk.Label(update_frame, text="数据集路径:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.config_data_path = ttk.Entry(update_frame, width=40)
        self.config_data_path.grid(row=0, column=1, columnspan=3, padx=5, pady=2, sticky=tk.W)

        # LoRA 参数行1: r, alpha, dropout
        ttk.Label(update_frame, text="LoRA r:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.config_lora_r = ttk.Entry(update_frame, width=8)
        self.config_lora_r.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(update_frame, text="LoRA alpha:").grid(row=1, column=2, sticky=tk.W, pady=2)
        self.config_lora_alpha = ttk.Entry(update_frame, width=8)
        self.config_lora_alpha.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)

        ttk.Label(update_frame, text="Dropout:").grid(row=1, column=4, sticky=tk.W, pady=2)
        self.config_lora_dropout = ttk.Entry(update_frame, width=8)
        self.config_lora_dropout.grid(row=1, column=5, sticky=tk.W, padx=5, pady=2)

        # LoRA 参数行2: epochs, lr, batch_size
        ttk.Label(update_frame, text="训练轮数:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.config_epochs = ttk.Entry(update_frame, width=8)
        self.config_epochs.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(update_frame, text="学习率:").grid(row=2, column=2, sticky=tk.W, pady=2)
        self.config_lr = ttk.Entry(update_frame, width=10)
        self.config_lr.grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)

        ttk.Label(update_frame, text="批次大小:").grid(row=2, column=4, sticky=tk.W, pady=2)
        self.config_batch_size = ttk.Entry(update_frame, width=8)
        self.config_batch_size.grid(row=2, column=5, sticky=tk.W, padx=5, pady=2)

        # LoRA 参数行3: max_seq_len
        ttk.Label(update_frame, text="最大序列长度:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.config_max_seq_len = ttk.Entry(update_frame, width=8)
        self.config_max_seq_len.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        # 保存按钮
        ttk.Button(update_frame, text="保存修改", command=self._update_config).grid(
            row=4, column=0, columnspan=6, pady=10
        )

        return frame

    def _build_dataset_tab(self, parent) -> ttk.Frame:
        """构建数据集管理选项卡"""
        frame = ttk.Frame(parent, padding="10")

        # 数据集列表
        ttk.Label(frame, text="数据集列表:").pack(anchor=tk.W)

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.dataset_listbox = tk.Listbox(list_frame, height=15)
        self.dataset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.dataset_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_listbox.config(yscrollcommand=scrollbar.set)

        # 按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="刷新列表", command=self._refresh_datasets).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="从 Cache 导入", command=self._upload_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="删除数据集", command=self._delete_dataset).pack(side=tk.LEFT, padx=2)

        return frame

    def _build_task_tab(self, parent) -> ttk.Frame:
        """构建训练任务选项卡"""
        frame = ttk.Frame(parent, padding="10")

        # 训练配置区域
        train_frame = ttk.LabelFrame(frame, text="启动训练", padding="10")
        train_frame.pack(fill=tk.X)

        # 行1: 配置ID
        ttk.Label(train_frame, text="配置ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.train_config_id = ttk.Entry(train_frame, width=20)
        self.train_config_id.grid(row=0, column=1, padx=5, pady=2)

        # 行2: LoRA 参数（可选覆盖）
        ttk.Label(train_frame, text="LoRA r:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.train_lora_r = ttk.Entry(train_frame, width=8)
        self.train_lora_r.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(train_frame, text="LoRA alpha:").grid(row=1, column=2, sticky=tk.W, pady=2)
        self.train_lora_alpha = ttk.Entry(train_frame, width=8)
        self.train_lora_alpha.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)

        # 行3: 训练参数
        ttk.Label(train_frame, text="训练轮数:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.train_epochs = ttk.Entry(train_frame, width=8)
        self.train_epochs.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(train_frame, text="学习率:").grid(row=2, column=2, sticky=tk.W, pady=2)
        self.train_lr = ttk.Entry(train_frame, width=10)
        self.train_lr.grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)

        ttk.Label(train_frame, text="批次大小:").grid(row=2, column=4, sticky=tk.W, pady=2)
        self.train_batch_size = ttk.Entry(train_frame, width=8)
        self.train_batch_size.grid(row=2, column=5, sticky=tk.W, padx=5, pady=2)

        # 启动按钮
        ttk.Button(train_frame, text="启动训练", command=self._start_training).grid(
            row=3, column=0, columnspan=6, pady=10
        )

        # 日志监控区域（保持不变）
        log_frame = ttk.LabelFrame(frame, text="训练日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        log_ctrl_frame = ttk.Frame(log_frame)
        log_ctrl_frame.pack(fill=tk.X)

        ttk.Label(log_ctrl_frame, text="配置ID:").pack(side=tk.LEFT)
        self.log_config_id = ttk.Entry(log_ctrl_frame, width=20)
        self.log_config_id.pack(side=tk.LEFT, padx=5)

        ttk.Button(log_ctrl_frame, text="获取日志", command=self._fetch_log).pack(side=tk.LEFT, padx=5)

        self.monitor_btn = ttk.Button(log_ctrl_frame, text="开始监控", command=self._toggle_log_monitor)
        self.monitor_btn.pack(side=tk.LEFT, padx=5)

        self.train_status_label = ttk.Label(log_ctrl_frame, text="状态: --")
        self.train_status_label.pack(side=tk.RIGHT)

        self.train_log_text = scrolledtext.ScrolledText(log_frame, height=15, state=tk.DISABLED)
        self.train_log_text.pack(fill=tk.BOTH, expand=True, pady=5)

        return frame

    def _build_chat_tab(self, parent) -> ttk.Frame:
        """构建对话推理选项卡"""
        frame = ttk.Frame(parent, padding="10")

        # 配置选择
        config_frame = ttk.Frame(frame)
        config_frame.pack(fill=tk.X)

        ttk.Label(config_frame, text="配置ID:").pack(side=tk.LEFT)
        self.chat_config_id = ttk.Entry(config_frame, width=20)
        self.chat_config_id.pack(side=tk.LEFT, padx=5)

        ttk.Label(config_frame, text="最大Token:").pack(side=tk.LEFT, padx=(20, 0))
        self.chat_max_tokens = ttk.Entry(config_frame, width=10)
        self.chat_max_tokens.insert(0, "512")
        self.chat_max_tokens.pack(side=tk.LEFT, padx=5)

        ttk.Label(config_frame, text="温度:").pack(side=tk.LEFT)
        self.chat_temperature = ttk.Entry(config_frame, width=10)
        self.chat_temperature.insert(0, "0.7")
        self.chat_temperature.pack(side=tk.LEFT, padx=5)

        ttk.Button(config_frame, text="清空对话", command=self._clear_chat).pack(side=tk.RIGHT)

        # 对话历史
        self.chat_history_text = scrolledtext.ScrolledText(frame, height=18, state=tk.DISABLED)
        self.chat_history_text.pack(fill=tk.BOTH, expand=True, pady=10)

        # 输入区域
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X)

        self.chat_input = ttk.Entry(input_frame)
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_input.bind("<Return>", lambda e: self._send_chat())

        ttk.Button(input_frame, text="发送", command=self._send_chat).pack(side=tk.RIGHT, padx=(5, 0))

        # 对话历史记录（用于 API 请求）
        self.chat_history = []

        return frame

    # ====================================================================
    # === 登录相关方法 ===
    # ====================================================================
    def _update_login_status(self):
        """更新登录状态显示"""
        if self.is_logged_in:
            self.status_label.config(text=f"已登录: {self.username.get()}", foreground="green")
            self.login_btn.config(state=tk.DISABLED)
            self.logout_btn.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="未登录", foreground="red")
            self.login_btn.config(state=tk.NORMAL)
            self.logout_btn.config(state=tk.DISABLED)

    def _show_login_dialog(self):
        """显示登录对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("登录")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="用户名:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        username_entry = ttk.Entry(dialog, width=25)
        username_entry.grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(dialog, text="密码:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        password_entry = ttk.Entry(dialog, show="*", width=25)
        password_entry.grid(row=1, column=1, padx=10, pady=10)

        def do_login():
            username = username_entry.get().strip()
            password = password_entry.get()
            if not username or not password:
                messagebox.showwarning("警告", "请输入用户名和密码")
                return

            client = self._get_client()
            try:
                resp = client.post("/auth/login", {"username": username, "pwd": password})
                if resp.status_code == 200:
                    data = resp.json()
                    self.session_token = data.get("session_token")
                    self.username.set(username)
                    self.is_logged_in = True

                    # 保存会话
                    save_session({
                        "session_token": self.session_token,
                        "username": username,
                        "server_url": self.server_url.get()
                    })

                    self._update_login_status()
                    self._log(f"登录成功: {username}")
                    dialog.destroy()
                else:
                    error = resp.json().get("detail", "登录失败")
                    messagebox.showerror("错误", error)
            except requests.exceptions.ConnectionError:
                messagebox.showerror("错误", f"无法连接到服务器 {self.server_url.get()}")
            except Exception as e:
                messagebox.showerror("错误", str(e))
            finally:
                client.close()

        ttk.Button(dialog, text="登录", command=do_login).grid(row=2, column=0, columnspan=2, pady=10)

    def _show_register_dialog(self):
        """显示注册对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("注册")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="用户名:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        username_entry = ttk.Entry(dialog, width=25)
        username_entry.grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(dialog, text="密码:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        password_entry = ttk.Entry(dialog, show="*", width=25)
        password_entry.grid(row=1, column=1, padx=10, pady=10)

        def do_register():
            username = username_entry.get().strip()
            password = password_entry.get()
            if not username or not password:
                messagebox.showwarning("警告", "请输入用户名和密码")
                return

            client = self._get_client()
            try:
                resp = client.post("/auth/register", {"username": username, "pwd": password})
                if resp.status_code == 200:
                    messagebox.showinfo("成功", "注册成功，请登录")
                    self._log(f"注册成功: {username}")
                    dialog.destroy()
                else:
                    error = resp.json().get("detail", "注册失败")
                    messagebox.showerror("错误", error)
            except requests.exceptions.ConnectionError:
                messagebox.showerror("错误", f"无法连接到服务器 {self.server_url.get()}")
            except Exception as e:
                messagebox.showerror("错误", str(e))
            finally:
                client.close()

        ttk.Button(dialog, text="注册", command=do_register).grid(row=2, column=0, columnspan=2, pady=10)

    def _logout(self):
        """注销"""
        if not self.is_logged_in:
            return

        client = self._get_client()
        try:
            resp = client.post("/auth/logout")
            if resp.status_code == 200:
                self._log(f"注销成功: {self.username.get()}")
        except Exception as e:
            self._log(f"注销请求失败: {e}")
        finally:
            client.close()

        # 清除本地会话
        self.session_token = None
        self.username.set("")
        self.is_logged_in = False
        clear_session()
        self._update_login_status()

    def _check_login(self) -> bool:
        """检查是否已登录"""
        if not self.is_logged_in:
            messagebox.showwarning("警告", "请先登录")
            return False
        return True

    # ====================================================================
    # === 配置管理方法 ===
    # ====================================================================
    def _refresh_configs(self):
        """刷新配置列表"""
        if not self._check_login():
            return

        client = self._get_client()
        try:
            resp = client.get("/config/all")
            if resp.status_code == 200:
                data = resp.json()
                self.config_listbox.delete(0, tk.END)
                for config_id in data.keys():
                    self.config_listbox.insert(tk.END, config_id)
                self._log(f"已刷新配置列表，共 {len(data)} 个配置")
            else:
                error = resp.json().get("detail", "获取失败")
                messagebox.showerror("错误", error)
        except requests.exceptions.ConnectionError:
            messagebox.showerror("错误", "无法连接到服务器")
        except Exception as e:
            messagebox.showerror("错误", str(e))
        finally:
            client.close()

    def _on_config_select(self, event):
        """配置列表选择事件"""
        selection = self.config_listbox.curselection()
        if not selection:
            return

        config_id = self.config_listbox.get(selection[0])
        self._show_config_detail(config_id)

    def _show_config_detail(self, config_id: str):
        """显示配置详情"""
        client = self._get_client()
        try:
            resp = client.get("/config/all")
            if resp.status_code == 200:
                data = resp.json()
                if config_id in data:
                    config = data[config_id]
                    detail = json.dumps(config, indent=2, ensure_ascii=False)

                    self.config_detail_text.config(state=tk.NORMAL)
                    self.config_detail_text.delete(1.0, tk.END)
                    self.config_detail_text.insert(tk.END, detail)
                    self.config_detail_text.config(state=tk.DISABLED)
        except Exception as e:
            self._log(f"获取配置详情失败: {e}")
        finally:
            client.close()

    def _create_config(self):
        """创建新配置"""
        if not self._check_login():
            return

        config_id = tk.simpledialog.askstring("新建配置", "请输入配置ID:")
        if not config_id:
            return

        client = self._get_client()
        try:
            resp = client.post("/config/new", {"config_id": config_id})
            if resp.status_code == 200:
                messagebox.showinfo("成功", f"配置 '{config_id}' 创建成功")
                self._log(f"创建配置: {config_id}")
                self._refresh_configs()
            else:
                error = resp.json().get("detail", "创建失败")
                messagebox.showerror("错误", error)
        except requests.exceptions.ConnectionError:
            messagebox.showerror("错误", "无法连接到服务器")
        except Exception as e:
            messagebox.showerror("错误", str(e))
        finally:
            client.close()

    def _update_config(self):
        """更新配置"""
        if not self._check_login():
            return

        selection = self.config_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请先选择一个配置")
            return

        config_id = self.config_listbox.get(selection[0])
        update_data = {}

        # 数据集路径
        data_path = self.config_data_path.get().strip()
        if data_path:
            update_data["data_path"] = data_path

        # 收集 LoRA 参数（使用后端 DEFAULT_LORA_PARAMS 中的键名）
        lora_params = {}

        # LoRA r - 后端键名为 "r"
        lora_r = self.config_lora_r.get().strip()
        if lora_r:
            try:
                lora_params["lora_r"] = int(lora_r)
            except ValueError:
                messagebox.showerror("错误", "LoRA r 必须是整数")
                return

        # LoRA alpha - 后端键名为 "lora_alpha"
        lora_alpha = self.config_lora_alpha.get().strip()
        if lora_alpha:
            try:
                lora_params["lora_alpha"] = int(lora_alpha)
            except ValueError:
                messagebox.showerror("错误", "LoRA alpha 必须是整数")
                return

        # Dropout - 后端键名为 "lora_dropout"
        lora_dropout = self.config_lora_dropout.get().strip()
        if lora_dropout:
            try:
                lora_params["lora_dropout"] = float(lora_dropout)
            except ValueError:
                messagebox.showerror("错误", "Dropout 必须是数字")
                return

        # 训练轮数 - 后端键名为 "num_train_epochs"
        epochs = self.config_epochs.get().strip()
        if epochs:
            try:
                lora_params["epochs"] = int(epochs)
            except ValueError:
                messagebox.showerror("错误", "训练轮数必须是整数")
                return

        # 学习率 - 后端键名为 "learning_rate"
        lr = self.config_lr.get().strip()
        if lr:
            try:
                lora_params["learning_rate"] = float(lr)
            except ValueError:
                messagebox.showerror("错误", "学习率必须是数字")
                return

        # 批次大小 - 后端键名为 "per_device_train_batch_size"
        batch_size = self.config_batch_size.get().strip()
        if batch_size:
            try:
                lora_params["batch_size"] = int(batch_size)
            except ValueError:
                messagebox.showerror("错误", "批次大小必须是整数")
                return

        # 最大序列长度 - 后端键名为 "max_seq_len"
        max_seq_len = self.config_max_seq_len.get().strip()
        if max_seq_len:
            try:
                lora_params["max_seq_len"] = int(max_seq_len)
            except ValueError:
                messagebox.showerror("错误", "最大序列长度必须是整数")
                return

        if lora_params:
            update_data["lora_params"] = lora_params

        if not update_data:
            messagebox.showinfo("提示", "未填写任何修改内容")
            return

        client = self._get_client()
        try:
            resp = client.put(f"/config/{config_id}", update_data)
            if resp.status_code == 200:
                messagebox.showinfo("成功", f"配置 '{config_id}' 更新成功")
                self._show_config_detail(config_id)
                self._log(f"配置 '{config_id}' 更新成功")
            else:
                error = resp.json().get("detail", "未知错误")
                messagebox.showerror("错误", f"更新失败: {error}")
        except requests.exceptions.ConnectionError:
            messagebox.showerror("连接错误", "无法连接到服务器")
        except Exception as e:
            messagebox.showerror("错误", str(e))
        finally:
            client.close()

    def _delete_config(self):
        """删除配置"""
        if not self._check_login():
            return

        selection = self.config_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个配置")
            return

        config_id = self.config_listbox.get(selection[0])
        if not messagebox.askyesno("确认", f"确定要删除配置 '{config_id}' 及其关联的模型权重吗？"):
            return

        client = self._get_client()
        try:
            resp = client.delete(f"/config/{config_id}")
            if resp.status_code == 200:
                messagebox.showinfo("成功", f"配置 '{config_id}' 已删除")
                self._log(f"删除配置: {config_id}")
                self._refresh_configs()
            else:
                error = resp.json().get("detail", "删除失败")
                messagebox.showerror("错误", error)
        except requests.exceptions.ConnectionError:
            messagebox.showerror("错误", "无法连接到服务器")
        except Exception as e:
            messagebox.showerror("错误", str(e))
        finally:
            client.close()

    # ====================================================================
    # === 数据集管理方法 ===
    # ====================================================================
    def _refresh_datasets(self):
        """刷新数据集列表"""
        if not self._check_login():
            return

        client = self._get_client()
        try:
            resp = client.get("/dataset/list")
            if resp.status_code == 200:
                data = resp.json()
                self.dataset_listbox.delete(0, tk.END)
                for ds in data:
                    self.dataset_listbox.insert(tk.END, ds)
                self._log(f"已刷新数据集列表，共 {len(data)} 个数据集")
            else:
                error = resp.json().get("detail", "获取失败")
                messagebox.showerror("错误", error)
        except requests.exceptions.ConnectionError:
            messagebox.showerror("错误", "无法连接到服务器")
        except Exception as e:
            messagebox.showerror("错误", str(e))
        finally:
            client.close()

    def _upload_dataset(self):
        """从 Cache 导入数据集"""
        if not self._check_login():
            return

        dataset_name = tk.simpledialog.askstring("导入数据集", "请输入数据集名称（不含扩展名）:")
        if not dataset_name:
            return

        client = self._get_client()
        try:
            resp = client.post("/dataset/upload", {"dataset_name": dataset_name})
            if resp.status_code == 200:
                messagebox.showinfo("成功", f"数据集 '{dataset_name}' 导入成功")
                self._log(f"导入数据集: {dataset_name}")
                self._refresh_datasets()
            else:
                error = resp.json().get("detail", "导入失败")
                messagebox.showerror("错误", error)
        except requests.exceptions.ConnectionError:
            messagebox.showerror("错误", "无法连接到服务器")
        except Exception as e:
            messagebox.showerror("错误", str(e))
        finally:
            client.close()

    def _delete_dataset(self):
        """删除数据集"""
        if not self._check_login():
            return

        selection = self.dataset_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个数据集")
            return

        dataset_name = self.dataset_listbox.get(selection[0])
        # 移除 .json 扩展名
        if dataset_name.endswith(".json"):
            dataset_name = dataset_name[:-5]

        if not messagebox.askyesno("确认", f"确定要删除数据集 '{dataset_name}' 吗？"):
            return

        client = self._get_client()
        try:
            resp = client.delete("/dataset/delete", {"dataset_name": dataset_name})
            if resp.status_code == 200:
                messagebox.showinfo("成功", f"数据集 '{dataset_name}' 已删除")
                self._log(f"删除数据集: {dataset_name}")
                self._refresh_datasets()
            else:
                error = resp.json().get("detail", "删除失败")
                messagebox.showerror("错误", error)
        except requests.exceptions.ConnectionError:
            messagebox.showerror("错误", "无法连接到服务器")
        except Exception as e:
            messagebox.showerror("错误", str(e))
        finally:
            client.close()

    # ====================================================================
    # === 训练任务方法 ===
    # ====================================================================
    def _start_training(self):
        """启动训练任务"""
        if not self._check_login():
            return

        config_id = self.train_config_id.get().strip()
        if not config_id:
            messagebox.showwarning("提示", "请输入配置ID")
            return

        request_data = {"config_id": config_id}

        # 收集可选覆盖参数（使用后端期望的字段名）
        try:
            lora_r = self.train_lora_r.get().strip()
            if lora_r:
                request_data["r"] = int(lora_r)

            lora_alpha = self.train_lora_alpha.get().strip()
            if lora_alpha:
                request_data["lora_alpha"] = int(lora_alpha)

            epochs = self.train_epochs.get().strip()
            if epochs:
                request_data["num_train_epochs"] = int(epochs)  # 关键修复

            lr = self.train_lr.get().strip()
            if lr:
                request_data["learning_rate"] = float(lr)

            batch_size = self.train_batch_size.get().strip()
            if batch_size:
                request_data["per_device_train_batch_size"] = int(batch_size)  # 关键修复

        except ValueError as e:
            messagebox.showerror("参数错误", "请检查输入的数值格式")
            return

        client = self._get_client()
        try:
            self._log(f"正在启动训练任务: {config_id}")
            resp = client.post("/task/train", request_data)
            if resp.status_code == 200:
                data = resp.json()
                messagebox.showinfo("成功", f"训练任务已启动\n任务ID: {data.get('task_id', 'N/A')}")
                self._log(f"训练任务启动成功: {data.get('task_id', 'N/A')}")
                self.log_config_id.delete(0, tk.END)
                self.log_config_id.insert(0, config_id)
            else:
                error = resp.json().get("detail", "未知错误")
                messagebox.showerror("错误", f"启动失败: {error}")
        except requests.exceptions.ConnectionError:
            messagebox.showerror("连接错误", "无法连接到服务器")
        except Exception as e:
            messagebox.showerror("错误", str(e))
        finally:
            client.close()

    def _fetch_log(self):
        """获取训练日志"""
        if not self._check_login():
            return

        config_id = self.log_config_id.get().strip()
        if not config_id:
            messagebox.showwarning("警告", "请输入配置ID")
            return

        self._do_fetch_log(config_id)

    def _do_fetch_log(self, config_id: str):
        """实际执行日志获取"""
        client = self._get_client()
        try:
            resp = client.get(f"/task/log/{config_id}")
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "UNKNOWN")
                log_content = data.get("log_content", "")

                # 更新状态标签
                status_colors = {"RUNNING": "orange", "COMPLETED": "green", "FAILED": "red"}
                self.train_status_label.config(text=f"状态: {status}",
                                               foreground=status_colors.get(status, "black"))

                # 只显示最后一行日志
                self.train_log_text.config(state=tk.NORMAL)
                self.train_log_text.delete(1.0, tk.END)
                if log_content:
                    lines = [line for line in log_content.strip().split('\n') if line.strip()]
                    if lines:
                        self.train_log_text.insert(tk.END, lines[-1])
                    else:
                        self.train_log_text.insert(tk.END, "(暂无日志)")
                else:
                    self.train_log_text.insert(tk.END, "(暂无日志)")
                self.train_log_text.config(state=tk.DISABLED)

                return status
            else:
                error = resp.json().get("detail", "获取失败")
                self.train_log_text.config(state=tk.NORMAL)
                self.train_log_text.delete(1.0, tk.END)
                self.train_log_text.insert(tk.END, f"错误: {error}")
                self.train_log_text.config(state=tk.DISABLED)
                return "ERROR"
        except requests.exceptions.ConnectionError:
            return "CONNECTION_ERROR"
        except Exception as e:
            self._log(f"获取日志失败: {e}")
            return "ERROR"
        finally:
            client.close()

    def _toggle_log_monitor(self):
        """切换日志监控状态"""
        if self.log_monitor_running:
            self._stop_log_monitor()
        else:
            self._start_log_monitor()

    def _start_log_monitor(self):
        """启动日志监控"""
        if not self._check_login():
            return

        config_id = self.log_config_id.get().strip()
        if not config_id:
            messagebox.showwarning("警告", "请输入配置ID")
            return

        self.log_monitor_running = True
        self.monitor_btn.config(text="停止监控")
        self._log(f"开始监控日志: {config_id}")

        def monitor_loop():
            while self.log_monitor_running:
                status = self._do_fetch_log(config_id)
                if status in ("COMPLETED", "FAILED"):
                    self.log_monitor_running = False
                    self.root.after(0, lambda: self.monitor_btn.config(text="开始监控"))
                    self.root.after(0, lambda: self._log(f"任务已结束: {status}"))
                    break
                time.sleep(5)

        self.log_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.log_monitor_thread.start()

    def _stop_log_monitor(self):
        """停止日志监控"""
        self.log_monitor_running = False
        self.monitor_btn.config(text="开始监控")
        self._log("已停止日志监控")

    # ====================================================================
    # === 对话推理方法 ===
    # ====================================================================
    def _send_chat(self):
        """发送对话消息"""
        if not self._check_login():
            return

        config_id = self.chat_config_id.get().strip()
        if not config_id:
            messagebox.showwarning("警告", "请输入配置ID")
            return

        message = self.chat_input.get().strip()
        if not message:
            return

        # 清空输入框
        self.chat_input.delete(0, tk.END)

        # 显示用户消息
        self._append_chat("你", message)

        # 准备请求数据
        try:
            max_tokens = int(self.chat_max_tokens.get())
        except ValueError:
            max_tokens = 512

        try:
            temperature = float(self.chat_temperature.get())
        except ValueError:
            temperature = 0.7

        request_data = {
            "config_id": config_id,
            "prompt": message,
            "history": self.chat_history,
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }

        # 在后台线程发送请求
        def do_request():
            client = self._get_client()
            try:
                resp = client.post("/inference/chat", request_data)
                if resp.status_code == 200:
                    data = resp.json()
                    response = data.get("response", "")
                    time_taken = data.get("time_taken", 0)

                    # 更新对话历史
                    self.chat_history.append({"role": "user", "content": message})
                    self.chat_history.append({"role": "assistant", "content": response})

                    # 显示回复
                    self.root.after(0, lambda: self._append_chat("AI", f"{response}\n(耗时: {time_taken:.2f}s)"))
                else:
                    error = resp.json().get("detail", "请求失败")
                    self.root.after(0, lambda: self._append_chat("系统", f"错误: {error}"))
            except requests.exceptions.ConnectionError:
                self.root.after(0, lambda: self._append_chat("系统", "无法连接到服务器"))
            except Exception as e:
                self.root.after(0, lambda: self._append_chat("系统", f"错误: {e}"))
            finally:
                client.close()

        threading.Thread(target=do_request, daemon=True).start()

    def _append_chat(self, sender: str, message: str):
        """追加对话消息到显示区域"""
        self.chat_history_text.config(state=tk.NORMAL)
        self.chat_history_text.insert(tk.END, f"[{sender}]: {message}\n\n")
        self.chat_history_text.see(tk.END)
        self.chat_history_text.config(state=tk.DISABLED)

    def _clear_chat(self):
        """清空对话历史"""
        self.chat_history = []
        self.chat_history_text.config(state=tk.NORMAL)
        self.chat_history_text.delete(1.0, tk.END)
        self.chat_history_text.config(state=tk.DISABLED)
        self._log("对话历史已清空")

    # ====================================================================
    # === 日志方法 ===
    # ====================================================================
    def _log(self, message: str):
        """添加操作日志"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        """清空操作日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)


# ====================================================================
# === 入口点 ===
# ====================================================================
def main():
    # 导入 simpledialog（需要在 Tk 初始化后）
    import tkinter.simpledialog as simpledialog
    tk.simpledialog = simpledialog

    root = tk.Tk()
    app = CoserGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
