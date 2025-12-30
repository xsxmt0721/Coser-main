# Coser/Core/schemas.py

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any, Union


# --- 认证 (Auth) 模型 ---
class AuthBase(BaseModel):
    """基础认证模型，用于登录和注册"""
    username: str = Field(..., min_length=4, max_length=20, description="用户名")
    pwd: str = Field(..., min_length=6, description="密码 (明文存储)")


class UserRegister(AuthBase):
    """用户注册模型"""
    pass


class UserLogin(AuthBase):
    """用户登录模型"""
    pass


# --- 配置文件管理 (Config) 模型 ---

class LoRAConfig(BaseModel):
    """LoRA训练参数配置"""
    learning_rate: float = Field(2e-4, gt=0, description="学习率")
    epochs: int = Field(3, gt=0, description="训练轮数")
    batch_size: int = Field(4, gt=0, description="批次大小")
    lora_r: int = Field(32, gt=0, description="LoRA秩 (R)")
    lora_alpha: int = Field(64, gt=0, description="LoRA Alpha")
    lora_dropout: float = Field(0.05, ge=0, lt=1, description="LoRA Dropout")
    max_seq_len: int = Field(1024, gt=0, description="最大序列长度")


class ConfigBase(BaseModel):
    """训练配置文件的基础结构 (与 user_configs/{user}/{id}.json 对应)"""
    ID: str = Field(..., description="配置/角色ID (e.g., firefly)")
    USERNAME: str = Field(..., description="所属用户名")
    MODEL_PATH: str = Field(..., description="基础模型路径 (e.g., /path/to/deepseek-r1-8b)")
    DATA_PATH: str = Field("", description="数据集文件的服务器绝对路径")
    WEIGHT_PATH: str = Field("", description="LoRA权重保存目录的基础路径")

    # 将 LoRA 参数嵌套在内部
    DEFAULT_LORA_PARAMS: LoRAConfig = Field(..., description="训练参数")


class ConfigCreate(BaseModel):
    """新建配置时需要传入的参数"""
    config_id: str = Field(..., min_length=1, description="新配置ID (用于文件名)")


class ConfigUpdate(BaseModel):
    """修改现有配置参数的模型"""
    # 使用 Optional 允许只修改部分参数
    data_path: Optional[str] = Field(None, description="新的数据集服务器路径")
    # 允许直接传入完整的 LORA 参数对象进行覆盖，或只传入部分修改的 LORA 参数
    lora_params: Optional[Dict[str, Union[int, float, str]]] = Field(None, description="要修改的LoRA参数键值对")


# --- 数据集管理 (Dataset) 模型 ---

class DatasetUpload(BaseModel):
    """用户上传数据集时的请求体"""
    dataset_name: str = Field(..., min_length=1, description="数据集在服务器上的存储名称 (不含.json)")


class DatasetDelete(BaseModel):
    """删除数据集时的请求体"""
    dataset_name: str = Field(..., min_length=1, description="要删除的数据集名称 (不含.json)")


# --- 任务调度 (Task) 模型 ---

class TaskRequest(BaseModel):
    """启动模型训练或对话的基础请求"""
    config_id: str = Field(..., description="要使用的训练配置ID")


class TrainRequest(TaskRequest):
    """
    启动模型训练的请求体。
    允许用户临时覆盖训练参数，但不修改配置文件。
    """
    # 继承 TaskRequest: config_id
    # 允许在训练时临时覆盖 LoRA 参数
    epochs: Optional[int] = Field(None, gt=0, description="临时覆盖训练轮数 (epochs)")
    learning_rate: Optional[float] = Field(None, gt=0, description="临时覆盖学习率 (learning_rate)")
    batch_size: Optional[int] = Field(None, gt=0, description="临时覆盖批次大小 (batch_size)")
    lora_r: Optional[int] = Field(None, gt=0, description="临时覆盖 LoRA 秩 (R)")
    lora_alpha: Optional[int] = Field(None, gt=0, description="临时覆盖 LoRA Alpha")
    lora_dropout: Optional[float] = Field(None, ge=0, lt=1, description="临时覆盖 LoRA Dropout")
    max_seq_len: Optional[int] = Field(None, gt=0, description="临时覆盖最大序列长度")

class TrainStartResponse(BaseModel):
    """
    模型训练任务启动后的响应。
    """
    config_id: str = Field(..., description="已启动训练任务的配置ID")
    task_id: str = Field(..., description="任务的唯一标识符 (例如：config_id + timestamp)")
    log_file_path: str = Field(..., description="日志文件在服务器上的绝对路径，用于查看训练进度。")
    message: str = Field("训练任务已在后台异步启动。", description="状态消息")


class TaskLogResponse(BaseModel):
    """
    异步任务状态和日志查询响应。
    """
    config_id: str
    log_file_path: str = Field(..., description="最新的日志文件在服务器上的路径")

    # 检索任务的简要状态
    status: str = Field(..., description="任务的简要状态 (RUNNING, COMPLETED, FAILED, NOT_FOUND)")

    # 日志内容，用于展示训练进度
    log_content: str = Field(..., description="日志文件的最新内容 (例如最后 100 行)")

    # 可选字段：如果任务完成，返回完成时间
    completion_time: Optional[str] = Field(None, description="任务完成时间 (如果已完成)")


class ChatRequest(BaseModel):
    """
    用户发送对话请求的请求体。
    """
    config_id: str = Field(..., description="要使用的微调配置ID（决定 LoRA 权重）")
    prompt: str = Field(..., min_length=1, description="用户输入的指令或问题")
    # 允许临时覆盖生成参数
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="生成温度 (0.0 - 1.0)")
    max_length: Optional[int] = Field(None, gt=0, description="最大生成长度")

class ChatResponse(BaseModel):
    """
    模型对话响应。
    """
    config_id: str = Field(..., description="当前使用的配置ID")
    response: str = Field(..., description="模型生成的回复文本")
    time_taken: float = Field(..., description="推理耗时 (秒)")


# --- 状态与日志 (Status) 模型 ---

class TaskStatusResponse(BaseModel):
    """异步任务状态和日志查询响应"""
    config_id: str
    log_file_path: str = Field(..., description="最新的日志文件在服务器上的路径")
    status_summary: str = Field(..., description="任务的简要状态 (e.g., RUNNING, COMPLETED, FAILED)")
    log_content: str = Field(..., description="日志文件的最新内容 (例如最后 100 行)")