# Coser/Core/api_service.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, Any, List, Optional

# 解决模块导入路径问题：将项目根目录添加到 sys.path
import sys
import os

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CORE_DIR)
sys.path.append(PROJECT_ROOT)

# 导入核心业务逻辑和数据模型
from Core import user
from Core.user import get_user_config_data, save_config_file
from Core.schemas import (
    UserRegister,
    UserLogin,
    ConfigCreate,
    ConfigUpdate,
    DatasetUpload,
    DatasetDelete,
    LoRAConfig
)

from Core import scheduler
from Core.schemas import TrainRequest, TrainStartResponse, TaskLogResponse
from Core import model_manager
from Core.schemas import ChatRequest, ChatResponse

# --- 1. 应用初始化 ---
app = FastAPI(
    title="LLM Cloud Service Backend",
    description="基于 DeepSeek 和 LoRA 的多用户模型训练与推理服务 API",
    version="1.0.0"
)

# 简单的基于用户名的认证（实际项目中需使用 JWT）
# 为了简化，我们使用全局字典模拟已认证用户的会话
# Key: Session/Token ID, Value: Username
ACTIVE_SESSIONS: Dict[str, str] = {}


# --- 2. 依赖注入：认证函数 (Dependency Injection for Auth) ---
# 这是一个简化的认证流程，用于确保用户已登录
def get_current_user(session_token: str) -> str:
    """检查会话 token 是否有效，并返回用户名"""
    if session_token not in ACTIVE_SESSIONS:
        # HTTP 401 Unauthorized
        raise HTTPException(
            status_code=401,
            detail="会话无效或未登录，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return ACTIVE_SESSIONS[session_token]


# ====================================================================
# === A. 认证 (Auth) 路由器：/auth
# ====================================================================

@app.post("/auth/register", summary="用户注册")
def register(new_user: UserRegister):
    """
    注册新用户，创建目录结构，并将用户信息写入 users.json。
    """
    error = user.register_user(new_user.username, new_user.pwd)

    if error:
        raise HTTPException(status_code=400, detail=f"注册失败: {error}")

    return {"message": "注册成功，请使用新账号登录"}


@app.post("/auth/login", summary="用户登录")
def login(login_info: UserLogin):
    """
    用户登录，检查用户名和密码，并返回一个会话 Token (即用户名本身)。
    """
    config_data = user.login_user(login_info.username, login_info.pwd)

    if config_data is None:
        raise HTTPException(status_code=401, detail="登录失败：用户名或密码错误")

    # 模拟生成一个 Session Token (为了简单，直接使用用户名作为 Token)
    session_token = login_info.username
    ACTIVE_SESSIONS[session_token] = login_info.username

    return {
        "message": f"登录成功，欢迎 {login_info.username}",
        "session_token": session_token,
        "config_data": config_data  # 登录时返回配置列表，方便客户端初始化
    }


@app.post("/auth/logout", summary="用户注销")
def logout(current_user: str = Depends(get_current_user)):
    """
    用户注销，销毁会话 Token。
    """
    # 查找并删除所有与该用户相关的会话
    tokens_to_remove = [token for token, username in ACTIVE_SESSIONS.items() if username == current_user]

    for token in tokens_to_remove:
        del ACTIVE_SESSIONS[token]

    return {"message": f"用户 {current_user} 注销成功"}


# ====================================================================
# === B. 配置管理 (Config) 路由器：/config
# ====================================================================

@app.get("/config/all", summary="获取用户所有训练配置")
def get_configs(current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """
    获取当前用户所有的模型训练配置（以字典形式返回）。
    """
    return get_user_config_data(current_user)


@app.post("/config/new", summary="新建训练配置")
def create_config(
        new_config: ConfigCreate,
        current_user: str = Depends(get_current_user)
):
    """
    为当前用户创建一个新的训练配置文件，基于默认模板。
    """
    error = user.create_new_config(current_user, new_config.config_id)

    if error:
        raise HTTPException(status_code=400, detail=f"新建配置失败: {error}")

    return {"message": f"配置 '{new_config.config_id}' 创建成功"}


@app.put("/config/{config_id}", summary="修改配置参数")
def update_config(
        config_id: str,
        update: ConfigUpdate,
        current_user: str = Depends(get_current_user)
):
    """
    修改指定配置的 DATA_PATH 或 LORA 训练参数。
    """
    # 1. 加载当前配置
    all_configs = get_user_config_data(current_user)
    if config_id not in all_configs:
        raise HTTPException(status_code=404, detail=f"配置ID '{config_id}' 不存在")

    config = all_configs[config_id]

    # 2. 修改 DATA_PATH
    if update.data_path is not None:
        # 验证数据集路径是否存在 (user.py 模块已检查)
        data_path_check = os.path.join(user.DATA_DIR, current_user, update.data_path)
        if not os.path.exists(data_path_check):
            raise HTTPException(status_code=400, detail=f"数据集文件 {update.data_path} 不存在于服务器")

        config['DATA_PATH'] = data_path_check

    # 3. 修改 LORA 参数
    if update.lora_params is not None:
        lora_params = config['DEFAULT_LORA_PARAMS']

        for key, value in update.lora_params.items():
            if key in lora_params:
                # 尝试类型转换和校验（这里依赖 Pydantic 的校验能力）
                try:
                    # 使用 Pydantic 模型来校验和转换类型
                    lora_model = LoRAConfig(**{key: value, **{k: v for k, v in lora_params.items() if k != key}})
                    lora_params[key] = getattr(lora_model, key)
                except Exception as e:
                    raise HTTPException(status_code=400,
                                        detail=f"LORA参数修改失败: 键 '{key}' 值 '{value}' 无效或类型错误 ({e})")
            else:
                raise HTTPException(status_code=400, detail=f"LORA参数键 '{key}' 不存在或不可修改")

    # 4. 保存修改
    save_config_file(current_user, config_id, config)
    return {"message": f"配置 '{config_id}' 更新成功", "config": config}


# ====================================================================
# === C. 数据集管理 (Dataset) 路由器：/dataset
# ====================================================================

@app.get("/dataset/list", summary="列出用户所有数据集")
def list_datasets(current_user: str = Depends(get_current_user)) -> List[str]:
    """
    列出当前用户在 data/USERNAME/ 目录下的所有数据集文件 (.json)。
    """
    return user.list_user_datasets(current_user)


@app.post("/dataset/upload", summary="【新】从 Cache 导入数据集")
def upload_dataset(
        upload_info: DatasetUpload,
        current_user: str = Depends(get_current_user)
):
    """
    从 Cache 目录中导入以 {username}.{dataset_name}.json 命名的文件，并将其移动到用户的个人目录。
    """
    # 调用新的 upload_dataset 逻辑，不再需要 local_path
    error = user.upload_dataset(
        current_user,
        upload_info.dataset_name
    )

    if error:
        # 400 Bad Request: Cache 文件不存在或导入失败
        raise HTTPException(status_code=400, detail=f"导入失败: {error}")

    return {"message": f"数据集 '{upload_info.dataset_name}.json' 从 Cache 成功导入到用户目录"}


@app.delete("/dataset/delete", summary="删除数据集")
def delete_dataset(
        delete_info: DatasetDelete,
        current_user: str = Depends(get_current_user)
):
    """
    删除用户数据仓库中的指定数据集文件。
    """
    error = user.delete_user_dataset(
        current_user,
        delete_info.dataset_name
    )

    if error:
        raise HTTPException(status_code=404, detail=f"删除失败: {error}")

    return {"message": f"数据集 '{delete_info.dataset_name}.json' 删除成功"}


@app.delete("/config/{config_id}", summary="删除指定配置和关联的模型权重")
def delete_config(
        config_id: str,
        current_user: str = Depends(get_current_user)
):
    """
    删除用户的指定配置文件 (user_configs/USER/{ID}.json)，
    并递归删除关联的模型权重目录 (weights/USER/{ID})。
    """
    error = user.delete_user_config(current_user, config_id)

    if error:
        # 如果配置不存在，返回 404
        if "不存在" in error:
            raise HTTPException(status_code=404, detail=error)
        # 如果是删除系统错误，返回 400
        raise HTTPException(status_code=400, detail=f"删除配置失败: {error}")

    return {"message": f"配置 '{config_id}' 及其关联的权重已成功删除"}


@app.post("/task/train", response_model=TrainStartResponse, summary="启动模型训练任务")
def start_train(
        request: TrainRequest,
        current_user: str = Depends(get_current_user)
):
    """
    启动指定配置的训练任务，支持临时覆盖 LoRA 参数。
    任务在后台启动，立即返回任务ID。
    """
    # 1. 收集覆盖参数
    overrides = request.model_dump(exclude_none=True)
    # 移除 config_id，只保留需要覆盖的 LoRA 参数
    overrides.pop('config_id', None)

    # 2. 获取并合并配置
    merged_config = scheduler.get_config_with_overrides(
        config_id=request.config_id,
        username=current_user,
        overrides=overrides
    )

    if merged_config is None:
        raise HTTPException(status_code=404, detail=f"配置ID '{request.config_id}' 不存在")

    # 校验 DATA_PATH 是否设置
    if not os.path.exists(merged_config['DATA_PATH']):
        raise HTTPException(status_code=400,
                            detail=f"数据集路径 '{merged_config['DATA_PATH']}' 无效或文件不存在，请先配置数据集。")

    # 3. 启动后台任务
    try:
        task_id = scheduler.start_training_task(
            username=current_user,
            config_id=request.config_id,
            merged_config=merged_config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练任务启动失败: {e}")

    # 4. 返回任务信息
    task_info = scheduler.ACTIVE_TRAINING_TASKS[request.config_id]
    return TrainStartResponse(
        config_id=request.config_id,
        task_id=task_id,
        log_file_path=task_info['log_path'],
        message="训练任务已在后台异步启动。"
    )


@app.get("/task/log/{config_id}", response_model=TaskLogResponse, summary="查询训练任务状态和日志")
def get_train_log(config_id: str, current_user: str = Depends(get_current_user)):
    """
    获取指定配置的训练任务的最新状态和日志内容。
    """
    # 注意：这里我们使用 config_id 作为键来查询任务，这隐含了同一时间一个用户只能对一个配置ID启动一次训练
    task_info = scheduler.get_training_task_status(config_id)

    if task_info['status'] == 'NOT_FOUND':
        raise HTTPException(status_code=404, detail="该配置ID下无活动或最近的任务记录。")

    # 构造响应模型
    return TaskLogResponse(
        config_id=config_id,
        log_file_path=task_info.get('log_path', 'N/A'),
        status=task_info['status'],
        log_content=task_info['log_content']
    )


@app.post("/inference/chat", response_model=ChatResponse, summary="使用指定配置的模型进行对话")
def chat(
        request: ChatRequest,
        current_user: str = Depends(get_current_user)
):
    """
    使用 ModelManager 加载/切换 LoRA 权重，并生成对话响应。
    """
    # 1. 提取配置数据
    all_configs = get_user_config_data(current_user)
    if request.config_id not in all_configs:
        raise HTTPException(status_code=404, detail=f"配置ID '{request.config_id}' 不存在。")

    config_data = all_configs[request.config_id]

    # 2. 构造 LoRA 权重路径并检查
    WEIGHT_SAVE_DIR_BASE = config_data['WEIGHT_PATH']
    LORA_WEIGHT_PATH = os.path.join(WEIGHT_SAVE_DIR_BASE, current_user, request.config_id, "final_lora_weights")
    if not os.path.isdir(LORA_WEIGHT_PATH):
        raise HTTPException(status_code=400, detail=f"找不到配置 '{request.config_id}' 的 LoRA 权重。请先完成训练。")

    try:
        # 3. 调用 ModelManager 进行生成
        response_text, time_taken = model_manager.model_manager.generate_response(
            username=current_user,
            request=request,
            config_data=config_data
        )

        return ChatResponse(
            config_id=request.config_id,
            response=response_text,
            time_taken=time_taken
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 捕获 GPU OOM 或其他运行时错误
        raise HTTPException(status_code=500, detail=f"推理时发生系统错误: {e}")

if __name__ == '__main__':
    # 启动 Uvicorn 服务器
    import uvicorn

    print("\n--- 启动 FastAPI 服务 ---")
    print("访问 http://127.0.0.1:8000/docs 查看 API 文档")
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

    # cmd
    # uvicorn Core.api_service:app --host 127.0.0.1 --port 8000