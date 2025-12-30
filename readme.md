## Coser 云端人格大模型训练服务框架说明

### 快速开始

1. 环境部署

```
cd Coser-main
pip install -r requirements.txt
```

2. 配置模型路径

修改settings.json为服务端python路径和大模型路径，例如：

```
{
    "python_bin": "/usr/bin/python3",
    "model_path": "/models/deepseek-r1-7b/"
}
```

3. 启动后端服务

```
uvicorn Core.api_service:app --host 127.0.0.1 --port 8000
```

4. 服务端本地测试GUI：
```
cd scripts
python GUI.py
```

可以使用预定义的example用户登录：

- 用户名：example
- 密码：123456
- 可用数据集：firefly.json 
- 可用配置：firefly
- 数据集来源：https://www.modelscope.cn/datasets/firefly123123/firefly

### 一、项目定位与目标

Coser 面向云端部署，核心目标是：利用自有本地服务器或私有云服务器的 GPU 算力，对外提供“特定人格的大模型训练与推理”云服务。使用者无需直接登录服务器，只需通过命令行工具或图形界面客户端，就可以远程完成人格配置、训练发起、推理调用和日志查看等操作。

项目旨在把一台或多台本地 GPU 服务器封装成一个统一的“人格大模型训练云平台”，对外以服务的方式提供：
1. 针对特定角色／人设的数据微调能力。
2. 多人格、多模型的集中管理与在线推理。
3. 基于云端的训练任务调度与日志管理。

### 二、整体架构概览

在典型部署中，Coser 运行在 Linux 服务器上，作为长期在线的后端服务，对外提供统一 API 接口。外部用户通过 CLI 客户端或 GUI 客户端连接服务器，完成所有训练与推理请求。

主要组成部分如下（以项目根目录为 `/home/user/Coser-main` 为例）：

1. `cli.py`  
   - 作为命令行入口，封装所有与后端交互的操作。  
   - 支持发起训练任务、执行推理、修改训练参数、查询训练日志等。  
   - 可以部署在服务器本机，也可以在能访问服务器的任意终端上作为“云服务客户端”。

2. `scripts/GUI.py`  
   - 基于 tkinter 的桌面图形客户端。  
   - 设计为可脱离项目目录，在任意客户端机器（如 Windows）单独运行。  
   - 通过网络或 SSH 调用服务器上的服务，实现可视化配置与训练监控。

3. `Core/` 核心服务目录  
   - `api_service.py`：后端云服务接口层，将本地算力和模型能力封装为统一 API，是“人格模型云服务”的核心。  
   - `finetune.py`：执行 LoRA 微调的训练逻辑，在服务器 GPU 上运行。  
   - `infer.py`：在线推理与对话逻辑，为各人格模型提供推理服务。  
   - `model_manager.py`：负责模型和 LoRA 权重的加载与切换，支撑多人格多模型。  
   - `scheduler.py`：训练任务的调度管理，可扩展为更复杂的云端任务队列。  
   - `schemas.py`：请求、响应和配置的数据结构定义，是扩展参数与接口的基础。  
   - `user.py`：用户与简单鉴权相关逻辑。  
   - `utils.py`：通用工具函数。

4. 数据与资源目录  
   - `data/`：缓存数据、对话历史等运行时信息。  
   - `logs/`：按角色／人格划分的训练日志，支持远程查询与排错。  
   - `user_configs/`：用户与模型的配置文件，包括训练参数、人设信息、模型选择等。  
   - `weights/`：各人格的 LoRA 微调权重和中间 checkpoint，是云端推理和持续训练的基础。  

5. 其他  
   - `settings.json`：云服务提供的服务端资源路径。
   - `requirements.txt`：依赖列表，用于在服务器端快速部署运行环境。  
   - `readme.md`：项目说明文档。

### 三、云端服务能力与使用场景

1. 人格大模型云服务能力

通过 `Core/api_service.py`，Coser 将训练与推理能力以服务形态对外提供，支持的核心操作包括：
- 为某个“人格”创建或更新训练配置（例如某个虚构角色、IP 形象、品牌人格等）。  
- 在服务器上发起 LoRA 微调任务，利用云端 GPU 资源进行训练。  
- 查询训练进度、查看实时日志或历史日志文件。  
- 加载指定人格模型并进行在线对话测试或推理调用。  
- 管理模型版本与权重文件，实现多轮迭代与回滚。

训练与推理中可配置的典型参数包括（但不限于）：
- `lora_r`  
- `lora_alpha`  
- `lora_dropout`  
- `batch_size`  
- `learning_rate`  
- `epochs`  
- `max_seq_len`  

这些参数由客户端通过网络传递到服务器，由 `api_service` 解析后传入训练和推理流程，实现“远程调参 + 远程训练”。

2. CLI 客户端使用场景

`cli.py` 可以被视作面向开发者和高级用户的云服务客户端，典型用法包括：
- 在本地准备好某人格的训练数据，然后通过 CLI 上传或挂载到服务器，再用一条命令触发对应人格的训练。  
- 多名开发者共用同一台或一组 GPU 服务器，各自通过 CLI 发起自己的训练任务。  
- 编写脚本批量发起多个不同人格的训练，或进行自动化评估与回归测试。

CLI 适合集成到自动化流程、CI/CD、内部工具脚本中，实现对“人格训练云平台”的程序化控制。

CL提供的命令行示例：
```
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
```

3. GUI 图形客户端使用场景

`GUI.py` 为不熟悉命令行的终端用户提供了可视化访问方式，可在 Windows、Linux 或 macOS 上独立运行。主要功能包括：
- 直观地查看和修改各人格模型的训练配置参数，如 `epochs`、`learning_rate`、`lora_r`、`lora_alpha`、`lora_dropout`、`batch_size`、`max_seq_len` 等。  
- 点击按钮发起云端训练任务，并在窗口中实时显示训练日志输出。  
- 选择某个人格，在云端加载对应模型，并在 GUI 中进行对话测试。  
- 浏览当前已有的角色／人格以及对应模型权重与训练记录。

后端服务通常在服务器上常驻运行，GUI 只负责发起请求和展示结果，不参与服务端启动与维护。

### 四、服务器端环境要求与部署

1. 环境要求
- 操作系统：Linux（可为物理机、虚拟机或容器环境）。  
- Python 版本：建议 3.10 及以上。  
- GPU：推荐带 CUDA 的显卡，用于大模型训练和推理。  

2. 依赖安装

在服务器的项目根目录（如 `/home/user/Coser-main`）执行：

- 使用 pip 安装项目依赖，确保 PyTorch 等深度学习框架版本与 CUDA 环境匹配。  
- 根据需要配置基础模型路径、默认用户配置等。

3. 后端服务部署

后端主要由 `Core/api_service.py`（以及可能的 `Core/service_cli.py`）提供服务逻辑。典型部署步骤：
- 将项目代码部署到服务器，并按需配置基础模型和缓存目录。  
- 安装依赖后，使用项目内提供的命令或脚本启动 API 服务进程。  
- 建议结合 systemd、Docker、screen 或 tmux 等工具，将服务以守护进程方式长期运行。  
- 按实际安全策略配置访问方式：  
  - 仅内网可访问；  
  - 通过 VPN 访问；  
  - 通过反向代理或域名对公网开放，并配合鉴权与访问控制。

服务部署完成后，只要用户掌握服务器的地址、端口及凭证，即可通过 CLI 或 GUI 客户端使用该人格模型云服务。

### 五、远程 GUI 与 SSH 结合示例

在一个常见场景中，云端服务运行在 Linux 服务器上，终端用户在 Windows 电脑上通过 SSH 或网络访问该服务。

典型步骤为：
1. 在 Windows 本地安装 Python，并确保包含 tkinter。  
2. 将 `GUI.py` 拷贝到 Windows 本地任意目录。  
3. 使用 SSH 命令连接到服务器，确认网络连通和账号密码正确。  
4. 在 `GUI.py` 中配置服务器地址和端口（例如通过 HTTP API 或通过 SSH 执行远程命令的方式）。  
5. 在 Windows 本地运行 `python GUI.py`，使用图形界面对服务器上的人格训练云服务进行操作。

在这种架构下，Windows 只负责界面与控制逻辑，真正的训练和推理全在云端服务器执行。

### 六、云端日志与模型资源管理

在服务器端，Coser 对资源按人格／角色进行分层组织，以便远程管理：
- `logs/`：记录各人格的训练日志，可通过 CLI 或 GUI 查看，用于调试与监控。  
- `weights/`：存放每个角色对应的 LoRA 微调结果和 checkpoint，是云端推理与多版本管理的核心。  
- `user_configs/`：存放用户和人格相关配置文件，包括训练参数、人设描述、基础模型选择等，可通过服务接口动态修改。

客户端只通过接口访问这些资源，并不直接操作服务器文件，从而保持统一管理和安全控制。

### 七、扩展与二次开发

Coser 从设计上面向“云服务化”，便于后续扩展为更完整的多租户云平台或内部私有云服务。

可扩展方向包括：
1. 参数与接口扩展  
   - 在 `schemas.py` 中增加新的配置字段（例如新的正则化参数或评估指标）。  
   - 在 `api_service.py` 中实现对应的参数解析和校验逻辑。  
   - 在 `cli.py` 与 `GUI.py` 中增加对应的命令行参数和输入控件。

2. 新人格与新模型  
   - 在 `weights/`、`user_configs/`、`data/` 中增加新的人格配置与训练数据。  
   - 在 `model_manager.py` 中注册新基础模型或新的人格模型加载逻辑。

3. 云服务能力增强  
   - 在 `api_service.py` 中增加账号、多租户、权限控制、配额限制、任务排队等云平台特性。  
   - 在 `scheduler.py` 中增加更复杂的任务调度算法，例如按优先级或按资源配额分配 GPU。

4. 前端能力提升  
   - 在 CLI 中增加批量任务、自动评估、定时任务等命令。  
   - 在 GUI 中增加训练任务列表、资源使用情况可视化（GPU 占用、显存、吞吐等），进一步向完整的“云管理控制台”演进。

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

