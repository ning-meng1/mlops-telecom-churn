# 电信用户流失预测项目
基于XGBoost + SHAP可解释性的客户流失二分类建模工程，具备完整数据处理、超参网格搜索、模型版本归档、推理预测模块。

## 项目亮点

1. 📊 **样本不平衡分析**：统计标签分布，采用 `stratify` 保持训练集与测试集类别比例，并通过 `scale_pos_weight` 对照实验评估类别加权效果。
2. 🔁 **训练推理对齐**：训练与 API 推理复用统一的数据处理逻辑，避免训练与线上推理产生特征偏移。
3. 📦 **多版本模型管理**：按 `v1/v2` 独立归档模型资产，同时保存 scaler、特征列表、评估指标和训练元数据。
4. 🛡️ **接口容错**：支持必填字段、类型、数值范围校验，并统一捕获未预期异常。
5. 📝 **完整日志体系**：记录业务运行日志、Gunicorn 访问日志和错误日志，支持线上问题排查。
6. 🧪 **自动化测试**：覆盖健康检查、单条预测、批量预测以及参数异常、内部异常等场景。
7. 🐳 **容器工程化**：使用 Docker 完成推理服务容器化，支持 dev / prod 双模式，并通过锁定依赖版本保证环境一致性。

## 一、数据集介绍
### 业务背景
使用经典Telco Customer Churn电信用户数据集，模拟电信运营商业务场景：基于用户套餐、缴费、服务开通、消费时长等信息，构建模型预测客户是否会流失，帮助运营人员精准定位高流失风险用户，开展挽留营销。

- 样本总量：7043条用户记录
- 预测任务：二分类任务
- 目标标签 Churn
  - 1：客户发生流失（取消电信服务）
  - 0：客户正常留存

**核心字段分类**
1. 用户基础信息：gender、Partner、Dependents
2. 服务开通信息：PhoneService、InternetService、增值业务（安全备份、技术支持、流媒体服务等）
3. 合约与支付：Contract合约类型、PaymentMethod支付方式
4. 消费特征：tenure在网时长、MonthlyCharges月费、TotalCharges总消费

> 特征工程衍生字段：avg_monthly = TotalCharges / tenure，代表用户平均月消费压力

## 二、项目目录结构
```plaintext
mlops‑lite
├── assets                  # SHAP图、特征重要性图、超参搜索csv实验结果
│   ├── hyperparam_search_result_v1.csv
│   ├── hyperparam_search_result_v2.csv
│   ├── shap_bar_summary_v1.png
│   ├── shap_bar_summary_v2.png
│   ├── xgb_feature_importance_v1.png
│   └── xgb_feature_importance_v2.png
├── config
│   ├── __init__.py
│   └── settings.py         # 全局统一配置中心：路径、模型版本、特征规则、常量
├── data
│   └── telco_churn.csv     # 电信用户流失原始数据集
├── docs
│   ├── api_test.md                 # 接口文档、测试说明
│   ├── batch_test.json             # 批量预测正常请求样例
│   ├── curl_examples.sh            # curl调试命令脚本
│   ├── error_missing.json          # 异常用例：缺失特征
│   ├── error_range.json            # 异常用例：数值越界
│   ├── error_type.json             # 异常用例：类型错误
│   ├── postman_collection.json     # Postman接口集合导入文件
│   ├── scale_pos_weight_experiment.md # 样本不平衡scale_pos_weight对照实验记录
│   └── single_test.json            # 单条预测正常请求样例
├── logs                         # 服务运行日志持久化目录
│   ├── gunicorn.pid
│   ├── gunicorn_access.log
│   ├── gunicorn_error.log
│   └── mlops_service.log
├── models
│   ├── v1                  # v1版本模型（存在标签泄露缺陷，仅供对比）
│   └── v2                  # v2稳定版本（训练/推理特征对齐，线上可用）
│       ├── eval_metrics.json          # 模型评估指标(AUC、Acc、F1)
│       ├── std_scaler.pkl             # 特征标准化处理器
│       ├── train_feature_columns.txt  # 训练使用的特征清单
│       ├── train_metadata.json        # 训练元数据、最优超参数
│       └── xgb_best_model.pkl         # XGBoost训练模型文件
├── scripts
│   ├── experiment_scale_pos_weight.py # 样本不平衡scale_pos_weight对照实验脚本
│   ├── gunicorn.conf.py    # Gunicorn进程配置
│   ├── run.sh              # Docker统一启动入口，支持dev/prod
│   ├── start.sh            # 容器内部启动入口，根据ENV选择启动方式
│   └── start_gunicorn.sh   # 生产环境启动脚本
├── src
│   ├── __init__.py
│   ├── api_app.py          # Flask API层：接收HTTP、基础请求校验、路由、统一响应
│   ├── data_process.py     # 数据清洗、缺失值处理、特征工程、训练推理对齐逻辑
│   ├── logger.py           # 全局日志封装，统一日志格式与输出位置
│   ├── predict.py          # 底层模型加载、原始推理函数（只处理DataFrame）
│   ├── service.py          # Service业务层：编排流程、调用validator、组装数据、调用predict
│   ├── train_model.py      # XGBoost训练、超参搜索、SHAP可视化、模型保存
│   └── validator.py        # 特征校验模块：必填字段、类型、业务取值范围校验
├── tests
│   ├── __init__.py
│   ├── test_api.py         # API接口自动化测试（正常/异常场景）
│   ├── test_batch.py       # python脚本代替curl进行批量样本预测测试
│   ├── test_single.py      # python脚本代替curl进行单条样本预测测试
│   └── test_train.py       # 训练流程、模型加载单元测试
├── .gitignore              # Git忽略配置（虚拟环境、日志、缓存、畸形windows路径等不上传）
├── .dockerignore           # Docker构建忽略文件，不把本地venv、日志、大缓存打进镜像
├── Dockerfile              # Docker镜像构建文件，分层构建，支持dev/prod双模式
├── README.md               # 项目介绍、部署步骤、架构、接口文档
├── build.sh                # 镜像构建一键脚本，封装docker build命令
├── performance_test_record.md # 压测、性能测试记录文档
├── requirements.txt        # 基础依赖清单，写大版本，供开发环境使用
├── requirements‑lock.txt   # 锁定精确版本，保证环境一致性（Docker构建使用）
├── run‑dev.sh              # 一键启动容器开发模式（ENV=dev Flask）
└── run‑prod.sh             # 一键启动容器生产模式（ENV=prod Gunicorn）
```
💡 venv 虚拟环境目录已写入.gitignore，不上传 Git 仓库。

## 三、核心模块功能

### data_process.py
负责原始数据清洗、缺失值填充、特征衍生、类别编码、数据集划分。单独封装推理数据转换函数，自动适配训练 / 推理场景，解决线上无标签报错，保证训练推理数据逻辑完全一致。

### train_model.py
固定随机种子保证实验可复现，网格搜索完成 XGBoost 超参优选。输出准确率、F1、AUC 评估指标，结合 SHAP 完成模型可解释性可视化，自动归档模型、标准化器、特征列表、训练元数据。

### predict.py
实现模型全局缓存机制，服务启动一次性加载模型，避免重复磁盘 IO。支持单样本预测、批量预测，自动特征对齐补零、标准化，批量推理结束主动释放内存，避免内存溢出。

### api_app.py
搭建标准 Flask 推理服务，统一接口返回格式与全局异常捕获，提供健康检查、单条预测、批量预测三大接口。

### logger.py
全局日志
### service.py
业务分层核心（承接校验，业务逻辑编排，隔离API与底层逻辑）

### validator.py
独立特征校验工具，被service调用

## 四、版本迭代说明
### v1 版本
基础建模完成，可正常训练预测，但存在严重工程问题：训练数据包含标签列、推理无标签，导致训练推理不一致、存在标签泄露隐患，线上不可用。
### v2 最终修复版本
严格拆分训练集 X、y，杜绝标签数据泄露；数据处理增加字段存在性判断，适配线上推理无标签场景；统一训练、推理清洗逻辑，解决 KeyError 报错；批量接口输出工程标准双列表格式，满足线上接口规范。

## 五、模型优化实验

### 5.1 样本不平衡分析

Telco Customer Churn 数据集中：

- 不流失（No）：5174
- 流失（Yes）：1869
- 正负样本比例约为 1:2.77

因此针对 XGBoost 的 `scale_pos_weight` 进行了对照实验。

### 5.2 对照实验

保持随机种子、数据划分和模型超参数不变，仅修改 `scale_pos_weight`：

| 指标 | v2：scale_pos_weight=1 | 加权实验：2.7683 |
|---|---:|---:|
| Accuracy | 0.7956 | 0.7431 |
| AUC | 0.8457 | 0.8437 |
| Precision | 0.6536 | 0.5100 |
| Recall | 0.4893 | 0.8155 |
| F1 | 0.5596 | 0.6276 |

实验结果表明，类别加权后 Recall 和 F1 明显提高，同时 Precision 和 Accuracy 下降。

这说明类别加权会改变模型对正类的关注程度，需要结合实际业务目标在漏检和误报之间进行权衡。

当前项目仍保留 v2 作为正式模型，实验结果详见：

`docs/scale_pos_weight_experiment.md`

## 六、项目运行步骤
### 安装项目依赖

pip install -r requirements.txt
执行模型训练，生成 v2 版本模型

python src/train_model.py
启动 Flask 推理服务

python src/api_app.py
执行接口自动化测试

python tests/test_api.py
## 七、接口返回规范
### 单条预测返回格式
json
{
    "code": 0,
    "msg": "success",
    "data": {
        "pred_class": 0,
        "confidence": 0.1835,
        "label_desc": "0=不流失，1=流失",
        "model_version": "v2"
    }
}
### 批量预测返回格式
json
{
    "code": 0,
    "msg": "success",
    "data": {
        "pred_label": [0, 1],
        "pred_proba": [0.1835, 0.7501]
    }
}
### 🚀推理服务使用说明
服务启动

开发环境（Windows 本地开发）
python src/api_app.py

生产环境
（见第九章）

接口测试
💡 Windows PowerShell 用户请使用 curl.exe 代替 curl，避免 PowerShell 别名弹窗。
### 1. 健康检查接口 GET
curl http://127.0.0.1:5000/health
正常返回示例：
{"code":0,"data":{"model_loaded":true,"model_version":"v2","service_status":"running"},"msg":"service ok"}

### 2. 单样本预测接口 POST

请求：

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d @docs/single_test.json
```
说明：
-X POST：指定请求方式是 POST
-H：指定请求头，告诉 Flask 发送的是 JSON
-d @docs/single_test.json：
-d = data，发送请求数据
@ = 从文件读取数据
docs/single_test.json = 项目目录下 docs 文件夹里的测试文件

正常返回示例：
{
    "code":0,
    "msg":"success",
    "data":{
        "pred_class":0,
        "confidence":0.1835,
        "label_desc":"0=不流失，1=流失",
        "model_version":"v2"
    }
}

### 3. 批量预测接口 POST

请求：

```bash
curl -X POST http://127.0.0.1:5000/batch_predict \
-H "Content-Type: application/json" \
-d @docs/batch_test.json
```
正常返回示例：
{
    "code":0,
    "msg":"success",
    "data":{
        "pred_label":[0,1],
        "pred_proba":[0.1835,0.7501]
    }
}

## 八、接口测试记录
### Postman测试
已保存以下接口请求：

|接口|方法|用途|
|-|-|-|
|/health|GET|服务健康检查|
|/predict|POST|单用户预测|
|/batch_predict|POST|批量用户预测|
测试环境：

开发环境：
python src/api_app.py

生产环境：
gunicorn -w 2 -b 0.0.0.0:5000 src.api_app:app


测试结果：

- HTTP状态码正常
- 返回JSON结构正常
- code=0代表预测成功

## 九、生产部署说明
### Linux生产环境

安装：
pip install gunicorn

启动：
gunicorn -w 2 \
-b 0.0.0.0:5000 \
src.api_app:app

参数说明：
-w 2
表示启动2个worker进程
-b
绑定服务地址和端口
### Windows环境
gunicorn依赖Linux系统调用，Windows无法直接运行。

Windows生产模拟：
pip install waitress

启动：
waitress-serve \
--port=5000 \
src.api_app:app

## 十、工程稳定性与并发优化

### 1. 日志分级

项目采用分级日志机制：

- INFO：正常请求、模型加载、预测流程等运行信息
- WARNING：参数校验失败、非法请求等可预期异常
- ERROR：未捕获的内部异常，并记录完整Traceback
- CRITICAL：模型启动加载失败等严重错误

### 2. API参数强校验

API进入模型推理前进行参数校验：

- JSON格式校验
- 请求结构校验
- 必填字段校验
- 字段类型校验
- 数值范围校验
- 批量数据非空校验

通过接口层提前拦截非法请求，避免异常数据进入模型推理流程。

### 3. 全局异常处理

服务统一捕获未预期异常。

客户端仅返回：

{
    "code": 500,
    "msg": "服务器内部异常",
    "data": {}
}

服务端日志保留完整Traceback，方便线上问题定位，同时避免向客户端暴露内部实现细节。

### 4. 模型缓存优化

模型服务启动时预加载模型资产，并通过全局缓存复用模型。

避免每次API请求重复读取模型文件，降低磁盘IO和推理延迟。

### 5. Gunicorn生产部署

开发环境使用：

python src/api_app.py

Linux生产环境使用：

gunicorn -w 2 -b 0.0.0.0:5000 src.api_app:app

通过多个worker进程提升服务并发处理能力。

### 6. 自动化回归测试

通过tests/test_api.py自动验证：

- 健康检查
- 单样本预测
- 批量预测
- 参数缺失
- 参数类型错误
- 数值越界
- 空批量请求
- 内部异常
- 异常后服务恢复

测试结果：

PASS：14
FAIL：0
TOTAL：14

## 十一、Docker部署

### 1 环境要求

项目使用 Docker 完成模型推理服务的容器化部署。
训练代码及模型资产一并纳入项目镜像和工程目录，生产启动时直接加载已归档的 v2 模型提供推理服务。

建议环境：

- Docker Desktop / Docker Engine
- Git Bash（Windows用户）
- Linux / WSL 环境

项目不要求生产环境预先安装 Python 依赖，Python 运行环境及项目依赖由 Docker 镜像统一提供。

### 2 构建 Docker 镜像

在项目根目录执行：

```bash
docker build -t mlops-lite:day31 .
```
构建完成后查看镜像：
```bash
docker images
```
### 3 一键启动开发环境

使用统一启动脚本：

```bash 
bash scripts/run.sh dev
```
开发模式：

容器名称：mlops-dev-api
宿主机端口：5001
容器端口：5000
Flask 开发服务器
环境变量：ENV=dev

访问：
```bash
curl http://127.0.0.1:5001/health
```
### 4 一键启动生产环境

执行：

```bash 
bash scripts/run.sh prod
```
生产模式：

容器名称：mlops-prod-api
宿主机端口：5000
容器端口：5000
Gunicorn
环境变量：ENV=prod
--restart always 自动重启

访问：
```Bash
curl http://127.0.0.1:5000/health
```
### 5 查看运行中的容器
```bash
docker ps
```
正常情况下可以看到：
```
mlops-dev-api
mlops-prod-api

开发环境使用：
5001 -> 5000

生产环境使用：
5000 -> 5000
```
### 6 进入容器

查看开发容器：
```bash
docker exec -it mlops-dev-api /bin/bash
```
查看生产容器：
```bash
docker exec -it mlops-prod-api /bin/bash
```
进入容器后可以查看项目：
```bash
ls
```
以及模型：
```bash
ls -lh /app/models/v2
```
### 7 查看日志

宿主机日志目录：

logs/

容器内部日志目录：

/app/logs

项目通过 Bind Mount 将宿主机日志目录挂载到容器：
```bash
-v "$(pwd)/logs:/app/logs"
```
因此删除容器后，宿主机日志仍然保留。

### 8 模型持久化

项目通过 Bind Mount 将宿主机模型目录挂载到容器：
```bash
-v "$(pwd)/models:/app/models"
```
宿主机：

models/

对应容器：

/app/models/

因此删除并重新创建容器后，模型文件不会因为容器删除而丢失。

### 9 停止服务

停止开发服务：
```bash
docker rm -f mlops-dev-api
```
停止生产服务：
```bash
docker rm -f mlops-prod-api
```
### 10 Docker镜像优化

Dockerfile 使用：
```dockerfile
FROM python:3.13-slim
```
减少基础镜像体积。

安装 Python 依赖时使用：
```dockerfile
RUN pip install --no-cache-dir -r requirements-lock.txt
```
避免在镜像中保留 pip 缓存。

同时通过 .dockerignore 排除本地虚拟环境、日志及其他不需要进入镜像的文件。

本项目 Docker 镜像主要空间消耗来自：

```text
XGBoost
SHAP
NumPy
Pandas
Scikit-learn
```

因此虽然进行了基础镜像和缓存优化，最终镜像体积仍主要受到机器学习依赖影响。

优化前：
mlops-lite:day30 ≈ 1.67GB

优化后：
mlops-lite:day31 ≈ 1.68GB

本次优化重点并非单纯追求镜像体积下降，而是完成基础镜像、依赖安装方式、构建上下文和部署方式的工程化优化。

### 11 Dev / Prod 启动架构

项目通过 ENV 环境变量选择启动方式：
```
scripts/run.sh
      │
      ├── dev
      │     ↓
      │  run-dev.sh
      │     ↓
      │  ENV=dev
      │     ↓
      │  Flask
      │
      └── prod
            ↓
         run-prod.sh
            ↓
         ENV=prod
            ↓
         Gunicorn
```
这样可以使用同一个统一入口，根据启动参数选择开发或生产运行模式。




## 十二、项目踩坑总结
训练推理不一致：训练数据集包含标签，线上接口传入原始业务数据无标签，导致代码报错，通过字段判断兼容解决。
Python 缓存问题：__pycache__缓存旧字节码，修改代码不生效，需清空缓存并重启服务进程。
模型性能问题：每次请求重复加载模型 IO 耗时高，通过全局模型缓存优化接口速度。
标签泄露问题：v1 版本训练特征混入目标标签，模型虚高，v2 版本严格分离特征与标签。