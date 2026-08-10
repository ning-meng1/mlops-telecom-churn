# 电信用户流失预测项目
基于XGBoost + SHAP可解释性的客户流失二分类建模工程，具备完整数据处理、超参网格搜索、模型版本归档、推理预测模块。

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
mlops-lite
├── assets                  # SHAP特征重要性可视化图片、超参数搜索结果
├── config
│   ├── __init__.py
│   └── settings.py         # 全局路径、模型参数、文件路径统一配置中心
├── data
│   └── telco_churn.csv     # 电信用户流失原始业务数据集
├── ├── docs
│   ├── api_test.md                 # API测试记录
│   ├── curl_examples.sh            # curl调用示例
│   ├── postman_collection.json     # Postman接口集合
│   ├── single_test.json            # 单条预测测试数据
│   └── batch_test.json             # 批量预测测试数据
├── logs                    # 运行日志目录
├── models
│   ├── v1                  # v1版本归档模型（存在标签泄露BUG）
│   └── v2                  # v2修复版本（训练推理完全一致，生产可用）
│       ├── eval_metrics.json          # 模型AUC/F1/准确率评估指标
│       ├── std_scaler.pkl             # 特征标准化器
│       ├── train_feature_columns.txt  # 训练使用特征列表
│       ├── train_metadata.json        # 训练元信息、最优超参数
│       └── xgb_best_model.pkl         # XGBoost最优训练模型
├── src
│   ├── __init__.py
│   ├── api_app.py          # Flask推理服务、健康检查、单条/批量预测接口
│   ├── data_process.py    # 数据清洗、缺失值填充、特征衍生、训练推理对齐
│   ├── predict.py         # 模型缓存加载、单样本/批量预测推理
│   ├── test_api.py        # API接口测试脚本
│   └── train_model.py     # XGBoost训练、超参寻优、SHAP绘图、模型持久化
├── tests
│   └── test_train.py      # 模型加载与预测单元测试脚本
├── .gitignore              # Git忽略文件配置
├── README.md               # 项目说明文档
└── requirements.txt        # 项目所有依赖包清单
```
💡 venv 虚拟环境目录已写入.gitignore，不上传 Git 仓库。

## 三、核心模块功能
data_process.py
负责原始数据清洗、缺失值填充、特征衍生、类别编码、数据集划分。单独封装推理数据转换函数，自动适配训练 / 推理场景，解决线上无标签报错，保证训练推理数据逻辑完全一致。
train_model.py
固定随机种子保证实验可复现，网格搜索完成 XGBoost 超参优选。输出准确率、F1、AUC 评估指标，结合 SHAP 完成模型可解释性可视化，自动归档模型、标准化器、特征列表、训练元数据。
predict.py
实现模型全局缓存机制，服务启动一次性加载模型，避免重复磁盘 IO。支持单样本预测、批量预测，自动特征对齐补零、标准化，批量推理结束主动释放内存，避免内存溢出。
api_app.py
搭建标准 Flask 推理服务，统一接口返回格式与全局异常捕获，提供健康检查、单条预测、批量预测三大接口。
## 四、版本迭代说明
### v1 版本
基础建模完成，可正常训练预测，但存在严重工程问题：训练数据包含标签列、推理无标签，导致训练推理不一致、存在标签泄露隐患，线上不可用。
### v2 最终修复版本
严格拆分训练集 X、y，杜绝标签数据泄露；数据处理增加字段存在性判断，适配线上推理无标签场景；统一训练、推理清洗逻辑，解决 KeyError 报错；批量接口输出工程标准双列表格式，满足线上接口规范。
## 五、项目运行步骤
### 安装项目依赖

pip install -r requirements.txt
执行模型训练，生成 v2 版本模型

python src/train_model.py
启动 Flask 推理服务

python src/api_app.py
执行接口自动化测试

python src/test_api.py
## 六、接口返回规范
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
开发环境（全平台可用，Windows 本地开发）
python src/api_app.py

生产环境
（见第八章）

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

## 七、接口测试记录
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

## 八、生产部署说明
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

## 九、项目踩坑总结
训练推理不一致：训练数据集包含标签，线上接口传入原始业务数据无标签，导致代码报错，通过字段判断兼容解决。
Python 缓存问题：__pycache__缓存旧字节码，修改代码不生效，需清空缓存并重启服务进程。
模型性能问题：每次请求重复加载模型 IO 耗时高，通过全局模型缓存优化接口速度。
标签泄露问题：v1 版本训练特征混入目标标签，模型虚高，v2 版本严格分离特征与标签。