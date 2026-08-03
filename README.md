# 电信用户流失预测项目
基于XGBoost + SHAP可解释性的客户流失二分类建模工程，具备完整数据处理、超参网格搜索、模型版本归档、推理预测模块。

## 一、数据集介绍
### 业务背景
使用经典Telco Customer Churn电信用户数据集，模拟电信运营商业务场景：基于用户套餐、缴费、服务开通、消费时长等信息，构建模型预测客户是否会流失，帮助运营人员精准定位高流失风险用户，开展挽留营销。

- 样本总量：7043条用户记录
- 预测任务：二分类任务
- 目标标签 `Churn`
  - 1：客户发生流失（取消电信服务）
  - 0：客户正常留存

### 核心字段分类
1. 用户基础信息：gender、Partner、Dependents
2. 服务开通信息：PhoneService、InternetService、增值业务（安全备份、技术支持、流媒体服务等）
3. 合约与支付：Contract合约类型、PaymentMethod支付方式
4. 消费特征：tenure在网时长、MonthlyCharges月费、TotalCharges总消费
> 特征工程衍生字段：avg_monthly = TotalCharges / tenure，代表用户平均月消费压力

## 二、项目目录结构
```plaintext
mlops-lite
├── assets # SHAP 特征重要性可视化图片、超参数搜索结果
├── config
│ ├── init.py
│ └── settings.py # 全局路径、模型参数、文件路径统一配置中心
├── data
│ └── telco_churn.csv # 电信用户流失原始业务数据集
├── logs # 运行日志目录
├── models
│ └── v1 # v1 版本归档模型
│ ├── eval_metrics.json # 模型 AUC/F1 / 准确率评估指标
│ ├── std_scaler.pkl # 特征标准化器
│ ├── train_feature_columns.txt # 训练使用特征列表
│ ├── train_metadata.json # 训练元信息、最优超参数
│ └── xgb_best_model.pkl # XGBoost 最优训练模型
├── src
│ ├── init.py
│ ├── api_app.py # Day15 Flask 推理服务、健康检查接口
│ ├── data_process.py # 数据清洗、缺失值填充、特征衍生、数据集划分
│ ├── predict.py # 模型加载、单样本 / 批量预测推理
│ └── train_model.py # XGBoost 训练、超参数寻优、SHAP 绘图、模型持久化
├── tests
│ └── test_train.py # 模型加载与预测单元测试脚本
├── venv # 本地虚拟环境（.gitignore 忽略不上传）
├── .gitignore # Git 忽略文件配置
├── README.md # 项目说明文档
└── requirements.txt # 项目所有依赖包清单