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
MLOps_Project
├── config
│   └── settings.py        # 全局统一配置中心（路径、超参、特征列表、版本规则）
├── data
│   └── telco_churn.csv    # 原始数据集
├── src
│   ├── data_process.py    # 数据清洗、特征工程、数据集划分
│   ├── train.py           # 模型训练、网格搜索、版本归档、SHAP绘图
│   ├── predict.py         # 模型加载、单样本推理
│   └── test_pred.py       # 预测功能单元测试
├── models                 # 模型版本归档目录，按版本分文件夹存储资产
├── assets                 # 可视化图表、超参搜索结果CSV
├── logs                   # 运行日志
├── requirements.txt       # 项目依赖包
└── README.md