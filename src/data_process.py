import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from config.settings import BASE_DIR, DATA_PATH

# 确保数据目录存在
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)


def load_and_process_data():
    # 1. 读取数据
    data = pd.read_csv(DATA_PATH)
    print("✅ 数据加载完成")
    print(f"数据集形状: {data.shape}")

    # 2. 处理TotalCharges列（空字符串转为数值）
    data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    print(f"TotalCharges缺失值数量: {data['TotalCharges'].isnull().sum()}")

    # 3. 缺失值填充
    data['TotalCharges'] = data['TotalCharges'].fillna(data['MonthlyCharges'])
    print("✅ 缺失值处理完成")

    # 4. 派生特征计算 + 异常值处理
    data['avg_monthly_charges'] = data['TotalCharges'] / data['tenure']
    data['avg_monthly_charges'] = data['avg_monthly_charges'].replace([np.inf, -np.inf], np.nan)
    data['avg_monthly_charges'] = data['avg_monthly_charges'].fillna(data['MonthlyCharges'])
    print("✅ 派生特征 avg_monthly_charges 计算完成")

    # 5. 目标变量编码
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

    # 6. 二分类变量编码
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in data.columns:
            if col == 'gender':
                data[col] = data[col].map({'Female': 1, 'Male': 0})
            else:
                data[col] = data[col].map({'Yes': 1, 'No': 0})

    # 7. 多分类变量独热编码
    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    data = pd.get_dummies(data, columns=multi_cols, drop_first=True)

    # 8. 特征与标签划分
    drop_cols = ['customerID', 'Churn']
    X = data.drop(columns=drop_cols).copy()
    y = data['Churn'].copy()

    # 9. 数值特征标准化（强制类型转换+异常值兜底）
    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_charges']

    # 强制数值列转为数值类型并处理异常值
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())
        X[col] = X[col].replace([np.inf, -np.inf], X[col].median())

    # 执行标准化
    scaled_data = scaler.fit_transform(X[numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=X.index)
    X = pd.concat([X.drop(columns=numeric_cols), scaled_df], axis=1)

    # 10. 训练集/测试集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✅ 训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()


if __name__ == "__main__":
    # 运行数据处理流程
    X_train, X_test, y_train, y_test, scaler, features = load_and_process_data()
    print("\n🎉 数据处理全流程完成！")
    print(f"最终特征列表: {features}")