# test_model.py
import joblib
from config.settings import MODEL_SAVE_PATH
import numpy as np

def test_model_load_and_predict():
    # 1. 加载模型
    try:
        model = joblib.load(MODEL_SAVE_PATH)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 构造一个和训练数据维度一致的随机样本（糖尿病数据集是10个特征）
    sample = np.random.rand(1, 10)  # shape: (1, 10)
    print(f"构造测试样本，维度: {sample.shape}")

    # 3. 执行预测
    try:
        prediction = model.predict(sample)
        print(f"✅ 预测成功，结果: {prediction[0]:.2f}")
    except Exception as e:
        print(f"❌ 预测失败: {e}")



if __name__ == "__main__":
    test_model_load_and_predict()