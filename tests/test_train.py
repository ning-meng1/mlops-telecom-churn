import pandas as pd

from src.predict import predict


def test_prediction():
    """模型预测单元测试：校验接口返回格式与基础可用性"""
    print("======开始模型预测功能测试======")

    # 加载数据集，选取测试集样本
    from src.data_process import get_full_dataset
    _, X_test, _, _, _, _ = get_full_dataset()
    sample = X_test.iloc[[5]]

    # 执行预测
    result = predict(sample)
    print("预测结果：")
    print(result)

    # 断言校验返回字段
    assert "prediction" in result, "返回结果缺少 prediction 字段"
    assert "churn_probability" in result, "返回结果缺少 churn_probability 字段"

    # 额外健壮性校验
    assert isinstance(result["prediction"], int), "prediction 必须为整型"
    assert 0.0 <= result["churn_probability"] <= 1.0, "流失概率必须在0~1区间内"

    print("✅ 模型预测测试全部通过！")


if __name__ == "__main__":
    test_prediction()