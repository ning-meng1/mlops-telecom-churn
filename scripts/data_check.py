import pandas as pd
import json
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config.settings import (
    DATA_PATH,
    DATA_DIR,
    NUMERIC_COLS,
    TARGET_COL
)

def run_data_check(report_tag="v1",data_file_path=None) -> bool:
    """
    数据集质量门禁校验
    :param report_tag: 数据版本标签 v1/v2，用于报告命名
    :param data_file_path: 待校验csv路径，不传就默认DATA_PATH(v1)
    :return: True=校验通过，False=校验不通过
    """
    # 报告输出路径
    report_name = f"data_quality_report_{report_tag}.json"
    report_path = os.path.join(DATA_DIR, report_name)

    if data_file_path is None:
        data_file_path = DATA_PATH


    # 读取原始数据
    df = pd.read_csv(data_file_path)
    total_samples, total_cols = df.shape

    result = {
        "version_tag": report_tag,
        "file_path": data_file_path,
        "total_samples": total_samples,
        "total_cols": total_cols,
        "check_passed": True,
        "check_items": {}
    }

    # ========== 1. 缺失值检查 ==========
    missing_info = {}
    has_missing_alert = False
    missing_threshold = 0.15
    for col in df.columns:
        missing_cnt = df[col].isna().sum()
        missing_rate = missing_cnt / total_samples
        missing_info[col] = {
            "missing_count": int(missing_cnt),
            "missing_rate": round(missing_rate, 4)
        }
        if missing_rate > missing_threshold:
            has_missing_alert = True
            result["check_passed"] = False
    result["check_items"]["missing_check"] = {
        "threshold": f"{missing_threshold*100}%",
        "alert": has_missing_alert,
        "detail": missing_info
    }

    # ========== 2. 数值列负数异常检查 ==========
    negative_info = {}
    has_negative_alert = False
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        # 转数值，不能转的当成异常
        df[col] = pd.to_numeric(df[col], errors="coerce")
        neg_cnt = (df[col] < 0).sum()
        negative_info[col] = {"negative_count": int(neg_cnt)}
        if neg_cnt > 0:
            has_negative_alert = True
            result["check_passed"] = False
    result["check_items"]["numeric_negative_check"] = {
        "alert": has_negative_alert,
        "detail": negative_info
    }

    # ========== 3. 目标标签分布检查 ==========
    target_dist = df[TARGET_COL].value_counts(normalize=True).to_dict()
    target_dist = {k: round(v, 4) for k, v in target_dist.items()}
    imbalance_threshold = 0.85
    has_imbalance_alert = any(v > imbalance_threshold for v in target_dist.values())
    if has_imbalance_alert:
        result["check_passed"] = False
    result["check_items"]["target_dist_check"] = {
        "threshold": f"{imbalance_threshold*100}%",
        "alert": has_imbalance_alert,
        "dist": target_dist
    }

    # ========== 控制台打印日志 ==========
    print(f"===== 数据校验开始 | 版本：{report_tag} =====")
    print(f"样本数：{total_samples}, 特征数：{total_cols}")
    print(f"缺失值告警：{has_missing_alert}")
    print(f"数值负数告警：{has_negative_alert}")
    print(f"标签分布告警：{has_imbalance_alert}")
    print(f"✅ 整体校验结果：{result['check_passed']}")

    # 保存json报告
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"报告已保存至：{report_path}")

    return result["check_passed"]


if __name__ == "__main__":
    # 校验v2：指定清洗后的文件路径
    clean_data_path = os.path.join(DATA_DIR, "telco_churn_clean.csv")
    ok = run_data_check(report_tag="v2", data_file_path=clean_data_path)
    # 校验失败直接退出，阻断后续训练
    if not ok:
        raise SystemExit("数据质量校验不通过，停止执行！")


"""
1. 导入需要的包：pandas、json，还有 settings 里的路径、特征配置
2. 读取 `DATA_PATH` 也就是 `telco_churn.csv`
3. 准备一个字典，用来存所有检查结果，后面直接保存成 json 报告
4. 依次执行 4 项检查：
- 检查 1：基础信息：样本总数、列数量
- 检查 2：缺失值遍历，按我们约定的阈值：缺失率 > 15% 告警
- 检查 3：只遍历 `NUMERIC_COLS` 里面这些数值列，看有没有负数
- 检查 4：统计 `TARGET_COL（Churn）` 的标签分布，某一类占比 > 85% 告警
5. 每发现一个问题，就标记整体状态为：**校验不通过**
6. 控制台打印所有检查信息，有警告就标出来
7. 把结果字典写入 json 报告，放到 data 文件夹
8. 最后返回 True / False（通过 / 不通过）
"""

