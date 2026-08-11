from config.settings import (
    REQUIRED_FEATURES,
    FEATURE_RANGE_RULES
)



def validate_features(features):
    """
    API输入参数统一校验

    返回:
        None:
            校验通过

        str:
            错误信息
    """


    # =====================
    # 1. 基础类型校验
    # =====================

    if not isinstance(features, dict):

        return "features必须是JSON对象"



    # =====================
    # 2. 字段完整性校验
    # =====================

    missing = [

        field

        for field in REQUIRED_FEATURES

        if field not in features

    ]


    if missing:

        return f"缺少字段:{missing}"



    # =====================
    # 3. 数值类型校验
    # =====================

    for field in FEATURE_RANGE_RULES:


        value = features[field]


        if not isinstance(
            value,
            (int,float)
        ):

            return (
                f"{field}类型错误，"
                "必须为数字"
            )



    # =====================
    # 4. 数值范围校验
    # =====================

    for field,(min_value,max_value) in FEATURE_RANGE_RULES.items():


        value = features[field]


        if value < min_value or value > max_value:


            return (

                f"{field}数值异常，"

                f"允许范围:{min_value}-{max_value}"

            )


    return None