# steam-reviews-analysis/src/utils/new_helpers.py
import pandas as pd
import numpy as np

def min_max_normalize(series):
    """
    对Pandas Series进行Min-Max归一化。
    :param series: 输入的Pandas Series。
    :return: 归一化后的Series。
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val: # 避免除以零
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)

def safe_log1p(series):
    """
    对Pandas Series进行log1p转换，处理可能存在的负值或零。
    :param series: 输入的Pandas Series。
    :return: 转换后的Series。
    """
    return np.log1p(series.apply(lambda x: max(0, x))) # 确保输入非负

