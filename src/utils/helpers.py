"""
Steam评论分析项目的辅助函数。
"""
import re
from snownlp import SnowNLP
from typing import Dict, List, Any, Union, Optional


def get_sentiment(text: str) -> float:
    """
    使用SnowNLP获取文本的情感得分。
    
    参数:
        text: 要分析的输入文本
        
    返回:
        介于0（消极）和1（积极）之间的情感得分。
        如果分析失败，则返回0.5。
    """
    try:
        return SnowNLP(str(text)).sentiments
    except Exception as e:
        return 0.5


def safe_hstack(features: List[Any]) -> Any:
    """
    安全地水平堆叠特征，处理None值和不同类型。
    
    参数:
        features: 特征矩阵列表，可能包含None值
        
    返回:
        堆叠后的特征矩阵，如果没有有效特征则返回None
    """
    from scipy.sparse import hstack, csr_matrix
    import numpy as np
    
    # 过滤掉None特征
    valid_features = []
    for f in features:
        if f is not None and not isinstance(f, np.ndarray):
            # 保持稀疏矩阵不变
            valid_features.append(f)
        elif f is not None and isinstance(f, np.ndarray):
            # 将密集数组转换为稀疏格式
            if f.ndim == 1:
                f = f.reshape(-1, 1)
            valid_features.append(csr_matrix(f))
            
    # 堆叠特征
    if len(valid_features) > 1:
        return hstack(valid_features)
    elif len(valid_features) == 1:
        return valid_features[0]
    else:
        return None


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建。
    
    参数:
        directory: 目录路径
    """
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)


def normalize_range(values: List[float], new_min: float = 0, new_max: float = 1) -> List[float]:
    """
    将值列表归一化到新的范围。
    
    参数:
        values: 要归一化的数值列表
        new_min: 新范围的最小值
        new_max: 新范围的最大值
        
    返回:
        归一化后的值列表
    """
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return [new_min for _ in values]
        
    return [(x - min_val) / (max_val - min_val) * (new_max - new_min) + new_min for x in values]