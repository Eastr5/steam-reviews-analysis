"""
用于评论分析的文本预处理工具。
该模块提供文本预处理功能，包括停用词移除和分词。
"""
import jieba
import re
from typing import Set, List, Dict, Any


def load_stopwords() -> Set[str]:
    """
    加载并返回中文停用词集合。
    
    返回:
        用于文本处理中移除的停用词集合
    """
    stopwords = set([
        '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
        '或', '一个', '没有', '我们', '你们', '他们', '很', '好',
        '这个', '那个', '这些', '那些', '啊', '呢', '吧', '啦'
    ])
    return stopwords


def preprocess_text(text: str, stopwords: Set[str]) -> str:
    """
    通过分词和去除停用词对文本进行预处理。
    
    参数:
        text: 需要预处理的输入文本
        stopwords: 需要移除的停用词集合
        
    返回:
        经过预处理的文本，单词之间用空格分隔
    """
    # 分词并去除停用词
    words = jieba.cut(str(text))
    words = [w for w in words if w not in stopwords and len(w.strip()) > 1]
    return ' '.join(words)


def count_sentences(text: str) -> int:
    """
    基于标点符号统计文本中的句子数量。
    
    参数:
        text: 输入文本
        
    返回:
        检测到的句子数量
    """
    return len(re.findall(r'[。！？.!?]', str(text)))


def calculate_text_features(text: str) -> Dict[str, float]:
    """
    从给定文本中计算各种文本特征。
    
    参数:
        text: 输入文本
        
    返回:
        包含文本特征的字典，包括长度和句子数量
    """
    text_length = len(str(text))
    sentence_count = count_sentences(text)
    avg_sentence_length = text_length / (sentence_count + 1)  # +1避免除零错误
    
    return {
        "text_length": text_length,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length
    }