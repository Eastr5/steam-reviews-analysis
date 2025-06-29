# steam-reviews-analysis/src/analysis/rational_score_calculator.py
import re
import numpy as np
from src.utils.new_helpers import min_max_normalize, safe_log1p

class RationalScoreCalculator:
    """
    用于计算评论理性分数的类。
    """
    def __init__(self, specific_keywords=None, weights=None):
        """
        :param specific_keywords: 用于计算特定关键词个数的列表。
        :param weights: 理性分数各组成部分的权重字典，例如 {'comment_length': w1, ...}。
        """
        self.specific_keywords = specific_keywords if specific_keywords is not None else []
        # 默认权重，可以根据业务需求调整
        self.weights = weights if weights is not None else {
            'comment_length': 0.25,
            'average_word_length': 0.25,
            'number_count': 0.25,
            'specific_keyword_count': 0.25
        }

    def _calculate_comment_length(self, text):
        """计算评论长度 (字符数)。"""
        return len(str(text))

    def _calculate_average_word_length(self, text):
        """计算评论的平均词长。"""
        words = str(text).split()
        if not words:
            return 0.0
        return np.mean([len(word) for word in words])

    def _calculate_number_count(self, text):
        """计算评论中数字的个数。"""
        return len(re.findall(r'\d', str(text)))

    def _calculate_specific_keyword_count(self, text):
        """
        计算评论中特定关键词的个数。
        :param text: 输入文本。
        :return: 关键词出现的总次数。
        """
        text = str(text).lower()
        count = 0
        for keyword in self.specific_keywords:
            count += text.count(keyword.lower())
        return count

    def calculate_rational_score(self, df, comment_col='comment'):
        """
        计算DataFrame中每条评论的理性分数。
        :param df: 包含评论的DataFrame。
        :param comment_col: 评论列的名称。
        :return: 包含理性分数及其组成部分的DataFrame。
        """
        df['comment_length'] = df[comment_col].apply(self._calculate_comment_length)
        df['average_word_length'] = df[comment_col].apply(self._calculate_average_word_length)
        df['number_count'] = df[comment_col].apply(self._calculate_number_count)
        df['specific_keyword_count'] = df[comment_col].apply(self._calculate_specific_keyword_count)

        # 对用于理性分数的特征进行归一化处理
        df['log_comment_length_norm'] = min_max_normalize(safe_log1p(df['comment_length']))
        df['average_word_length_norm'] = min_max_normalize(df['average_word_length'])
        df['number_count_norm'] = min_max_normalize(df['number_count'])
        df['specific_keyword_count_norm'] = min_max_normalize(df['specific_keyword_count'])

        # 计算理性分数
        df['rational_score'] = (
            self.weights['comment_length'] * df['log_comment_length_norm'] +
            self.weights['average_word_length'] * df['average_word_length_norm'] +
            self.weights['number_count'] * df['number_count_norm'] +
            self.weights['specific_keyword_count'] * df['specific_keyword_count_norm']
        )
        return df

