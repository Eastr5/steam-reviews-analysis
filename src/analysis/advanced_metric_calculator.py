# steam-reviews-analysis/src/analysis/advanced_metric_calculator.py
import pandas as pd
from datetime import timedelta
from src.utils.new_helpers import min_max_normalize

class AdvancedMetricCalculator:
    """
    用于计算高级用户评论指标（CPIV, VPV, 负面关键词触发指数）的类。
    """
    def __init__(self, negative_keywords=None):
        self.negative_keywords = negative_keywords if negative_keywords is not None else []

    def calculate_normalized_metrics(self, df, playtime_col='playtime', helpful_col='helpful',
                                     rational_score_col='rational_score', emotional_score_col='emotional_score'):
        """
        计算归一化后的指标。
        :param df: 包含原始指标的DataFrame。
        :param playtime_col: 游戏时长列名。
        :param helpful_col: 有帮助点赞数列名。
        :param rational_score_col: 理性分数列名。
        :param emotional_score_col: 感性分数列名。
        :return: 包含归一化指标的DataFrame。
        """
        # 确保列存在，如果不存在则创建默认值
        if playtime_col not in df.columns:
            df[playtime_col] = 0
        if helpful_col not in df.columns:
            df[helpful_col] = 0
        if rational_score_col not in df.columns:
            df[rational_score_col] = 0.5 # 默认中性理性分数
        if emotional_score_col not in df.columns:
            df[emotional_score_col] = 0.0 # 默认中性情感分数

        df['playtime_norm'] = min_max_normalize(df[playtime_col])
        df['rational_norm'] = min_max_normalize(df[rational_score_col])
        df['emotional_norm'] = min_max_normalize(df[emotional_score_col])
        df['helpful_norm'] = min_max_normalize(df[helpful_col])
        return df

    def calculate_cpiv_score(self, df):
        """
        计算核心玩家深度反馈价值 (CPIV) 分数。
        CPIV Score = (0.4 * playtime_norm) + (0.4 * rational_norm) + (0.2 * helpful_norm) + (0.0 * emotional_norm)
        :param df: 包含归一化指标的DataFrame。
        :return: 包含CPIV分数的DataFrame。
        """
        df['CPIV_Score'] = (
            0.4 * df['playtime_norm'] +
            0.4 * df['rational_norm'] +
            0.2 * df['helpful_norm'] +
            0.0 * df['emotional_norm']
        )
        return df

    def calculate_vpv_score(self, df):
        """
        计算病毒式传播潜力价值 (VPV) 分数。
        VPV Score = (0.5 * emotional_norm) + (0.4 * helpful_norm) + (0.1 * playtime_norm) + (0.0 * rational_norm)
        :param df: 包含归一化指标的DataFrame。
        :return: 包含VPV分数的DataFrame。
        """
        df['VPV_Score'] = (
            0.5 * df['emotional_norm'] +
            0.4 * df['helpful_norm'] +
            0.1 * df['playtime_norm'] +
            0.0 * df['rational_norm']
        )
        return df

    def calculate_negative_keyword_trigger_index(self, df, comment_col='comment', timestamp_col='timestamp', days=7):
        """
        计算负面关键词触发指数。
        :param df: 包含评论和时间戳的DataFrame。
        :param comment_col: 评论列名。
        :param timestamp_col: 时间戳列名。
        :param days: 计算近期评论的天数。
        :return: 负面关键词触发指数。
        """
        if timestamp_col not in df.columns or df[timestamp_col].empty:
            return 0.0 # 如果没有时间戳或为空，则无法计算

        end_date = df[timestamp_col].max()
        start_date = end_date - timedelta(days=days)
        recent_comments_df = df[(df[timestamp_col] >= start_date) & (df[timestamp_col] <= end_date)].copy()

        if recent_comments_df.empty:
            return 0.0

        recent_comments_df['has_negative_keyword'] = recent_comments_df[comment_col].apply(
            lambda x: any(kw in str(x).lower() for kw in self.negative_keywords)
        )
        negative_comment_count = recent_comments_df['has_negative_keyword'].sum()
        total_recent_comments = len(recent_comments_df)

        return (negative_comment_count / total_recent_comments) if total_recent_comments > 0 else 0.0

