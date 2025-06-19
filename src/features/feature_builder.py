import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.preprocessing import StandardScaler
import jieba
import logging
from ..utils.helpers import get_sentiment # FIX: Correct relative import path

# 获取logger实例
logger = logging.getLogger(__name__)

def clean_text(text, stopwords):
    """文本清洗和分词"""
    if not isinstance(text, str):
        return ""
    # 移除特殊字符和数字
    text = ''.join(filter(lambda x: '\u4e00' <= x <= '\u9fa5', text))
    # 分词
    words = jieba.cut(text)
    # 移除停用词
    return ' '.join(word for word in words if word not in stopwords and len(word.strip()) > 0)

def extract_all_features(data, stopwords):
    """
    从原始数据中提取所有特征，并返回特征DataFrame和目标Series。

    :param data: 原始DataFrame。
    :param stopwords: 停用词列表。
    :return: (features_df, target_series)
    """
    df = data.copy()
    
    # 1. 预处理目标变量
    df['target'] = df['推荐'].apply(lambda x: 1 if x == 'Recommended' else 0)
    
    # 2. 文本预处理
    logger.info("开始文本预处理...")
    df['clean_review'] = df['评论'].apply(lambda text: clean_text(text, stopwords))
    logger.info("文本预处理完成。")

    # 3. 提取文本特征 (TF-IDF, CountVectorizer, LDA)
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_features = tfidf.fit_transform(df['clean_review'])
    
    cv = CountVectorizer(max_features=1000)
    cv_features = cv.fit_transform(df['clean_review'])
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_features = lda.fit_transform(cv_features)

    # 4. 提取数值特征
    logger.info("提取并标准化数值特征...")
    df['sentence_count'] = df['评论'].apply(lambda x: len(str(x).split('。')))
    df['avg_sentence_len'] = df.apply(lambda row: row['评论长度'] / row['sentence_count'] if row['sentence_count'] > 0 else 0, axis=1)
    
    numeric_features_df = df[['评论长度', '情感得分', 'sentence_count', 'avg_sentence_len']].fillna(0)
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features_df)
    
    # 5. 合并所有特征
    lda_df = pd.DataFrame(lda_features, columns=[f'lda_{i}' for i in range(lda_features.shape[1])])
    numeric_df = pd.DataFrame(numeric_features_scaled, columns=['comment_len_scaled', 'sentiment_score_scaled', 'sentence_count_scaled', 'avg_sentence_len_scaled'])
    
    logger.info("对TF-IDF和CV特征进行PCA降维...")
    pca = PCA(n_components=5)
    tfidf_pca = pca.fit_transform(tfidf_features.toarray())
    cv_pca = pca.fit_transform(cv_features.toarray())
    tfidf_pca_df = pd.DataFrame(tfidf_pca, columns=[f'tfidf_pca_{i}' for i in range(5)])
    cv_pca_df = pd.DataFrame(cv_pca, columns=[f'cv_pca_{i}' for i in range(5)])

    # 最终的特征DataFrame
    features_df = pd.concat([lda_df, tfidf_pca_df, cv_pca_df, numeric_df], axis=1)
    
    # 6. 分离并返回特征和目标
    target_series = df['target']
    
    logger.info(f"特征工程完成，最终特征数量: {features_df.shape[1]}")
    
    return features_df, target_series
