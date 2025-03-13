"""
模型训练和评估的实验执行。
此模块处理运行评论分类的机器学习实验。
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, 
    accuracy_score, classification_report
)
import joblib
from typing import Dict, List, Tuple, Any, Optional

from src.features.feature_builder import build_features
from src.utils.helpers import ensure_dir


def run_experiment(
    round_name: str, 
    data: pd.DataFrame, 
    round_config: Dict[str, Any], 
    save_dir: str
) -> pd.DataFrame:
    """
    使用给定配置运行机器学习实验。
    
    参数:
        round_name: 实验轮次名称
        data: 包含预处理数据的DataFrame
        round_config: 实验的配置字典
        save_dir: 保存结果的目录
        
    返回:
        包含实验结果的DataFrame
    """
    # 确保保存目录存在
    ensure_dir(save_dir)
    
    target_column = '推荐'
    content_column = '评论'
    
    print(f"\\n### 开始实验: {round_name} ###")
    
    # 映射目标变量
    data['y_target'] = data[target_column].map({'Recommended': 1, 'Not Recommended': 0})
    
    # 分割数据
    X = data['processed_text']
    y = data['y_target']
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集样本数: {X_train_text.shape[0]}, 测试集样本数: {X_test_text.shape[0]}")
    
    # 构建特征
    features_dict = build_features(X_train_text, X_test_text, data, round_config)
    X_train_features = features_dict['X_train']
    X_test_features = features_dict['X_test']
    print("特征构建完成。")
    
    # 定义特征描述
    text_features = ', '.join(round_config['text_features'])
    numeric_features = ', '.join(round_config['numeric_features']) if round_config.get('numeric_features') else '否'
    pca = '是' if round_config.get('use_pca', False) else '否'
    rfe = '是' if round_config.get('use_rfe', False) else '否'
    features_used = f"文本特征: {text_features}; 数值特征: {numeric_features}; PCA: {pca}; RFE: {rfe}"
    print(f"使用的特征: {features_used}")
    
    # 根据配置初始化模型
    models = _initialize_models(round_config, y_train)
    print(f"使用的模型: {', '.join(models.keys())}")
    
    # 训练和评估模型
    comparison = []
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid')
    
    for model_name, model in models.items():
        print(f"\\n训练模型: {model_name}")
        
        # 训练模型
        model.fit(X_train_features, y_train)
        joblib.dump(model, f'{save_dir}/{model_name}_model.pkl')
        print(f"模型 {model_name} 训练完成。")
        
        # 评估模型
        evaluation = _evaluate_model(model, X_test_features, y_test)
        evaluation.update({
            '轮次': round_name,
            '模型': model_name,
            '特征': features_used
        })
        comparison.append(evaluation)
        
        # 绘制ROC曲线
        plt.plot(evaluation['fpr'], evaluation['tpr'], 
                 label=f"{model_name} (AUC = {evaluation['AUC']:.4f})")
    
    # 完成ROC曲线
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title(f'{round_name} 各模型的 ROC 曲线')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # 保存结果
    roc_curve_path = f'{save_dir}/roc_curves.png'
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"ROC 曲线已保存到 {roc_curve_path}")
    
    # 创建并保存比较DataFrame
    comparison_df = pd.DataFrame(comparison)
    comparison_csv_path = f'{save_dir}/model_comparison.csv'
    comparison_df.sort_values(by='AUC', ascending=False).to_csv(comparison_csv_path, index=False)
    print(f"模型比较表格已保存到 {comparison_csv_path}")
    
    # 返回排序后的比较DataFrame
    return comparison_df


def _initialize_models(config: Dict[str, Any], y_train: pd.Series) -> Dict[str, Any]:
    """
    根据配置初始化机器学习模型。
    
    参数:
        config: 配置字典
        y_train: 用于类权重的训练目标值
        
    返回:
        模型实例字典
    """
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    # 定义所有可用模型
    all_models = {
        'XGBoost': xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=42,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        ),
        'SVM': SVC(
            random_state=42, 
            probability=True, 
            class_weight='balanced'
        ),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(
            random_state=42, 
            class_weight='balanced'
        )
    }
    
    # 根据配置筛选模型
    if 'models' in config:
        return {k: v for k, v in all_models.items() if k in config['models']}
    else:
        return all_models


def _evaluate_model(model: Any, X_test: Any, y_test: pd.Series) -> Dict[str, Any]:
    """
    在测试数据上评估训练好的模型。
    
    参数:
        model: 训练好的模型实例
        X_test: 测试特征矩阵
        y_test: 测试目标值
        
    返回:
        包含评估指标的字典
    """
    # 获取预测
    y_pred = model.predict(X_test)
    
    # 获取概率或决策函数值用于ROC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    else:
        y_proba = y_pred
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    
    # 计算混淆矩阵和派生指标
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        false_positive_rate = 0
        false_negative_rate = 0
    
    # 计算ROC指标
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    # 返回所有指标
    return {
        '准确率': round(accuracy, 4),
        '召回率': round(recall, 4),
        'F1分数': round(f1, 4),
        '假正率': round(false_positive_rate, 4),
        '假负率': round(false_negative_rate, 4),
        'AUC': round(roc_auc, 4),
        'fpr': fpr,
        'tpr': tpr
    }


def get_experiment_configs() -> Dict[str, Dict[str, Any]]:
    """
    获取预定义的实验配置。
    
    返回:
        实验配置字典
    """
    # 轮次一：文本特征: LDA, TF-IDF, CV; PCA; RFE; 所有数值特征
    round1_config = {
        'text_features': ['LDA', 'TF-IDF', 'CV'],
        'numeric_features': ['评论长度', '情感得分', '句子数', '平均句子长度'],
        'use_pca': True,
        'use_rfe': True,
        'models': ['XGBoost', 'RandomForest', 'SVM', 'KNN', 'DecisionTree']
    }
    
    # 轮次二：文本特征: LDA, TF-IDF, CV; 无PCA; 无RFE; 所有数值特征
    round2_config = {
        'text_features': ['LDA', 'TF-IDF', 'CV'],
        'numeric_features': ['评论长度', '情感得分', '句子数', '平均句子长度'],
        'use_pca': False,
        'use_rfe': False,
        'models': ['XGBoost', 'RandomForest', 'SVM', 'KNN', 'DecisionTree']
    }
    
    # 轮次三：文本特征: LDA, TF-IDF; 无PCA; 无RFE; 所有数值特征
    round3_config = {
        'text_features': ['LDA', 'TF-IDF'],
        'numeric_features': ['评论长度', '情感得分', '句子数', '平均句子长度'],
        'use_pca': False,
        'use_rfe': False,
        'models': ['XGBoost', 'RandomForest', 'SVM', 'KNN', 'DecisionTree']
    }
    
    # 轮次四：文本特征: LDA, CV; 无PCA; 无RFE; 所有数值特征
    round4_config = {
        'text_features': ['LDA', 'CV'],
        'numeric_features': ['评论长度', '情感得分', '句子数', '平均句子长度'],
        'use_pca': False,
        'use_rfe': False,
        'models': ['XGBoost', 'RandomForest', 'SVM', 'KNN', 'DecisionTree']
    }
    
    # 轮次五：文本特征: TF-IDF; 无PCA; 无RFE; 所有数值特征
    round5_config = {
        'text_features': ['TF-IDF'],
        'numeric_features': ['评论长度', '情感得分', '句子数', '平均句子长度'],
        'use_pca': False,
        'use_rfe': False,
        'models': ['XGBoost', 'RandomForest', 'SVM', 'KNN', 'DecisionTree']
    }
    
    # 轮次六：文本特征: TF-IDF, LDA; 无PCA; 无RFE; 无情感，但有其他数值特征
    round6_config = {
        'text_features': ['TF-IDF', 'LDA'],
        'numeric_features': ['评论长度', '句子数', '平均句子长度'],  # 排除情感得分
        'use_pca': False,
        'use_rfe': False,
        'models': ['SVM', 'XGBoost', 'RandomForest', 'KNN', 'DecisionTree']
    }
    
    # 轮次七：文本特征: TF-IDF, LDA; 无PCA; 无RFE; 只有情感
    round7_config = {
        'text_features': ['TF-IDF', 'LDA'],
        'numeric_features': ['情感得分'],  # 只包含情感得分
        'use_pca': False,
        'use_rfe': False,
        'models': ['SVM', 'XGBoost', 'RandomForest', 'KNN', 'DecisionTree']
    }
    
    return {
        '轮次一': round1_config,
        '轮次二': round2_config,
        '轮次三': round3_config,
        '轮次四': round4_config,
        '轮次五': round5_config,
        '轮次六': round6_config,
        '轮次七': round7_config,
    }


def run_all_experiments(data: pd.DataFrame, base_dir: str = 'results') -> pd.DataFrame:
    """
    在给定数据上运行所有预定义实验。
    
    参数:
        data: 包含预处理数据的DataFrame
        base_dir: 保存结果的基础目录
        
    返回:
        包含所有实验结果的合并DataFrame
    """
    # 获取实验配置
    configs = get_experiment_configs()
    all_comparisons = []
    
    # 运行每个实验
    for round_name, config in configs.items():
        save_dir = f"{base_dir}/{round_name.replace(' ', '_')}"
        comparison_df = run_experiment(round_name, data, config, save_dir)
        all_comparisons.append(comparison_df)
    
    # 合并所有结果
    combined_results = pd.concat(all_comparisons).reset_index(drop=True)
    
    # 保存合并结果
    excel_path = f'{base_dir}/所有轮次比较.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for idx, round_name in enumerate(configs.keys()):
            comparison_df = all_comparisons[idx]
            comparison_df_sorted = comparison_df.sort_values(by='AUC', ascending=False).reset_index(drop=True)
            # 使Excel表名安全
            safe_sheet_name = round_name[:31].replace(':', '').replace('/', '').replace('\\', '').replace('*', '').replace('?', '').replace('[', '').replace(']', '')
            comparison_df_sorted.to_excel(writer, sheet_name=safe_sheet_name, index=False)
    
    print(f"\\n所有轮次的比较结果已保存到 {excel_path}")
    return combined_results