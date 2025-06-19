import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# 导入 FontProperties 用于处理字体
from matplotlib.font_manager import FontProperties

# --- 字体设置开始 ---

def find_chinese_font():
    """
    在系统中查找可用的中文字体文件。
    会按顺序查找常见的几种中文字体。
    """
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/msyh.ttc',
        '/System/Library/Fonts/STHeiti.ttc'
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            logging.info(f"找到中文字体: {path}")
            return FontProperties(fname=path, size=12)
    
    logging.warning("未找到指定的中文字体，图表中的中文可能无法正常显示。")
    return None

CHINESE_FONT = find_chinese_font()

# --- 字体设置结束 ---


logger = logging.getLogger(__name__)

def plot_roc_curves(results, round_name, output_dir_round):
    """为单次实验中的所有模型绘制ROC曲线。"""
    plt.figure(figsize=(12, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    for index, row in results.iterrows():
        plt.plot(row['fpr'], row['tpr'], label=f"{row['Model']} (AUC = {row['AUC']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    
    plt.title(f'轮次 {round_name} 各模型的ROC曲线', fontproperties=CHINESE_FONT)
    plt.xlabel('假正率 (False Positive Rate)', fontproperties=CHINESE_FONT)
    plt.ylabel('真正率 (True Positive Rate)', fontproperties=CHINESE_FONT)
    
    legend = plt.legend(loc='lower right')
    for text in legend.get_texts():
        text.set_fontproperties(CHINESE_FONT)
    
    plt.grid(True)
    plt.tight_layout()
    
    roc_curve_path = os.path.join(output_dir_round, 'roc_curves.png')
    plt.savefig(roc_curve_path)
    plt.close()
    logger.info(f"ROC 曲线已保存到 {roc_curve_path}")

def run_single_experiment(data, features_config, models_to_run, output_dir_round, round_name):
    """运行单次实验，包括特征选择、模型训练、评估和结果保存。"""
    logger.info(f"\n### 开始实验: {round_name} ###")
    
    # 确保特征列表存在于数据中
    valid_features = [feat for feat in features_config['features'] if feat in data.columns]
    X = data[valid_features]
    y = data['target']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_index, test_index = next(iter(skf.split(X, y)))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    logger.info(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

    features_desc = (f"文本特征: {', '.join(features_config.get('text_features', [])) if features_config.get('text_features') else '无'}; "
                     f"数值特征: {', '.join(features_config.get('numeric_features', [])) if features_config.get('numeric_features') else '无'}; "
                     f"PCA: {'是' if features_config.get('use_pca') else '否'}")

    logger.info(f"使用的特征: {features_desc}")
    logger.info(f"使用的模型: {', '.join(models_to_run.keys())}")
    
    results = []
    
    for model_name, model in models_to_run.items():
        logger.info(f"\n训练模型: {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        results.append({
            'Round': round_name,
            'Model': model_name,
            'Accuracy': accuracy,
            'AUC': auc,
            'fpr': fpr,
            'tpr': tpr
        })
        
        model_path = os.path.join(output_dir_round, f'{model_name}.pkl')
        joblib.dump(model, model_path)
        logger.info(f"模型 {model_name} 训练完成并保存到 {model_path}。")
        
    results_df = pd.DataFrame(results)
    plot_roc_curves(results_df, round_name, output_dir_round)
    
    comparison_table_path = os.path.join(output_dir_round, 'model_comparison.csv')
    results_df.drop(columns=['fpr', 'tpr']).to_csv(comparison_table_path, index=False)
    logger.info(f"模型比较表格已保存到 {comparison_table_path}")
    
    return results_df

def run_all_experiments(processed_data, target, output_dir):
    """
    运行所有预定义的实验轮次。
    :param processed_data: 只包含特征的DataFrame。
    :param target: 包含目标标签的Series。
    :param output_dir: 保存结果的目录。
    """
    data_with_target = processed_data.copy()
    data_with_target['target'] = target.values

    all_feature_columns = processed_data.columns.tolist()
    
    experiment_configs = {
        '轮次一': {'features': all_feature_columns, 'text_features': ['LDA', 'TF-IDF_PCA', 'CV_PCA'], 'numeric_features': ['评论长度', '情感得分', '句子数', '平均句子长度'], 'use_pca': True},
        '轮次二': {'features': [col for col in all_feature_columns if 'pca' not in col], 'text_features': ['LDA'], 'numeric_features': ['评论长度', '情感得分', '句子数', '平均句子长度'], 'use_pca': False},
        '轮次三': {'features': [col for col in all_feature_columns if 'cv_pca' not in col], 'text_features': ['LDA', 'TF-IDF_PCA'], 'numeric_features': ['评论长度', '情感得分', '句子数', '平均句子长度'], 'use_pca': True},
    }
    
    models_to_run = {
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(random_state=42)
    }
    
    all_results = []
    
    for round_name, config in experiment_configs.items():
        output_dir_round = os.path.join(output_dir, round_name)
        os.makedirs(output_dir_round, exist_ok=True)
        round_results = run_single_experiment(data_with_target, config, models_to_run, output_dir_round, round_name)
        all_results.append(round_results)
        
    final_results_df = pd.concat(all_results, ignore_index=True)
    final_comparison_path = os.path.join(output_dir, '所有轮次比较.xlsx')
    final_results_df.drop(columns=['fpr', 'tpr']).to_excel(final_comparison_path, index=False)
    
    return final_results_df
