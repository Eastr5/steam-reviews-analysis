import os
import pandas as pd
import argparse
import logging

# 注意：确保您的项目结构支持从根目录运行
# 即 'src' 文件夹和 main.py 在同一级
from src.preprocessing.text_processor import load_stopwords
from src.features.feature_builder import extract_all_features
from src.models.experiment_runner import run_all_experiments
from src.utils.helpers import ensure_dir

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='训练Steam评论分类模型')
    parser.add_argument('--data', type=str, required=True, help='标记评论数据CSV文件的路径')
    parser.add_argument('--output', type=str, default='results', help='保存结果的目录')
    return parser.parse_args()

def setup_logging(output_dir):
    """配置日志系统。"""
    log_file_path = os.path.join(output_dir, 'run.log')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def main():
    """运行评论分类流程的主函数。"""
    args = parse_args()
    data_path = args.data
    output_dir = args.output
    
    ensure_dir(output_dir)
    setup_logging(output_dir)
    
    main_logger = logging.getLogger(__name__)
    
    if not os.path.exists(data_path):
        main_logger.error(f"错误: 数据文件 {data_path} 不存在。")
        return
    
    main_logger.info(f"从 {data_path} 加载数据...")
    data = pd.read_csv(data_path)
    main_logger.info(f"成功加载数据，数据形状为: {data.shape}")
    main_logger.info("数据预览:\n" + str(data.head()))
    
    stopwords = load_stopwords()
    main_logger.info(f"已加载 {len(stopwords)} 个停用词用于文本预处理。")
    
    main_logger.info("\n开始特征工程...")
    features_df, target_series = extract_all_features(data, stopwords)
    
    main_logger.info("\n开始运行所有实验...")
    results = run_all_experiments(features_df, target_series, output_dir)
    
    best_model_idx = results['AUC'].idxmax()
    best_model = results.loc[best_model_idx]
    
    main_logger.info("\n=== 总体最佳模型 ===")
    # 确保 'fpr' 和 'tpr' 列存在时才丢弃
    columns_to_drop = [col for col in ['fpr', 'tpr'] if col in best_model.index]
    best_model_str = best_model.drop(columns_to_drop).to_string()
    main_logger.info(best_model_str)
    
    main_logger.info(f"\n所有实验结果已保存在 {output_dir}/所有轮次比较.xlsx")
    main_logger.info("模型训练成功完成！")

if __name__ == '__main__':
    main()
