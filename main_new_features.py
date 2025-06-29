# steam-reviews-analysis/main_new_features.py
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import numpy as np

# 导入新的分析模块
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.rational_score_calculator import RationalScoreCalculator
from src.analysis.advanced_metric_calculator import AdvancedMetricCalculator
from src.utils.new_helpers import min_max_normalize # 确保导入了归一化函数

# 辅助函数：配置日志
def setup_logging(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

# 辅助函数：加载数据
def load_data(file_path):
    if not os.path.exists(file_path):
        logging.error(f"Error: Data file not found at {file_path}")
        return None
    try:
        # 尝试使用不同的编码加载，以防乱码
        df = pd.read_csv(file_path, encoding='utf-8')
        logging.info(f"Data loaded successfully from {file_path} with utf-8 encoding.")
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
            logging.info(f"Data loaded successfully from {file_path} with gbk encoding.")
            return df
        except Exception as e:
            logging.error(f"Error loading data from {file_path} with gbk encoding: {e}")
            return None
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

# 辅助函数：保存结果到CSV
def save_results_csv(df, output_path, file_name):
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, file_name)
    try:
        df.to_csv(full_path, index=False, encoding='utf-8')
        logging.info(f"Results saved successfully to {full_path}")
    except Exception as e:
        logging.error(f"Error saving results to {full_path}: {e}")

def generate_html_report(df_metrics, negative_trigger_index, output_dir):
    """
    生成包含分析结果和商业建议的HTML报告。
    :param df_metrics: 包含所有计算指标的DataFrame。
    :param negative_trigger_index: 负面关键词触发指数。
    :param output_dir: 输出目录。
    """
    os.makedirs(output_dir, exist_ok=True)
    html_file_path = os.path.join(output_dir, 'analysis_dashboard.html')

    # 获取高CPIV和高VPV的评论示例
    # 使用原始的中文列名 '评论', '游戏时长'
    # 确保 'helpful' 列在 df_metrics 中存在
    top_cpiv_comments = df_metrics.nlargest(5, 'CPIV_Score')[['评论', 'CPIV_Score', '游戏时长', 'rational_score', 'helpful']].to_html(index=False)
    top_vpv_comments = df_metrics.nlargest(5, 'VPV_Score')[['评论', 'VPV_Score', 'emotional_score', 'helpful', '游戏时长']].to_html(index=False)
    
    # 获取低感性高理性评论示例 (事实陈述区)
    # 假设低感性为 emotional_norm < 0.3, 高理性为 rational_norm > 0.7
    fact_comments = df_metrics[(df_metrics['emotional_norm'] < 0.3) & (df_metrics['rational_norm'] > 0.7)].nlargest(5, 'rational_norm')[['评论', 'emotional_norm', 'rational_norm']].to_html(index=False)
    
    # 获取高感性低理性评论示例 (情感宣泄区)
    # 假设高感性为 emotional_norm > 0.7, 低理性为 rational_norm < 0.3
    emotion_comments = df_metrics[(df_metrics['emotional_norm'] > 0.7) & (df_metrics['rational_norm'] < 0.3)].nlargest(5, 'emotional_norm')[['评论', 'emotional_norm', 'rational_norm']].to_html(index=False)


    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Steam 评论分析报告</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #f3f4f6;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 2rem auto;
                padding: 2rem;
                background-color: #ffffff;
                border-radius: 1rem;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            }}
            h1, h2, h3 {{
                color: #1a202c;
                font-weight: 700;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 1rem;
            }}
            th, td {{
                border: 1px solid #e2e8f0;
                padding: 0.75rem;
                text-align: left;
            }}
            th {{
                background-color: #edf2f7;
                font-weight: 600;
            }}
            .section-card {{
                background-color: #fdfdfd;
                border-radius: 0.75rem;
                padding: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }}
            .alert-box {{
                background-color: #fee2e2;
                color: #dc2626;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1.5rem;
                border: 1px solid #ef4444;
            }}
            .business-insight {{
                background-color: #e0f2fe;
                color: #0c4a6e;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-top: 1.5rem;
                border: 1px solid #38b2ac;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-4xl text-center mb-8">Steam 评论深度分析报告</h1>

            <div class="section-card">
                <h2 class="text-2xl mb-4">核心指标概览</h2>
                <p class="text-lg mb-2"><strong>负面关键词触发指数 (近7天):</strong> <span class="text-red-600 text-xl">{negative_trigger_index:.4f}</span></p>
                {"<div class='alert-box'><p><strong>警报！</strong> 负面关键词触发指数较高，可能存在潜在问题，请立即关注相关评论和游戏情况。</p></div>" if negative_trigger_index > 0.05 else ""}
                <p class="business-insight">
                    <strong>商业建议：</strong> 负面关键词触发指数是一个敏感指标，能快速捕捉用户对特定问题的关注度。当指数达到预设阈值时，应立即启动内部预警机制，并下钻分析具体是哪些关键词的提及率暴增，从而快速定位问题根源（如新版本Bug、服务器问题等）。
                </p>
            </div>

            <div class="section-card">
                <h2 class="text-2xl mb-4">用户画像矩阵新思路：感性分数 VS 理性分数</h2>
                <p class="mb-4">我们引入了**感性分数 (Emotional Score)** 和 **理性分数 (Rational Score)** 来更全面地理解用户评论的内在价值。</p>
                
                <h3 class="text-xl mb-3">感性分数 (Emotional Score)</h3>
                <p class="mb-2">基于 VADER 情感分析库，生成一个 -1 (极度负面) 到 +1 (极度正面) 的复合分数。</p>
                <p class="business-insight">
                    <strong>商业建议：</strong> 感性分数能有效捕捉用户的情绪倾向，特别适用于处理口语化和包含网络流行语的评论。高感性评论是社区氛围和用户情绪的直接反映，有助于运营团队快速识别用户情绪热点，进行舆情监控和危机管理。
                </p>

                <h3 class="text-xl mb-3 mt-6">理性分数 (Rational Score)</h3>
                <p class="mb-2">结合评论长度、平均词长、数字个数和特定关键词个数等维度，量化评论的信息密度和深度。</p>
                <p class="business-insight">
                    <strong>商业建议：</strong> 理性分数反映了评论的“含金量”。高理性评论通常包含有价值的反馈、技术细节或具体问题描述。产品和开发团队应重点关注高理性评论，从中获取产品改进的直接依据。
                </p>
                
                <h3 class="text-xl mb-3 mt-6">“情感宣泄区”社区 (高感性, 低理性) 评论示例</h3>
                <p class="mb-2">这类用户是情绪的放大器，喜欢玩梗、跟风，评价通常是“太好玩了！”或“垃圾游戏！”。</p>
                {emotion_comments}
                <p class="business-insight">
                    <strong>营销策略:</strong> 针对这类社区，应多发起“晒出你的高光时刻”、“最佳吐槽”等话题活动，用奖励刺激用户生产内容。内容推送上，多用表情包、短视频、直播等强视觉、强情感冲击的内容。对于产品，可以针对他们做爆款营销，打造“网红”产品。
                </p>

                <h3 class="text-xl mb-3 mt-6">“事实陈述区”社区 (低感性, 高理性) 评论示例</h3>
                <p class="mb-2">这类用户是冷静的观察者和问题的发现者。评价是“更新后FPS从60掉到40”、“3号BOSS有BUG”。</p>
                {fact_comments}
                <p class="business-insight">
                    <strong>沟通方式:</strong> 对这类用户，沟通应透明、真诚、高效。发布详细的更新日志（Patch Notes），建立公开的BUG追踪列表。内容推送上，多做技术拆解、数据报告、开发者Q&A。用事实和数据说话。对于产品，应关注其“核心功能”和“性能指标”，用清晰的数据标签来吸引他们。
                </p>
            </div>

            <div class="section-card">
                <h2 class="text-2xl mb-4">完善的用户评论评价体系</h2>
                <p class="mb-4">通过归一化指标，我们定义了两种新的评论价值：</p>

                <h3 class="text-xl mb-3">配置A：“核心玩家深度反馈价值” (CPIV)</h3>
                <p class="mb-2">精准定位出那些由最资深、最核心的玩家所写的、包含大量事实信息和深度思考的反馈。是产品经理和开发团队最需要倾听的声音。</p>
                <p class="mb-2"><strong>公式:</strong> $CPIV Score = (0.4 * playtime\_norm) + (0.4 * rational\_norm) + (0.2 * helpful\_norm) + (0.0 * emotional\_norm)$</p>
                <h4 class="text-lg mb-2 mt-4">高 CPIV 评论示例:</h4>
                {top_cpiv_comments}
                <p class="business-insight">
                    <strong>商业建议：</strong> 高 CPIV 评论是产品和开发团队的“金矿”。应建立专门的渠道收集和处理这些评论，例如定期评审高 CPIV 评论，将其转化为产品需求或缺陷报告。这些评论通常指向核心痛点或改进方向。
                </p>

                <h3 class="text-xl mb-3 mt-6">配置B：“病毒式传播潜力价值” (VPV)</h3>
                <p class="mb-2">识别出那些最有可能在社交媒体上被大量转发、讨论、点赞的“神评”或“爆款差评”。这些评论是市场和运营团队的宝贵素材，它们是品牌口碑的“放大器”和“风向标”。</p>
                <p class="mb-2"><strong>公式:</strong> $VPV Score = (0.5 * emotional\_norm) + (0.4 * helpful\_norm) + (0.1 * playtime\_norm) + (0.0 * rational\_norm)$</p>
                <h4 class="text-lg mb-2 mt-4">高 VPV 评论示例:</h4>
                {top_vpv_comments}
                <p class="business-insight">
                    <strong>商业建议：</strong> 高 VPV 评论是市场和运营团队的“利器”。可以利用这些评论进行内容营销，例如将“神评”制作成宣传海报、短视频素材，或将“爆款差评”作为改进的公开承诺，展示积极响应用户反馈的态度。
                </p>
            </div>

            <div class="section-card">
                <h2 class="text-2xl mb-4">数据指标体系总结与应用</h2>
                <ul class="list-disc pl-5 mb-4">
                    <li><strong>感性分数 & 理性分数：</strong> 构建用户画像矩阵，区分“情感宣泄区”和“事实陈述区”用户，制定差异化沟通和营销策略。</li>
                    <li><strong>CPIV Score：</strong> 帮助产品和开发团队精准定位核心玩家的深度反馈，指导产品迭代和优化。</li>
                    <li><strong>VPV Score：</strong> 赋能市场和运营团队识别具有病毒式传播潜力的内容，用于品牌宣传和舆情管理。</li>
                    <li><strong>负面关键词触发指数：</strong> 作为早期预警机制，快速发现潜在的产品问题和用户不满，避免舆情扩散。</li>
                </ul>
                <p class="business-insight">
                    <strong>整体策略：</strong> 结合这些新的指标，企业可以建立一个更全面、更敏感的用户反馈监控系统。通过自动化这些指标的计算和可视化，可以实现对用户声音的实时洞察，从而更快速、更精准地响应市场变化和用户需求，提升用户满意度和产品竞争力。
                </p>
            </div>

            <footer class="text-center text-gray-500 text-sm mt-8">
                报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </footer>
        </div>
    </body>
    </html>
    """
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logging.info(f"HTML report generated at {html_file_path}")

def main():
    # 配置日志
    log_dir = 'results'
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(os.path.join(log_dir, 'new_features_run.log'))

    logging.info("Starting New Features Analysis for Steam Reviews.")

    # 1. 加载数据
    data_path = 'data/labeled_comments.csv'
    df = load_data(data_path)
    if df is None:
        logging.error("Failed to load data. Exiting.")
        return

    # 确保评论列存在且为字符串类型 (现在直接使用中文列名 '评论')
    if '评论' not in df.columns:
        logging.error("DataFrame does not contain a '评论' column. Exiting.")
        return
    df['评论'] = df['评论'].astype(str)

    # 确保 '游戏时长' 和 'helpful' 列存在，如果不存在则创建默认值
    if '游戏时长' not in df.columns:
        df['游戏时长'] = 0 # 默认游戏时长为0
        logging.warning("Missing '游戏时长' column, defaulting to 0.")
    
    # === 关键修改：确保 'helpful' 列被包含在 df 中 ===
    if 'helpful' not in df.columns:
        df['helpful'] = 0 # 默认有帮助点赞数为0
        logging.warning("Missing 'helpful' column, defaulting to 0.")

    # 确保 'timestamp' 列存在并转换为日期时间对象
    # 假设 'timestamp' 列是 Unix 时间戳或可解析的日期字符串
    if 'timestamp' not in df.columns:
        # 如果没有时间戳，则创建一个假的，用于负面关键词触发指数
        df['timestamp'] = pd.to_datetime(datetime.now() - pd.to_timedelta(np.random.randint(0, 365, size=len(df)), unit='D'))
        logging.warning("Missing 'timestamp' column, generating dummy timestamps.")
    else:
        # 尝试将时间戳转换为 datetime 对象
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        except:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # 移除无法解析的时间戳

    # 初始化分析器和计算器
    sentiment_analyzer = SentimentAnalyzer()
    
    # 定义特定关键词和权重
    # 示例：游戏相关的技术词汇
    rational_keywords = ['bug', 'fps', '服务器', '优化', '崩溃', '延迟', '更新', '显卡', '配置', 'bug', '外挂', '闪退', '卡顿']
    # 示例：餐饮相关的技术词汇
    # rational_keywords = ['卡路里', '蛋白质', '出餐时间', '不新鲜', '馊', '有异物', '等太久', '送错', '变质', '拉肚子']
    rational_weights = {
        'comment_length': 0.25,
        'average_word_length': 0.25,
        'number_count': 0.25,
        'specific_keyword_count': 0.25
    }
    rational_calculator = RationalScoreCalculator(specific_keywords=rational_keywords, weights=rational_weights)

    # 定义负面关键词库
    negative_keywords = ['闪退', '崩溃', 'bug', '外挂', '服务器', '延迟', '无法启动', '不新鲜', '馊', '有异物', '等太久', '送错', '变质', '拉肚子'] # 示例
    advanced_metric_calculator = AdvancedMetricCalculator(negative_keywords=negative_keywords)

    # 2. 计算感性分数 (Emotional Score)
    # 如果CSV中已经有情感得分（例如您提供的“情感得分”列），可以选择是否重新计算
    # 这里我们假设您希望重新计算，以使用VADER的复合分数
    logging.info("Calculating emotional scores...")
    df['emotional_score'] = df['评论'].apply(sentiment_analyzer.get_emotional_score)
    logging.info("Emotional scores calculated.")

    # 3. 计算理性分数 (Rational Score)
    logging.info("Calculating rational scores...")
    df = rational_calculator.calculate_rational_score(df, comment_col='评论') # 使用中文列名
    logging.info("Rational scores calculated.")

    # 4. 归一化后的指标
    logging.info("Normalizing key metrics...")
    # 传递正确的列名给归一化函数
    df = advanced_metric_calculator.calculate_normalized_metrics(
        df, 
        playtime_col='游戏时长', # 使用中文列名
        helpful_col='helpful', # 即使没有此列，也会默认为0
        rational_score_col='rational_score', 
        emotional_score_col='emotional_score'
    )
    logging.info("Metrics normalized.")

    # 5. 计算 CPIV 和 VPV
    logging.info("Calculating CPIV and VPV scores...")
    df = advanced_metric_calculator.calculate_cpiv_score(df)
    df = advanced_metric_calculator.calculate_vpv_score(df)
    logging.info("CPIV and VPV scores calculated.")

    # 6. 负面关键词触发指数
    logging.info("Calculating negative keyword trigger index...")
    negative_keyword_trigger_index = advanced_metric_calculator.calculate_negative_keyword_trigger_index(df, comment_col='评论', timestamp_col='timestamp') # 使用中文列名
    logging.info(f"Negative Keyword Trigger Index (last 7 days): {negative_keyword_trigger_index:.4f}")

    # 7. 保存包含所有新指标的DataFrame
    # 确保保存时包含原始的中文列名，以及新计算的指标
    output_df = df[['用户名', '推荐', '游戏时长', '评论', '政治敏感', # 原始列
                    'emotional_score', 'rational_score', # 新计算的感性/理性分数
                    'playtime_norm', 'rational_norm', 'emotional_norm', 'helpful_norm', # 归一化指标
                    'CPIV_Score', 'VPV_Score', 'timestamp', 'helpful']].copy() # 新的复合指标和时间戳，以及原始helpful列
    save_results_csv(output_df, os.path.join(log_dir, 'new_metrics_analysis'), 'enriched_comments_with_new_metrics.csv')
    logging.info("Enriched comments with new metrics saved.")

    # 8. 生成HTML报告
    logging.info("Generating HTML report...")
    generate_html_report(output_df, negative_keyword_trigger_index, os.path.join(log_dir, 'html_report'))
    logging.info("HTML report generation finished.")

    logging.info("New Features Analysis Project finished.")

if __name__ == "__main__":
    main()
