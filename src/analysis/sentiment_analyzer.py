# steam-reviews-analysis/src/analysis/sentiment_analyzer.py
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

class SentimentAnalyzer:
    """
    用于计算文本感性分数的类。
    """
    def __init__(self):
        # 确保 VADER 词典已下载
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except nltk.downloader.DownloadError:
            nltk.download('vader_lexicon')
        self.analyzer = SentimentIntensityAnalyzer() # 初始化 VADER 情感分析器

    def get_emotional_score(self, text):
        """
        使用 VADER 情感分析器计算文本的复合情感分数。
        分数范围从 -1 (极度负面) 到 +1 (极度正面)。
        :param text: 输入文本。
        :return: 文本的复合情感分数。
        """
        if not isinstance(text, str):
            text = str(text) # 确保输入是字符串
        vs = self.analyzer.polarity_scores(text)
        return vs['compound']

