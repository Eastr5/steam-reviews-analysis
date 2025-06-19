"""
Steam评论和用户信息爬虫。
此模块从Steam游戏页面爬取评论和用户信息。
"""
import requests
import time
import xlwt
import urllib
import bs4
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Any, Optional


class SteamCrawler:
    """Steam游戏评论爬虫类。"""
    
    def __init__(self, appid: str, max_data_size: int = 1000):
        """
        初始化爬虫，设置Steam应用ID和数据大小。
        
        参数:
            appid: Steam游戏的应用ID
            max_data_size: 要爬取的最大评论数量（必须是10的倍数）
        """
        self.appid = appid
        self.max_data_size = int(max_data_size)
        self.app_name = "未知游戏"
        self.list_all_content = []  # 所有评论数据
        self.next_cursor = "*"  # 下一页评论的索引
        self.reload_data_num = 0  # 重试次数
        self.reload_data_num_max = 5  # 最大重试次数 = 5
        self.lasted_num = 0
        self.total_end_num = 0
        
    @staticmethod
    def dict_encode_url(data: dict) -> str:
        """
        将字典编码为URL参数。
        
        参数:
            data: URL参数字典
            
        返回:
            编码后的URL参数字符串
        """
        return '&'.join(f"{urllib.parse.quote(i, encoding='utf-8')}={urllib.parse.quote(data[i], encoding='utf-8')}" for i in data)
        
    def get_game_name(self) -> None:
        """
        从Steam商店页面获取游戏名称。
        同时对游戏名称进行处理，确保文件名安全。
        """
        try:
            resp = requests.get(f"https://store.steampowered.com/app/{self.appid}", timeout=5)
            soup = BeautifulSoup(resp.text, "lxml")
            
            for business in soup.find_all('span', {'itemprop': ['name']}):
                print("获取游戏名字:")
                self.app_name = business.string
                
                # 处理游戏名称，确保文件名安全
                for char in [':', '/', '|', '?', '？', '>', '<']:
                    self.app_name = self.app_name.replace(char, "-")
                    
                print(self.app_name)
            time.sleep(1)
        except Exception as e:
            print(f"获取游戏名称时出错: {e}")
            
    def get_reviews(self) -> bool:
        """
        使用当前游标获取一批评论。
        
        返回:
            bool: 如果可能还有更多评论则返回True，如果结束或出错则返回False
        """
        is_end = False
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        # 尝试获取评论
        try:
            resp = requests.get(
                f"https://store.steampowered.com/appreviews/{self.appid}?cursor={self.next_cursor}&language=schinese&day_range=365&review_type=all&purchase_type=all&filter=recent",
                timeout=5).json()
        except Exception as e:
            print(f"请求评论失败，尝试重新拉取...{self.reload_data_num}")
            self.reload_data_num += 1
            time.sleep(1)
            return not is_end
            
        # 成功后重置重试计数器
        self.reload_data_num = 0
        print(f"请求完成，当前数据列表长度 = {len(self.list_all_content)}")
        
        # 检查是否仍在获取新评论
        if self.lasted_num == len(self.list_all_content):
            self.total_end_num += 1
            print(f"请求不到更多评论...{self.total_end_num}")
            if self.total_end_num >= self.reload_data_num_max:
                print("结束请求...")
                is_end = True
                self.reload_data_num = 5
                time.sleep(1)
                return not is_end
        else:
            self.lasted_num = len(self.list_all_content)
            self.total_end_num = 0
            
        # 更新下一页游标
        cursor = resp["cursor"]
        cursor = cursor.replace("+", "%2B")
        self.next_cursor = cursor
        
        # 解析包含评论的HTML
        html = resp["html"]
        soup = BeautifulSoup(html, "lxml")
        
        # 提取用户名
        list_name = []
        for business in soup.find_all('div', class_="persona_name"):
            for bb in business.find_all('a'):
                list_name.append(bb.string)
                
        # 提取推荐（正面/负面）
        list_recommend = []
        for business in soup.find_all('div', class_="title ellipsis"):
            list_recommend.append(business.string)
            
        # 提取游戏时长
        list_time = []
        for business in soup.find_all('div', class_="hours ellipsis"):
            text1 = business.text.replace("\r\n\t\t\t\t\t\t", "")
            text2 = text1.replace("\t\t\t\t\t\t\t\t\t\t\t", "")
            list_time.append(text2)
            
        # 提取评论内容
        list_comment = []
        for business in soup.find_all('div', class_="content"):
            text1 = business.text.replace("\r\n\t\t\t\t\t", "")
            text2 = text1.replace("\t\t\t\t\t\n", "")
            list_comment.append(text2)
            
        # 合并所有数据
        for number in range(0, len(list_comment)):
            list1 = []
            list1.append(list_name[number])
            list1.append(list_recommend[number])
            list1.append(list_time[number])
            list1.append(list_comment[number])
            self.list_all_content.append(list1)
            
        return not is_end
        
    def save_to_excel(self) -> None:
        """将收集到的评论数据保存到Excel文件。"""
        print("开始存储!")
        book = xlwt.Workbook(encoding="utf-8", style_compression=0)
        sheet = book.add_sheet("游戏数据", cell_overwrite_ok=True)
        
        # 写入列标题
        col = ["用户名", "推荐", "游戏时长", "评论"]
        for i in range(0, 4):
            sheet.write(0, i, col[i])
            
        # 写入数据行
        for i in range(1, len(self.list_all_content) + 1):
            print(f'已经存储 {i - 1} 行数据')
            for j in range(0, len(self.list_all_content[i-1])):
                sheet.write(i, j, self.list_all_content[i - 1][j])
                
        # 保存文件
        filename = f"游戏[{self.app_name}][appid {self.appid}]评论数据[{self.max_data_size}条].xls"
        print(f"保存到: {filename}")
        book.save(filename)

    def run(self) -> None:
        """运行完整的爬取过程。"""
        self.get_game_name()
        
        while len(self.list_all_content) < self.max_data_size:
            has_more = self.get_reviews()
            time.sleep(0.5)
            
            if not has_more:
                if self.reload_data_num >= 5:
                    print("连接超时 5 次，结束")
                    break
                    
        self.save_to_excel()
        time.sleep(1)


def main():
    """运行爬虫的主入口点。"""
    appid = input("steam：请输入appid：")
    max_data_size = input("请输入要请求数据的条数(10的倍数)：")
    
    crawler = SteamCrawler(appid, max_data_size)
    crawler.run()


if __name__ == '__main__':
    main()