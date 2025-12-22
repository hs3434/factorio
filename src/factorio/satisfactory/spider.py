import requests
import re
import os
import json
import bs4
from bs4 import BeautifulSoup
import warnings
# 忽略requests的安全警告（可选）
# warnings.filterwarnings('ignore')

formula_path = os.path.join(os.path.dirname(__file__), 'formula.json')

default_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def deal_item(item: bs4.element.Tag):
    text = item.get_text(strip=True).replace(",", "")
    match1 = re.search(r"\((.*?) / min\)", text)
    speed = float(match1.group(1).replace("m³", ""))
    text2 = text[match1.end():]
    if "m³" in text2:
        match2 = re.search(r"(.*)m³(.*)", text2)
    else:
        match2 = re.search(r"(.*)x(.*)", text2)
    num = float(match2.group(1))
    name = match2.group(2)
    return {name: {"num": num, "speed": speed, "name": name}}

def deal_formula(formula_section: bs4.element.Tag, type_str: str = "formula"):
    formula_item = formula_section.find_all("div", recursive=False)
    info = formula_item[0].find_all("div", recursive=False)
    name = info[0].get_text(strip=True).replace("替代：", "")
    by = re.search(r"\((.*?)\)", info[1].get_text(strip=True)).group(1)
    formula = formula_item[1].find_all("div",class_="col-6")
    inputs = {}
    outputs = {}
    for input in formula[0].find_all("div", recursive=False):
        inputs.update(deal_item(input))
    for output in formula[1].find_all("div", recursive=False):
        outputs.update(deal_item(output))
    return {"name": name, "by": by, "type": type_str, "inputs": inputs, "outputs": outputs}
        
        
class SatisfactorySpider:
    def __init__(self, headers = default_headers):
        self.headers = headers
        self.base_url = "https://satisfactory-calculator.com/"
        self.items_path = {}
        self.all_formula = {}
    
    def get_all_formula(self):
        if not self.items_path:
            self.get_items_list()
        for item in self.items_path:
            item_info = self.get_item_info(item)
            self.all_formula[item] = item_info
        return self.all_formula

    def deal_items_html(self, soup: BeautifulSoup, item_name):
        base_info = {}
        main = soup.find("main").find_all("div", recursive=False)[1]
        # info
        media_body = main.find('div', class_='media-body')
        base_info["描述"] = media_body.find("div", class_="card-body").get_text(strip=True)
        
        ul_list = media_body.find_all('ul', class_='list-group')
        base_info["类别"] = ul_list[0].find("strong").get_text(strip=True)
        pattern1 = re.compile(r'堆叠数量(\d+)')
        pattern2 = re.compile(r'能量(\d+)MJ')
        pattern3 = re.compile(r'资源槽点数(.*)')
        for ul in ul_list[1:]:
            text = ul.get_text(strip=True).replace(",", "")
            if pattern1.search(text):
                base_info["堆叠数量"] = int(pattern1.search(text).group(1))
            elif pattern2.search(text):
                base_info["能量"] = int(pattern2.search(text).group(1))
            elif pattern3.search(text):
                base_info["资源槽点数"] = pattern3.search(text).group(1)
                try:
                    base_info["资源槽点数"] = int(base_info["资源槽点数"])
                except:
                    pass


        # formula        
        formula_all = main.find_all("div", recursive=False)[2].find_all("div", recursive=False)
        formula_list = []
        if len(formula_all) == 2:
            formula_list += [deal_formula(formula_section, type_str="formula") for formula_section in formula_all[0].find_all("div",class_="card-body")]
            formula_list += [deal_formula(formula_section, "replace_formula") for formula_section in formula_all[1].find_all("div",class_="card-body")]
        elif len(formula_all) == 1 and formula_all[0].get_text(strip=True):
            formula_list += [deal_formula(formula_section) for formula_section in formula_all[0].find_all("div",class_="card-body")]
        else:
            print(f"{item_name} 没有公式")
        
        return {"base_info": base_info, "formula_list": formula_list}
        
    def get_item_info(self, item_name):
        if item_name not in self.items_path:
            return None
        headers = self.headers
        url = os.path.join(self.base_url + self.items_path[item_name])
        timeout = 10
        item_info = {}
        
        try:
            # 1. 发送请求获取网页源码
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding  # 自动识别编码
            soup = BeautifulSoup(response.text, 'lxml')  # lxml解析器更高效（需安装：pip install lxml）

            try:
                item_info = self.deal_items_html(soup, item_name)
                item_info["base_info"].update({"url": url, "name": item_name})
                print("处理物品信息成功：", item_name)
            except Exception as e:
                print("处理物品信息失败：", item_name)
                print(e)
        except Exception as e:
            print("获取物品信息失败：", item_name)
            print(e)
        return item_info
        
    def get_items_list(self):
        """
        获取物品列表页面中的所有链接（即所有物品的详细页面）
        :return: dict[name, path] 所有物品链接列表，每个元素是一个字典，键为 name 值为 path 
        """
        pattern = ".*zh/items/detail/id/.*"
        url = os.path.join(self.base_url, "zh/items")
        headers = self.headers
        timeout = 10
        pattern = re.compile(pattern)
        items_path = {}
        
        try:
            # 1. 发送请求获取网页源码
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()  # 抛出HTTP错误（如404/500）
            response.encoding = response.apparent_encoding  # 自动识别编码

            # 2. 解析HTML
            soup = BeautifulSoup(response.text, 'lxml')  # 也可用'lxml'（需先安装：pip install lxml）

            # 3. 筛选href包含指定字符的a标签
            # 遍历所有a标签
            for a_tag in soup.find_all('a', href=True):  # href=True：只筛选有href属性的a标签
                href = a_tag['href']
                # 判断href是否包含目标字符（支持单个字符/多个字符）
                match = pattern.match(href)
                if match:
                    # 4. 提取strong子元素的文本（兼容无strong的情况）
                    strong_tag = a_tag.find('strong')  # 只找第一个strong
                    strong_text = strong_tag.get_text(strip=True) if strong_tag else ''
                    
                    # 整理结果（处理相对路径/绝对路径）
                    if strong_text:
                        if strong_text in items_path:
                            print(f"{strong_text} 重复：{href}")
                        else:
                            items_path[strong_text] = href
                        

        except requests.exceptions.RequestException as e:
            print(f"请求失败：{e}")
        except Exception as e:
            print(f"解析失败：{e}")
        self.items_path = items_path
        return items_path

# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    spider = SatisfactorySpider()
    spider.get_all_formula()
    with open(formula_path, 'w', encoding='utf-8') as f:
        json.dump(spider.all_formula, f, ensure_ascii=False, indent=4)
