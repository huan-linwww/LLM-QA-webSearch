import requests
import json
import re
from lxml import etree
import chardet
import jieba.analyse
from langchain_openai import OpenAIEmbeddings
from config import Config
import torch.nn.functional as F
from torch import cosine_similarity
import torch

# 设置embedding
embeddings_model = OpenAIEmbeddings(openai_api_key="",
                                    openai_api_base="")


def get_sim(vectors):
    """以[query，text1，text2...]来计算query与text1，text2,...的cosine相似度"""
    vectors = F.normalize(vectors, p=2, dim=1)
    q_vec = vectors[0, :]
    o_vec = vectors[1:, :]
    sim = cosine_similarity(q_vec, o_vec)
    sim = sim.data.cpu().numpy().tolist()
    return sim


def search_bing(query):
    """利用newbing搜索接口，用于检索与query相关的背景信息，作为检索内容
    input：query
    output：{'url':'','text':'','title':''}
    """

    # Add your Bing Search V7 subscription key and endpoint to your environment variables.
    subscription_key = ""
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    # Construct a request
    mkt = 'zh-CN'
    params = {'q': query, 'mkt': mkt}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    # Call the API
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()

        # print("\nHeaders:\n")
        # print(response.headers)

        #print("\nJSON Response:\n")
        webPages = [item['snippet'] for item in response.json()["webPages"]['value']]
        # pprint(webPages)
    except Exception as ex:
        raise ex
    return webPages


class TextRecallRank():
    """
    实现对检索内容的召回与排序
    """

    def __init__(self, cfg):
        self.topk = cfg.topk  # query关键词召回的数量
        self.topd = cfg.topd  # 召回文章的数量
        self.topt = cfg.topt  # 召回文本片段的数量
        self.maxlen = cfg.maxlen  # 召回文本片段的长度
        self.recall_way = cfg.recall_way  # 召回方式

    def query_analyze(self, query):
        """query的解析，目前利用jieba进行关键词提取
        input:query,topk
        output:
            keywords:{'word':[]}
            total_weight: float number
        """
        keywords = jieba.analyse.extract_tags(query, topK=self.topk, withWeight=True)
        total_weight = self.topk / sum([r[1] for r in keywords])
        return keywords, total_weight

    def text_segmentate(self, text, maxlen, seps='\n', strips=None):
        """将文本按照标点符号划分为若干个短句
        """
        text = text.strip().strip(strips)
        if seps and len(text) > maxlen:
            pieces = text.split(seps[0])
            text, texts = '', []
            for i, p in enumerate(pieces):
                if text and p and len(text) + len(p) > maxlen - 1:
                    texts.extend(self.text_segmentate(text, maxlen, seps[1:], strips))
                    text = ''
                if i + 1 == len(pieces):
                    text = text + p
                else:
                    text = text + p + seps[0]
            if text:
                texts.extend(self.text_segmentate(text, maxlen, seps[1:], strips))
            return texts
        else:
            return [text]

    def recall_title_score(self, title, keywords, total_weight):
        """计算query与标题的匹配度"""
        score = 0
        for item in keywords:
            kw, weight = item
            if kw in title:
                score += round(weight * total_weight, 4)
        return score

    def recall_text_score(self, text, keywords, total_weight):
        """计算query与text的匹配程度"""
        score = 0
        for item in keywords:
            kw, weight = item
            p11 = re.compile('%s' % kw)
            pr = p11.findall(text)
            # score += round(weight * total_weight, 4) * len(pr)
            score += round(weight * total_weight, 4)
        return score

    def rank_text_by_keywords(self, query, data):
        """通过关键词进行召回"""

        # query分析
        keywords, total_weight = self.query_analyze(query)

        # 先召回title
        title_score = {}
        for line in data:
            title = line['title']
            title_score[title] = self.recall_title_score(title, keywords, total_weight)
        title_score = sorted(title_score.items(), key=lambda x: x[1], reverse=True)
        recall_title_list = [t[0] for t in title_score[:self.topd]]

        # 召回sentence
        sentence_score = {}
        for line in data:
            title = line['title']
            text = line['text']
            if title in recall_title_list:
                for ct in self.text_segmentate(text, self.maxlen, seps='\n。'):
                    ct = re.sub('\s+', ' ', ct)
                    if len(ct) >= 20:
                        sentence_score[ct] = self.recall_text_score(ct, keywords, total_weight)

        sentence_score = sorted(sentence_score.items(), key=lambda x: x[1], reverse=True)
        recall_sentence_list = [s[0] for s in sentence_score[:self.topt]]
        return '\n'.join(recall_sentence_list)

    def rank_text_by_text2vec(self, query, data):
        """通过text2vec召回"""

        # 召回sentence
        sentence_list = [query]
        for line in data:
            for ct in self.text_segmentate(line, self.maxlen, seps='\n。'):
                ct = re.sub('\s+', ' ', ct)
                if len(ct) >= 20:
                    sentence_list.append(ct)

        # sentence_vectors = get_vector(sentence_list, 8)
        sentence_vectors = torch.tensor(embeddings_model.embed_documents(sentence_list))
        # print(f"嵌入个数:{len(sentence_vectors)}")
        sentence_score = get_sim(sentence_vectors)
        sentence_score = dict(zip(sentence_score, range(1, len(sentence_list))))
        sentence_score = sorted(sentence_score.items(), key=lambda x: x[0], reverse=True)
        recall_sentence_list = [sentence_list[s[1]] for s in sentence_score[:self.topt]]
        return '\n'.join(recall_sentence_list)

    def query_retrieve(self, query):

        # 利用搜索引擎获取相关信息
        data = search_bing(query)
        # 对获取的相关信息进行召回与排序，得到背景信息
        if self.recall_way == 'keyword':
            bg_text = self.rank_text_by_keywords(query, data)
        else:
            bg_text = self.rank_text_by_text2vec(query, data)
        return bg_text


cfg = Config()
trr = TextRecallRank(cfg)
q_searching = trr.query_retrieve
if __name__ == "__main__":
    cfg = Config()
    trr = TextRecallRank(cfg)
    q_searching = trr.query_retrieve
    # q_searching = trr.query_retrieve("什么是机器学习")
    # print(q_searching)
