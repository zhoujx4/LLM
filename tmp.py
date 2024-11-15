"""
@Time : 2024/7/214:35
@Auth : zhoujx
@File ：tmp.py
@DESCRIPTION:

"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from sentence_transformers import SentenceTransformer

from collections import Counter
from sklearn.cluster import \
    AgglomerativeClustering  # 注意：这里可能是笔误，应该是 AgglomerativeClustering -> AgglomerativeClustering 或 AgglomerativeClustering -> AgglomerativeClustering
import re
import pandas as pd


def preprocess_corpus(df, col_name):
    """
    预处理语料库，包括分割、清洗和去除重复项。
    """
    corpus = df[col_name].dropna().apply(lambda x: re.split(r'、|\||,|，', x)).tolist()
    corpus = [y.strip() for x in corpus for y in x if y.strip() and y.strip() != '无法回答']
    corpus = list(set(corpus))  # 去除重复项
    return corpus


def cluster_words(corpus, embedder, distance_threshold=0.4):
    """
    对预处理后的语料库进行聚类。
    """
    corpus_embeddings = embedder.encode(corpus)
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append(corpus[sentence_id])
    return clustered_sentences


def find_representative_words(clustered_sentences, word_counts):
    """
    为每个聚类找到代表性的词。
    """
    word_norm_2_word = {}
    for words in clustered_sentences.values():
        max_count = 0
        representative_word = None
        for word in words:
            if word_counts[word] > max_count:
                max_count = word_counts[word]
                representative_word = word
        word_norm_2_word[representative_word] = words
    return word_norm_2_word


def build_word_mappings(word_norm_2_word):
    """
    构建从词到代表词的映射。
    """
    word_2_word_norm = {}
    for word_norm, words in word_norm_2_word.items():
        for word in words:
            word_2_word_norm[word] = word_norm
    return word_2_word_norm


embedder = SentenceTransformer("/data/ethan/project/beijixing/data/models/bge-m3")

df = pd.read_excel(
    '/mnt/SSD_12TB/junius/llm_evaluation/paoshu/data/三元组码表/【内部】vivoX100u-首销周测试数据-修改维度名_生成码表socialgpts_postprocess.xlsx')

df = df.dropna(subset=['二级维度标签'])
second_2_cnt = df['二级维度标签'].value_counts().to_dict()
df = df.loc[df['二级维度标签'].apply(lambda x: second_2_cnt[x] >= 5)]
result = {}
result_word_norm_2_word = {}
result_word_2_word_norm = {}
results = []

for first in df['一级维度标签'].unique():
    for col_name in ['二级维度标签']:
        df_tmp = df.loc[df['一级维度标签'] == first]
        corpus = preprocess_corpus(df_tmp, col_name)
        word_counts = Counter(corpus)
        if len(corpus) == 1:
            results.append([first, col_name, corpus])
            continue
        clustered_sentences = cluster_words(corpus, embedder, distance_threshold=1)
        word_norm_2_word = find_representative_words(clustered_sentences, word_counts)
        word_2_word_norm = build_word_mappings(word_norm_2_word)

        # 对聚类结果进行排序和格式化
        tmp = list(clustered_sentences.values())
        tmp.sort(key=lambda x: len(x), reverse=True)
        tmp = ["|".join(x) for x in tmp]

        # 存储结果
        result[col_name] = tmp
        result_word_norm_2_word[col_name] = word_norm_2_word
        result_word_2_word_norm[col_name] = word_2_word_norm

        print(col_name, len(clustered_sentences), '聚类结果')
        print(clustered_sentences)

        results.append([first, col_name, tmp])

df_results = pd.DataFrame(results)
df_results = df_results.explode(2)
df_keyword = pd.DataFrame()

for key in result:
    df_keyword = pd.concat([df_keyword, pd.DataFrame(result[key], columns=[key])], axis=1)

df_results.to_excel('/mnt/SSD_12TB/junius/三级维度聚类.xlsx', index=False)
