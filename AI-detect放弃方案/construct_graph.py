# import sys
# import os
# import json
# import nltk
# import re
# from transformers import RobertaTokenizer
# from tqdm import tqdm
# import argparse

# # 获取当前脚本的目录
# script_dir = os.path.dirname(os.path.abspath(__file__)) 
# # 获取当前脚本父目录路径
# parent_dir = os.path.dirname(script_dir)  
# # 将父目录路径添加到系统路径，以便导入自定义模块
# sys.path.append(parent_dir)

# from utils.common import *  # 导入自定义的公共函数和模块

# # 构建图的函数
# def build_graph(all_info):
#     """
#     根据所有信息构建图结构，返回节点、边、实体出现次数、句子列表以及句子到节点的映射。

#     :param all_info: 包含句子和关键字的列表
#     :return: 节点列表、边列表、实体出现次数、句子列表、句子到节点的映射
#     """
#     nodes = []            # 用于存储图中的节点信息
#     edges = []            # 用于存储图中的边信息
#     entity_occur = {}     # 用于记录实体的出现次数
#     last_sen_cnt = 0      # 用于计算当前节点在图中的位置
#     sens = [sen["sentence"].replace("\t", " ") for sen in all_info]  # 提取所有句子
#     all_kws = [sen["keywords"]["entity"] for sen in all_info]  # 提取每句中的关键字
#     sen2node = []         # 存储句子到节点的映射关系
    
#     print(f"Total sentences: {len(sens)}")  # 输出总句子数

#     for sen_idx, sen_kws in enumerate(all_kws):
#         sen_tmp_node = []  # 暂时存储当前句子的节点
#         kws_cnt = 0        # 当前句子的关键字计数
#         # 去除停用词并检查句子是否符合条件
#         sen_kws = list([kw for kw in set(sen_kws) if kw.strip() not in cachedStopWords])
#         if not keep_sen(sen_kws):  # 跳过无关句子
#             sen2node.append([])
#             continue

#         # 遍历当前句子的所有关键字
#         for _, kw in enumerate(sen_kws):
#             # 清理关键字字符串
#             kw = re.sub(r"[^a-zA-Z0-9,.'\`!?]+", " ", kw)
#             words = [
#                 word
#                 for word in nltk.word_tokenize(kw)
#                 if (
#                     word not in cachedStopWords
#                     and word.capitalize() not in cachedStopWords
#                 )
#             ]
#             # 检查关键字是否可以作为节点
#             if keep_node(kw, words):
#                 sen_tmp_node.append(len(nodes))
#                 nodes.append({"text": kw, "words": words, "sentence_id": sen_idx})
#                 if kw not in entity_occur.keys():
#                     entity_occur[kw] = 0
#                 entity_occur[kw] += 1
#                 kws_cnt += 1

#         # 创建句子内部的节点连边关系
#         edges += [
#             tuple([last_sen_cnt + i, last_sen_cnt + i + 1, "inner"])
#             for i in list(range(kws_cnt - 1))
#         ]

#         # 更新计数和句子到节点的映射
#         last_sen_cnt += kws_cnt
#         sen2node.append(sen_tmp_node)

#     # 创建跨句子节点的连边关系
#     for i in range(len(nodes)):
#         for j in range(len(nodes)):
#             if j == i:
#                 continue
#             # 检查两个节点是否是相同的实体
#             if nodes[i]["text"].strip() == nodes[j]["text"].strip():
#                 edges.append(tuple([min(i, j), max(i, j), "inter"]))

#     print(f"Total nodes: {len(nodes)}, Total edges: {len(edges)}")  # 输出总节点数和边数
    
#     if not nodes:
#         return [], [], [], [], []
#     return nodes, list(set(edges)), entity_occur, sens, sen2node

# # 清理字符串的函数
# def clean_string(string):
#     """
#     清理字符串中无效字符，保留字母、数字和部分标点符号。
#     """
#     return re.sub(r"[^a-zA-Z0-9,.'!?]+", "", string)

# # 基于图生成代表掩码的函数
# def generate_rep_mask_based_on_graph(ent_nodes, sens, tokenizer):
#     """
#     基于图生成代表掩码，包含节点、所有标记、需要去除的节点和句子的索引对。

#     :param ent_nodes: 图中的节点
#     :param sens: 句子列表
#     :param tokenizer: 分词器
#     :return: 更新后的节点、所有标记、需要去除的节点、句子索引对
#     """
#     sen_start_idx = [0]  # 记录每句的起始索引
#     sen_idx_pair, sen_tokens, all_tokens, drop_nodes = [], [], [], []

#     for idx, sen in enumerate(sens):
#         # 对句子进行分词
#         sen_token = tokenizer.tokenize(sen)
#         cleaned_sen_token = [clean_string(token) for token in sen_token]
#         sen_tokens.append(cleaned_sen_token)
#         sen_idx_pair.append(
#             tuple([sen_start_idx[-1], sen_start_idx[-1] + len(sen_token)])
#         )
#         sen_start_idx.append(sen_start_idx[-1] + len(sen_token))
#         all_tokens += sen_token

#     for nidx, node in enumerate(ent_nodes):
#         node_text = node["text"]
#         # 找到节点文本的起始位置
#         start_pos, node_len = first_index_list(
#             sen_tokens[node["sentence_id"]], clean_string(node_text)
#         )
#         if start_pos != -1:
#             final_start_pos = sen_start_idx[node["sentence_id"]] + start_pos
#             max_pos = final_start_pos + node_len
#             ent_nodes[nidx]["spans"] = tuple([final_start_pos, max_pos])
#         else:
#             ent_nodes[nidx]["spans"] = tuple([-1, -1])
#         if ent_nodes[nidx]["spans"][0] == -1:
#             drop_nodes.append(nidx)

#     return ent_nodes, all_tokens, drop_nodes, sen_idx_pair

# # 设置命令行参数解析
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--kw_file_dir",
#     type=str,
#     required=True,
#     help="The path of the input dataset with keywords.",
# )
# parser.add_argument(
#     "--output_dir",
#     type=str,
#     default="",
#     help="The path to the output dataset with graph.",
# )
# args = parser.parse_args()

# if __name__ == "__main__":
#     print("Starting graph construction process...")  # 开始运行的提示
#     if args.output_dir == "":  # 如果未指定输出路径，则使用默认路径
#         args.output_dir = args.kw_file_dir.replace("_kws.jsonl", "_graph.jsonl")
    
#     print("Loading Dataset ...")
#     data = read_data(args.kw_file_dir)
#     print(f"Loaded {len(data)} instances from dataset.")

#     print("Loading Tokenizer ...")
#     local_tokenizer_path = "/home/yyh/yyh/PythonProject/Coh-MGT-Detection/preprocess/roberta-base"
#     tokenizer = RobertaTokenizer.from_pretrained(local_tokenizer_path, do_lower_case=False)
    
#     no_node = 0  # 记录没有节点的实例数量

#     with open(args.output_dir, "w", encoding="utf8") as outf:
#         for idx, line in tqdm(enumerate(data), desc="Processing Instances"):
#             kws = line["information"]["keywords"]
#             # 构建图
#             nodes, edges, entity_occur, sens, sen2node = build_graph(kws)
#             if not nodes:
#                 no_node += 1
#             else:
#                 print(f"Instance {idx + 1}: {len(nodes)} nodes, {len(edges)} edges.")
#             # 基于图生成掩码
#             nodes, all_tokens, drop_nodes, sen_idx_pair = generate_rep_mask_based_on_graph(nodes, sens, tokenizer)

#             # 添加图信息到数据
#             line["information"]["graph"] = {
#                 "nodes": nodes,
#                 "edges": edges,
#                 "all_tokens": all_tokens,
#                 "drop_nodes": drop_nodes,
#                 "sentence_to_node_id": sen2node,
#                 "sentence_start_end_idx_pair": sen_idx_pair,
#             }
#             outf.write(json.dumps(line) + "\n")

#     print(f"{no_node} instances are too short that have no graph.")  # 输出没有节点的实例数量
import sys
import os
import json
import nltk
import re
from transformers import RobertaTokenizer
from tqdm import tqdm
import argparse
import numpy as np

# 获取当前脚本的目录
script_dir = os.path.dirname(os.path.abspath(__file__)) 
# 获取当前脚本父目录路径
parent_dir = os.path.dirname(script_dir)  
# 将父目录路径添加到系统路径，以便导入自定义模块
sys.path.append(parent_dir)

from utils.common import *  # 导入自定义的公共函数和模块

# 计算两个实体的上下文相似性
def compute_context_similarity(window_i, window_j):
    """
    计算两个上下文窗口之间的相似度。
    :param window_i: 实体 i 的上下文窗口
    :param window_j: 实体 j 的上下文窗口
    :return: 相似度
    """
    # 这里简单使用Jaccard相似度，其他的可以根据需求替换为余弦相似度等
    set_i = set(window_i)
    set_j = set(window_j)
    intersection = len(set_i & set_j)
    union = len(set_i | set_j)
    return intersection / union if union != 0 else 0

# 构建图的函数
def build_graph(all_info):
    """
    根据所有信息构建图结构，返回节点、边、实体出现次数、句子列表以及句子到节点的映射。
    :param all_info: 包含句子和关键字的列表
    :return: 节点列表、边列表、实体出现次数、句子列表、句子到节点的映射
    """
    nodes = []            # 用于存储图中的节点信息
    edges = []            # 用于存储图中的边信息
    entity_occur = {}     # 用于记录实体的出现次数
    last_sen_cnt = 0      # 用于计算当前节点在图中的位置
    sens = [sen["sentence"].replace("\t", " ") for sen in all_info]  # 提取所有句子
    all_kws = [sen["keywords"]["entity"] for sen in all_info]  # 提取每句中的关键字
    sen2node = []         # 存储句子到节点的映射关系
    
    print(f"Total sentences: {len(sens)}")  # 输出总句子数

    for sen_idx, sen_kws in enumerate(all_kws):
        sen_tmp_node = []  # 暂时存储当前句子的节点
        kws_cnt = 0        # 当前句子的关键字计数
        # 去除停用词并检查句子是否符合条件
        sen_kws = list([kw for kw in set(sen_kws) if kw.strip() not in cachedStopWords])
        if not keep_sen(sen_kws):  # 跳过无关句子
            sen2node.append([])
            continue

        # 遍历当前句子的所有关键字
        for _, kw in enumerate(sen_kws):
            # 清理关键字字符串
            kw = re.sub(r"[^a-zA-Z0-9,.'\`!?]+", " ", kw)
            words = [
                word
                for word in nltk.word_tokenize(kw)
                if (
                    word not in cachedStopWords
                    and word.capitalize() not in cachedStopWords
                )
            ]
            # 检查关键字是否可以作为节点
            if keep_node(kw, words):
                sen_tmp_node.append(len(nodes))
                nodes.append({"text": kw, "words": words, "sentence_id": sen_idx})
                if kw not in entity_occur.keys():
                    entity_occur[kw] = 0
                entity_occur[kw] += 1
                kws_cnt += 1

        # 创建句子内部的节点连边关系
        edges += [
            tuple([last_sen_cnt + i, last_sen_cnt + i + 1, "inner"])
            for i in list(range(kws_cnt - 1))
        ]

        # 更新计数和句子到节点的映射
        last_sen_cnt += kws_cnt
        sen2node.append(sen_tmp_node)

    # 创建跨句子节点的连边关系
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if j == i:
                continue
            # 计算两个节点的上下文相似性
            context_i = nodes[i]["words"]
            context_j = nodes[j]["words"]
            sim_score = compute_context_similarity(context_i, context_j)
            if nodes[i]["text"].strip() == nodes[j]["text"].strip():
                edges.append(tuple([min(i, j), max(i, j), "inter", sim_score]))

    print(f"Total nodes: {len(nodes)}, Total edges: {len(edges)}")  # 输出总节点数和边数
    
    if not nodes:
        return [], [], [], [], []
    return nodes, list(set(edges)), entity_occur, sens, sen2node

# 清理字符串的函数
def clean_string(string):
    """
    清理字符串中无效字符，保留字母、数字和部分标点符号。
    """
    return re.sub(r"[^a-zA-Z0-9,.'!?]+", "", string)

# 基于图生成代表掩码的函数
def generate_rep_mask_based_on_graph(ent_nodes, sens, tokenizer):
    """
    基于图生成代表掩码，包含节点、所有标记、需要去除的节点和句子的索引对。
    :param ent_nodes: 图中的节点
    :param sens: 句子列表
    :param tokenizer: 分词器
    :return: 更新后的节点、所有标记、需要去除的节点、句子索引对
    """
    sen_start_idx = [0]  # 记录每句的起始索引
    sen_idx_pair, sen_tokens, all_tokens, drop_nodes = [], [], [], []

    for idx, sen in enumerate(sens):
        # 对句子进行分词
        sen_token = tokenizer.tokenize(sen)
        cleaned_sen_token = [clean_string(token) for token in sen_token]
        sen_tokens.append(cleaned_sen_token)
        sen_idx_pair.append(
            tuple([sen_start_idx[-1], sen_start_idx[-1] + len(sen_token)]))
        sen_start_idx.append(sen_start_idx[-1] + len(sen_token))
        all_tokens += sen_token

    for nidx, node in enumerate(ent_nodes):
        node_text = node["text"]
        # 找到节点文本的起始位置
        start_pos, node_len = first_index_list(
            sen_tokens[node["sentence_id"]], clean_string(node_text)
        )
        if start_pos != -1:
            final_start_pos = sen_start_idx[node["sentence_id"]] + start_pos
            max_pos = final_start_pos + node_len
            ent_nodes[nidx]["spans"] = tuple([final_start_pos, max_pos])
        else:
            ent_nodes[nidx]["spans"] = tuple([-1, -1])
        if ent_nodes[nidx]["spans"][0] == -1:
            drop_nodes.append(nidx)

    return ent_nodes, all_tokens, drop_nodes, sen_idx_pair

# 设置命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument(
    "--kw_file_dir",
    type=str,
    required=True,
    help="The path of the input dataset with keywords.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="The path to the output dataset with graph.",
)
args = parser.parse_args()

if __name__ == "__main__":
    print("Starting graph construction process...")  # 开始运行的提示
    if args.output_dir == "":  # 如果未指定输出路径，则使用默认路径
        args.output_dir = args.kw_file_dir.replace("_kws.jsonl", "_graph.jsonl")
    
    print("Loading Dataset ...")
    data = read_data(args.kw_file_dir)
    print(f"Loaded {len(data)} instances from dataset.")

    print("Loading Tokenizer ...")
    local_tokenizer_path = "/home/yyh/yyh/PythonProject/Coh-MGT-Detection/preprocess/roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(local_tokenizer_path, do_lower_case=False)
    
    no_node = 0  # 记录没有节点的实例数量

    with open(args.output_dir, "w", encoding="utf8") as outf:
        for idx, line in tqdm(enumerate(data), desc="Processing Instances"):
            kws = line["information"]["keywords"]
            all_info = line["information"]["sentences"]
            nodes, edges, entity_occur, sens, sen2node = build_graph(all_info)

            if not nodes:
                no_node += 1
                continue

            ent_nodes, all_tokens, drop_nodes, sen_idx_pair = generate_rep_mask_based_on_graph(
                nodes, sens, tokenizer
            )

            result = {
                "id": line["id"],
                "graph": {
                    "nodes": ent_nodes,
                    "edges": edges,
                    "entity_occur": entity_occur,
                },
                "sentence_pair": sen_idx_pair,
                "drop_nodes": drop_nodes,
            }

            json.dump(result, outf, ensure_ascii=False)
            outf.write("\n")

    print(f"Finished processing. No node instances: {no_node}")


# import sys
# import os
# import json
# import nltk
# import re
# from transformers import RobertaTokenizer
# from tqdm import tqdm
# import argparse

# # 获取当前脚本的目录
# script_dir = os.path.dirname(os.path.abspath(__file__)) 
# # 获取当前脚本父目录路径
# parent_dir = os.path.dirname(script_dir)  
# # 将父目录路径添加到系统路径，以便导入自定义模块
# sys.path.append(parent_dir)

# from utils.common import *  # 导入自定义的公共函数和模块

# # 构建图的函数
# def build_graph(all_info):
#     """
#     根据所有信息构建图结构，返回节点、边、实体出现次数、句子列表以及句子到节点的映射。

#     :param all_info: 包含句子和关键字的列表
#     :return: 节点列表、边列表、实体出现次数、句子列表、句子到节点的映射
#     """
#     nodes = []            # 用于存储图中的节点信息
#     edges = []            # 用于存储图中的边信息
#     entity_occur = {}     # 用于记录实体的出现次数
#     last_sen_cnt = 0      # 用于计算当前节点在图中的位置
#     sens = [sen["sentence"].replace("\t", " ") for sen in all_info]  # 提取所有句子
#     all_kws = [sen["keywords"]["entity"] for sen in all_info]  # 提取每句中的关键字
#     entropies = [sen["entropy"] for sen in all_info]  # 提取每句的熵值
#     sen2node = []         # 存储句子到节点的映射关系
    
#     print(f"Total sentences: {len(sens)}")  # 输出总句子数

#     for sen_idx, (sen_kws, entropy) in enumerate(zip(all_kws, entropies)):
#         sen_tmp_node = []  # 暂时存储当前句子的节点
#         kws_cnt = 0        # 当前句子的关键字计数
#         # 去除停用词并检查句子是否符合条件
#         sen_kws = list([kw for kw in set(sen_kws) if kw.strip() not in cachedStopWords])
#         if not keep_sen(sen_kws):  # 跳过无关句子
#             sen2node.append([])
#             continue

#         # 遍历当前句子的所有关键字
#         for _, kw in enumerate(sen_kws):
#             # 清理关键字字符串
#             kw = re.sub(r"[^a-zA-Z0-9,.'\`!?]+", " ", kw)
#             words = [
#                 word
#                 for word in nltk.word_tokenize(kw)
#                 if (
#                     word not in cachedStopWords
#                     and word.capitalize() not in cachedStopWords
#                 )
#             ]
#             # 检查关键字是否可以作为节点
#             if keep_node(kw, words):
#                 sen_tmp_node.append(len(nodes))
#                 nodes.append({"text": kw, "words": words, "sentence_id": sen_idx, "entropy": entropy})
#                 if kw not in entity_occur.keys():
#                     entity_occur[kw] = 0
#                 entity_occur[kw] += 1
#                 kws_cnt += 1

#         # 创建句子内部的节点连边关系
#         edges += [
#             tuple([last_sen_cnt + i, last_sen_cnt + i + 1, "inner"])
#             for i in list(range(kws_cnt - 1))
#         ]

#         # 更新计数和句子到节点的映射
#         last_sen_cnt += kws_cnt
#         sen2node.append(sen_tmp_node)

#     # 创建跨句子节点的连边关系
#     for i in range(len(nodes)):
#         for j in range(len(nodes)):
#             if j == i:
#                 continue
#             # 检查两个节点是否是相同的实体
#             if nodes[i]["text"].strip() == nodes[j]["text"].strip():
#                 edges.append(tuple([min(i, j), max(i, j), "inter"]))

#     print(f"Total nodes: {len(nodes)}, Total edges: {len(edges)}")  # 输出总节点数和边数
    
#     if not nodes:
#         return [], [], [], [], []
#     return nodes, list(set(edges)), entity_occur, sens, sen2node

# # 清理字符串的函数
# def clean_string(string):
#     """
#     清理字符串中无效字符，保留字母、数字和部分标点符号。
#     """
#     return re.sub(r"[^a-zA-Z0-9,.'!?]+", "", string)

# # 基于图生成代表掩码的函数
# def generate_rep_mask_based_on_graph(ent_nodes, sens, tokenizer):
#     """
#     基于图生成代表掩码，包含节点、所有标记、需要去除的节点和句子的索引对。

#     :param ent_nodes: 图中的节点
#     :param sens: 句子列表
#     :param tokenizer: 分词器
#     :return: 更新后的节点、所有标记、需要去除的节点、句子索引对
#     """
#     sen_start_idx = [0]  # 记录每句的起始索引
#     sen_idx_pair, sen_tokens, all_tokens, drop_nodes = [], [], [], []

#     for idx, sen in enumerate(sens):
#         # 对句子进行分词
#         sen_token = tokenizer.tokenize(sen)
#         cleaned_sen_token = [clean_string(token) for token in sen_token]
#         sen_tokens.append(cleaned_sen_token)
#         sen_idx_pair.append(
#             tuple([sen_start_idx[-1], sen_start_idx[-1] + len(sen_token)])
#         )
#         sen_start_idx.append(sen_start_idx[-1] + len(sen_token))
#         all_tokens += sen_token

#     for nidx, node in enumerate(ent_nodes):
#         node_text = node["text"]
#         # 找到节点文本的起始位置
#         start_pos, node_len = first_index_list(
#             sen_tokens[node["sentence_id"]], clean_string(node_text)
#         )
#         if start_pos != -1:
#             final_start_pos = sen_start_idx[node["sentence_id"]] + start_pos
#             max_pos = final_start_pos + node_len
#             ent_nodes[nidx]["spans"] = tuple([final_start_pos, max_pos])
#         else:
#             ent_nodes[nidx]["spans"] = tuple([-1, -1])
#         if ent_nodes[nidx]["spans"][0] == -1:
#             drop_nodes.append(nidx)

#     return ent_nodes, all_tokens, drop_nodes, sen_idx_pair

# # 设置命令行参数解析
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--kw_file_dir",
#     type=str,
#     required=True,
#     help="The path of the input dataset with keywords.",
# )
# parser.add_argument(
#     "--output_dir",
#     type=str,
#     default="",
#     help="The path to the output dataset with graph.",
# )
# args = parser.parse_args()

# if __name__ == "__main__":
#     print("Starting graph construction process...")  # 开始运行的提示
#     if args.output_dir == "":  # 如果未指定输出路径，则使用默认路径
#         args.output_dir = args.kw_file_dir.replace("_kws.jsonl", "_graph.jsonl")
    
#     print("Loading Dataset ...")
#     data = read_data(args.kw_file_dir)
#     print(f"Loaded {len(data)} instances from dataset.")

#     print("Loading Tokenizer ...")
#     local_tokenizer_path = "/home/yyh/yyh/PythonProject/Coh-MGT-Detection/preprocess/roberta-base"
#     tokenizer = RobertaTokenizer.from_pretrained(local_tokenizer_path, do_lower_case=False)
    
#     no_node = 0  # 记录没有节点的实例数量

#     with open(args.output_dir, "w", encoding="utf8") as outf:
#         for idx, line in tqdm(enumerate(data), desc="Processing Instances"):
#             kws = line["information"]["keywords"]
#             # 构建图
#             nodes, edges, entity_occur, sens, sen2node = build_graph(kws)
#             if not nodes:
#                 no_node += 1
#             else:
#                 print(f"Instance {idx + 1}: {len(nodes)} nodes, {len(edges)} edges.")
#             # 基于图生成掩码
#             nodes, all_tokens, drop_nodes, sen_idx_pair = generate_rep_mask_based_on_graph(nodes, sens, tokenizer)

#             # 添加图信息到数据
#             line["information"]["graph"] = {
#                 "nodes": nodes,
#                 "edges": edges,
#                 "all_tokens": all_tokens,
#                 "drop_nodes": drop_nodes,
#                 "sentence_to_node_id": sen2node,
#                 "sentence_start_end_idx_pair": sen_idx_pair,
#             }
#             outf.write(json.dumps(line) + "\n")

#     print(f"{no_node} instances are too short that have no graph.")  # 输出没有节点的实例数量

# import sys
# import os
# import json
# import nltk
# import re
# from transformers import RobertaTokenizer
# from tqdm import tqdm
# import argparse

# # 获取当前脚本的目录
# script_dir = os.path.dirname(os.path.abspath(__file__)) 
# # 获取当前脚本父目录路径
# parent_dir = os.path.dirname(script_dir)  
# # 将父目录路径添加到系统路径，以便导入自定义模块
# sys.path.append(parent_dir)

# from utils.common import *  # 导入自定义的公共函数和模块

# # 清理字符串的函数
# def clean_string(string):
#     """
#     清理字符串中无效字符，保留字母、数字和部分标点符号。
#     """
#     return re.sub(r"[^a-zA-Z0-9,.'!?]+", "", string)

# # 基于图生成代表掩码的函数
# def generate_rep_mask_based_on_graph(ent_nodes, sens, tokenizer):
#     """
#     基于图生成代表掩码，包含节点、所有标记、需要去除的节点和句子的索引对。

#     :param ent_nodes: 图中的节点
#     :param sens: 句子列表
#     :param tokenizer: 分词器
#     :return: 更新后的节点、所有标记、需要去除的节点、句子索引对
#     """
#     sen_start_idx = [0]  # 记录每句的起始索引
#     sen_idx_pair, sen_tokens, all_tokens, drop_nodes = [], [], [], []

#     for idx, sen in enumerate(sens):
#         # 对句子进行分词
#         sen_token = tokenizer.tokenize(sen)
#         cleaned_sen_token = [clean_string(token) for token in sen_token]
#         sen_tokens.append(cleaned_sen_token)
#         sen_idx_pair.append(
#             tuple([sen_start_idx[-1], sen_start_idx[-1] + len(sen_token)])
#         )
#         sen_start_idx.append(sen_start_idx[-1] + len(sen_token))
#         all_tokens += sen_token

#     for nidx, node in enumerate(ent_nodes):
#         node_text = node["text"]
#         # 找到节点文本的起始位置
#         start_pos, node_len = first_index_list(
#             sen_tokens[node["sentence_id"]], clean_string(node_text)
#         )
#         if start_pos != -1:
#             final_start_pos = sen_start_idx[node["sentence_id"]] + start_pos
#             max_pos = final_start_pos + node_len
#             ent_nodes[nidx]["spans"] = tuple([final_start_pos, max_pos])
#         else:
#             ent_nodes[nidx]["spans"] = tuple([-1, -1])
#         if ent_nodes[nidx]["spans"][0] == -1:
#             drop_nodes.append(nidx)

#     return ent_nodes, all_tokens, drop_nodes, sen_idx_pair

# # 构建图的函数，增加多尺度和超图的支持
# def build_graph(all_info):
#     """
#     根据所有信息构建多尺度图结构和超图，返回节点、边、超边、实体出现次数、句子列表以及句子到节点的映射。

#     :param all_info: 包含句子和关键字的列表
#     :return: 节点列表、边列表、超边列表、实体出现次数、句子列表、句子到节点的映射
#     """
#     nodes = []            # 用于存储图中的节点信息
#     edges = []            # 用于存储图中的边信息（句子内部和句子之间）
#     hyperedges = []       # 用于存储超边信息
#     entity_occur = {}     # 用于记录实体的出现次数
#     last_sen_cnt = 0      # 用于计算当前节点在图中的位置
#     sens = [sen["sentence"].replace("\t", " ") for sen in all_info]  # 提取所有句子
#     all_kws = [sen["keywords"]["entity"] for sen in all_info]  # 提取每句中的关键字
#     sen2node = []         # 存储句子到节点的映射关系
    
#     print(f"Total sentences: {len(sens)}")  # 输出总句子数

#     # 构建节点和句子内部的边
#     for sen_idx, sen_kws in enumerate(all_kws):
#         sen_tmp_node = []  # 暂时存储当前句子的节点
#         kws_cnt = 0        # 当前句子的关键字计数
#         # 去除停用词并检查句子是否符合条件
#         sen_kws = list([kw for kw in set(sen_kws) if kw.strip() not in cachedStopWords])
#         if not keep_sen(sen_kws):  # 跳过无关句子
#             sen2node.append([])
#             continue

#         # 遍历当前句子的所有关键字
#         for _, kw in enumerate(sen_kws):
#             # 清理关键字字符串
#             kw = re.sub(r"[^a-zA-Z0-9,.'\`!?]+", " ", kw)
#             words = [
#                 word
#                 for word in nltk.word_tokenize(kw)
#                 if (
#                     word not in cachedStopWords
#                     and word.capitalize() not in cachedStopWords
#                 )
#             ]
#             # 检查关键字是否可以作为节点
#             if keep_node(kw, words):
#                 sen_tmp_node.append(len(nodes))
#                 nodes.append({"text": kw, "words": words, "sentence_id": sen_idx})
#                 if kw not in entity_occur.keys():
#                     entity_occur[kw] = 0
#                 entity_occur[kw] += 1
#                 kws_cnt += 1

#         # 创建句子内部的节点连边关系
#         edges += [
#             tuple([last_sen_cnt + i, last_sen_cnt + i + 1, "inner"])
#             for i in list(range(kws_cnt - 1))
#         ]

#         # 更新计数和句子到节点的映射
#         last_sen_cnt += kws_cnt
#         sen2node.append(sen_tmp_node)

#     # 创建跨句子节点的连边关系
#     for i in range(len(nodes)):
#         for j in range(len(nodes)):
#             if j == i:
#                 continue
#             # 检查两个节点是否是相同的实体
#             if nodes[i]["text"].strip() == nodes[j]["text"].strip():
#                 edges.append(tuple([min(i, j), max(i, j), "inter"]))

#     print(f"Total nodes: {len(nodes)}, Total edges: {len(edges)}")  # 输出总节点数和边数

#     # 构建超边（同一实体的所有节点连接成一个超边）
#     entity_to_nodes = {}
#     for idx, node in enumerate(nodes):
#         entity = node["text"].strip()
#         if entity not in entity_to_nodes:
#             entity_to_nodes[entity] = []
#         entity_to_nodes[entity].append(idx)
    
#     for entity, node_indices in entity_to_nodes.items():
#         if len(node_indices) > 1:
#             hyperedges.append({"type": "hyper", "nodes": node_indices, "entity": entity})

#     print(f"Total hyperedges: {len(hyperedges)}")  # 输出超边数

#     if not nodes:
#         return [], [], [], [], [], []
#     return nodes, list(set(edges)), hyperedges, entity_occur, sens, sen2node

# # 基于超图的图卷积方法（需要在模型中实现，以下仅为预处理示例）
# def build_hypergraph(nodes, hyperedges):
#     """
#     构建超图结构。

#     :param nodes: 节点列表
#     :param hyperedges: 超边列表
#     :return: 超图的邻接矩阵或其他表示
#     """
#     # 这里仅返回超边信息，实际的超图卷积在模型中实现
#     return hyperedges

# # 设置命令行参数解析
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--kw_file_dir",
#     type=str,
#     required=True,
#     help="The path of the input dataset with keywords.",
# )
# parser.add_argument(
#     "--output_dir",
#     type=str,
#     default="",
#     help="The path to the output dataset with graph.",
# )
# args = parser.parse_args()

# if __name__ == "__main__":
#     print("Starting graph construction process...")  # 开始运行的提示
#     if args.output_dir == "":  # 如果未指定输出路径，则使用默认路径
#         args.output_dir = args.kw_file_dir.replace("_kws.jsonl", "_graph.jsonl")
    
#     print("Loading Dataset ...")
#     data = read_data(args.kw_file_dir)
#     print(f"Loaded {len(data)} instances from dataset.")

#     print("Loading Tokenizer ...")
#     local_tokenizer_path = "/home/yyh/yyh/PythonProject/Coh-MGT-Detection/preprocess/roberta-base"
#     tokenizer = RobertaTokenizer.from_pretrained(local_tokenizer_path, do_lower_case=False)
    
#     no_node = 0  # 记录没有节点的实例数量

#     with open(args.output_dir, "w", encoding="utf8") as outf:
#         for idx, line in tqdm(enumerate(data), desc="Processing Instances"):
#             kws = line["information"]["keywords"]
#             # 构建多尺度图和超图
#             nodes, edges, hyperedges, entity_occur, sens, sen2node = build_graph(kws)
#             if not nodes:
#                 no_node += 1
#             else:
#                 print(f"Instance {idx + 1}: {len(nodes)} nodes, {len(edges)} edges, {len(hyperedges)} hyperedges.")
#             # 基于图生成掩码
#             nodes, all_tokens, drop_nodes, sen_idx_pair = generate_rep_mask_based_on_graph(nodes, sens, tokenizer)

#             # 添加图信息到数据
#             line["information"]["graph"] = {
#                 "nodes": nodes,
#                 "edges": edges,
#                 "hyperedges": hyperedges,  # 添加超边信息
#                 "all_tokens": all_tokens,
#                 "drop_nodes": drop_nodes,
#                 "sentence_to_node_id": sen2node,
#                 "sentence_start_end_idx_pair": sen_idx_pair,
#             }
#             outf.write(json.dumps(line) + "\n")

#     print(f"{no_node} instances are too short that have no graph.")  # 输出没有节点的实例数量
