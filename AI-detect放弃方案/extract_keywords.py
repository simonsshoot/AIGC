# import json
# import nltk
# from tqdm import tqdm
# from allennlp.predictors.predictor import Predictor
# import re
# import argparse
# import os
# import requests

# print(nltk.data.path)


# # 添加 NLTK 数据路径
# nltk.data.path.append("/home/yyh/nltk_data")

# # 测试加载 punkt 数据包
# try:
#     nltk.sent_tokenize("This is a test sentence.")
#     print("NLTK punkt 数据包加载成功。")
# except Exception as e:
#     print(f"加载 punkt 数据包时出错: {e}")

# # 定义NLTK下载进度显示的函数
# def download_nltk_data(package):
#     """
#     Download NLTK data with progress.
#     :param package: The NLTK package to download
#     """
#     nltk_data_url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/{package}.zip"
#     save_path = os.path.join(nltk.data.find('tokenizers'), f"{package}.zip")

#     print(f"Downloading NLTK package: {package}")
#     if os.path.exists(save_path):
#         print(f"{package} already exists at {save_path}, skipping download.")
#         return save_path

#     try:
#         response = requests.get(nltk_data_url, stream=True)
#         response.raise_for_status()
#         total_size = int(response.headers.get('content-length', 0))
#         block_size = 1024

#         with open(save_path, 'wb') as file, tqdm(
#             total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {package}"
#         ) as progress_bar:
#             for data in response.iter_content(block_size):
#                 progress_bar.update(len(data))
#                 file.write(data)
#         print(f"{package} download completed and saved to {save_path}.")
#     except Exception as e:
#         print(f"Error downloading {package}: {e}")
#         raise e

# # 确保下载必要的NLTK数据
# download_nltk_data('punkt_tab')

# # # 确保下载必要的NLTK数据
# # nltk.download('punkt_tab')

# # Define a function to download the model and display download progress
# def download_model(url, save_path):
#     """
#     Download the model file and display download progress.

#     :param url: The URL to download the model from
#     :param save_path: The path to save the downloaded model
#     """
#     print(f"Starting model download from: {url}")
#     if os.path.exists(save_path):
#         print(f"Model already exists at {save_path}, skipping download.")
#         return save_path

#     try:
#         response = requests.get(url, stream=True)
#         response.raise_for_status()
#         print("Successfully connected to the model download URL.")

#         total_size = int(response.headers.get('content-length', 0))
#         block_size = 1024

#         with open(save_path, 'wb') as file, tqdm(
#             total=total_size, unit='iB', unit_scale=True, desc="Downloading Model"
#         ) as progress_bar:
#             for data in response.iter_content(block_size):
#                 progress_bar.update(len(data))
#                 file.write(data)
#         print(f"Model download completed and saved to {save_path}.")
#     except Exception as e:
#         print(f"Error downloading the model: {e}")
#         raise e

#     return save_path

# # Download the model and initialize the predictor
# model_url = "https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz"
# model_filename = "ner-elmo.2021-02-12.tar.gz"
# model_path = os.path.join(os.path.dirname(__file__), model_filename)

# print("Initializing model download and predictor.")
# download_model(model_url, model_path)

# print("Initializing AllenNLP NER Predictor.")
# try:
#     predictor_ner = Predictor.from_path(
#         model_path,
#         cuda_device=0,  # 使用GPU设备0，设置为-1以使用CPU
#     )
#     print("NER Predictor initialized successfully.")
# except Exception as e:
#     print(f"Error initializing NER Predictor: {e}")
#     raise e

# def load_jsonl(filename):
#     """
#     Read a JSONL file, parsing each line as a JSON object.

#     :param filename: Path to the JSONL file
#     :return: Generator that yields parsed JSON objects line by line
#     """
#     print(f"Starting to load JSONL file: {filename}")
#     try:
#         with open(filename, "r") as file:
#             for line in file:
#                 yield json.loads(line.strip())
#         print("JSONL file loaded successfully.")
#     except Exception as e:
#         print(f"Error loading JSONL file: {e}")
#         raise e

# def found_key_words(claims):
#     """
#     Use the NER predictor to perform batch predictions on sentences and extract keywords.

#     :param claims: List of sentences
#     :return: List of extracted keywords, each containing keywords and the corresponding sentence
#     """
#     print(f"Starting keyword extraction for {len(claims)} sentences.")
#     try:
#         all_ent_res = predictor_ner.predict_batch_json(
#             inputs=[{"sentence": text} for text in claims]
#         )
#     except Exception as e:
#         print(f"Error during NER prediction: {e}")
#         raise e

#     all_keywords = []
#     for i in range(len(claims)):
#         claim = claims[i]
#         ent_res = all_ent_res[i]

#         key_words = {"noun": [], "claim": claim, "subject": [], "entity": []}
#         all_ents = extract_entity_allennlp(ent_res["words"], ent_res["tags"])
#         key_words["entity"].extend(all_ents)
#         key_words = {"keywords": key_words, "sentence": claim}

#         all_keywords.append(key_words)

#         if (i + 1) % 100 == 0:
#             print(f"Processed {i + 1} sentences.")

#     print("Keyword extraction completed.")
#     return all_keywords

# def analyze_document(doc):
#     """
#     Analyze a document by splitting it into sentences and extracting keywords.

#     :param doc: Document content as a string
#     :return: List containing keywords
#     """
#     if not doc:
#         print("Document is empty. Skipping analysis.")
#         return []

#     print("Starting document analysis.")
#     sens = nltk.sent_tokenize(doc)
#     resplit_sens = []
#     for sen in sens:
#         resplit_sens += [s.strip() for s in sen.split("\n") if s.strip() != ""]
#     sens = resplit_sens
#     print(f"After splitting, obtained {len(sens)} sentences.")

#     try:
#         all_keywords = found_key_words(sens)
#     except Exception as e:
#         print(f"Error during document analysis: {e}")
#         all_keywords = []

#     print("Document analysis completed.")
#     return all_keywords

# def extract_entity_allennlp(words, tags):
#     """
#     Extract entities based on AllenNLP's NER tags.

#     :param words: List of words in the sentence
#     :param tags: List of NER tags corresponding to each word
#     :return: List of extracted entities
#     """
#     assert len(words) == len(tags), "Number of words and tags do not match."
#     e_list_cache = []
#     e_list_final = []
#     start_index = None

#     for i in range(len(tags)):
#         tag = tags[i]
#         if tag != "O":
#             if tag.startswith("B"):
#                 start_index = i
#                 e_list_cache = [re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[i])]
#             elif tag.startswith("I") and start_index is not None:
#                 e_list_cache.append(re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[i]))
#             elif tag.startswith("L") and start_index is not None:
#                 e_list_cache.append(re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[i]))
#                 entity = " ".join(e_list_cache)
#                 e_list_final.append(entity)
#                 e_list_cache = []
#                 start_index = None
#             elif tag.startswith("U"):
#                 cword = re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[i])
#                 e_list_final.append(cword)
#         else:
#             start_index = None

#     return e_list_final

# # Set up command-line argument parsing
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--raw_dir", 
#     type=str, 
#     required=True, 
#     help="The path of raw dataset."
# )
# parser.add_argument(
#     "--output_dir",
#     type=str,
#     default="",
#     help="The path to output dataset with keywords.",
# )
# args = parser.parse_args()

# if __name__ == "__main__":
#     print("Program started.")
    
#     if args.output_dir == "":
#         args.output_dir = args.raw_dir.replace(".jsonl", "_kws.jsonl")
    
#     print(f"Using output path: {args.output_dir}")

#     print("Loading the raw JSONL file.")
#     try:
#         file = list(load_jsonl(args.raw_dir))
#         print(f"Successfully loaded {len(file)} records.")
#     except Exception as e:
#         print(f"Error loading the raw file: {e}")
#         exit(1)

#     print(f"Starting to process and write to output file: {args.output_dir}")
#     try:
#         with open(args.output_dir, "w", encoding="utf-8") as out_file:
#             for idx, line in enumerate(tqdm(file, desc="Processing Documents")):
#                 doc = line.get("article", "")
#                 if not doc:
#                     print(f"Record {idx} is missing the 'article' field, skipping.")
#                     continue

#                 keywords = analyze_document(doc)
#                 line["information"] = {"keywords": keywords}

#                 out_file.write(json.dumps(line, ensure_ascii=False) + "\n")

#                 if (idx + 1) % 100 == 0:
#                     print(f"Processed {idx + 1} records.")
        
#         print("All records processed successfully.")
    
#     except Exception as e:
#         print(f"Error processing documents or writing to the output file: {e}")
#         exit(1)

#     print("Program finished successfully.")
# import json
# import nltk
# from tqdm import tqdm
# from allennlp.predictors.predictor import Predictor
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch
# import re
# import argparse
# import os
# import requests
# import numpy as np

# print(nltk.data.path)

# # 添加 NLTK 数据路径
# nltk.data.path.append("/home/yyh/nltk_data")

# # 测试加载 punkt 数据包
# try:
#     nltk.sent_tokenize("This is a test sentence.")
#     print("NLTK punkt 数据包加载成功。")
# except Exception as e:
#     print(f"加载 punkt 数据包时出错: {e}")

# # 定义NLTK下载进度显示的函数
# def download_nltk_data(package):
#     """
#     Download NLTK data with progress.
#     :param package: The NLTK package to download
#     """
#     nltk_data_url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/{package}.zip"
#     save_path = os.path.join(nltk.data.find('tokenizers'), f"{package}.zip")

#     print(f"Downloading NLTK package: {package}")
#     if os.path.exists(save_path):
#         print(f"{package} already exists at {save_path}, skipping download.")
#         return save_path

#     try:
#         response = requests.get(nltk_data_url, stream=True)
#         response.raise_for_status()
#         total_size = int(response.headers.get('content-length', 0))
#         block_size = 1024

#         with open(save_path, 'wb') as file, tqdm(
#             total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {package}"
#         ) as progress_bar:
#             for data in response.iter_content(block_size):
#                 progress_bar.update(len(data))
#                 file.write(data)
#         print(f"{package} download completed and saved to {save_path}.")
#     except Exception as e:
#         print(f"Error downloading {package}: {e}")
#         raise e

# # 确保下载必要的NLTK数据
# download_nltk_data('punkt_tab')

# # Define a function to download the model and display download progress
# def download_model(url, save_path):
#     """
#     Download the model file and display download progress.

#     :param url: The URL to download the model from
#     :param save_path: The path to save the downloaded model
#     """
#     print(f"Starting model download from: {url}")
#     if os.path.exists(save_path):
#         print(f"Model already exists at {save_path}, skipping download.")
#         return save_path

#     try:
#         response = requests.get(url, stream=True)
#         response.raise_for_status()
#         print("Successfully connected to the model download URL.")

#         total_size = int(response.headers.get('content-length', 0))
#         block_size = 1024

#         with open(save_path, 'wb') as file, tqdm(
#             total=total_size, unit='iB', unit_scale=True, desc="Downloading Model"
#         ) as progress_bar:
#             for data in response.iter_content(block_size):
#                 progress_bar.update(len(data))
#                 file.write(data)
#         print(f"Model download completed and saved to {save_path}.")
#     except Exception as e:
#         print(f"Error downloading the model: {e}")
#         raise e

#     return save_path

# # Download the model and initialize the predictor
# model_url = "https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz"
# model_filename = "ner-elmo.2021-02-12.tar.gz"
# model_path = os.path.join(os.path.dirname(__file__), model_filename)

# print("Initializing model download and predictor.")
# download_model(model_url, model_path)

# print("Initializing AllenNLP NER Predictor.")
# try:
#     predictor_ner = Predictor.from_path(
#         model_path,
#         cuda_device=0,  # 使用GPU设备0，设置为-1以使用CPU
#     )
#     print("NER Predictor initialized successfully.")
# except Exception as e:
#     print(f"Error initializing NER Predictor: {e}")
#     raise e

# # 加载 GPT-2 模型和分词器
# model_name = "gpt2"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Loading GPT-2 model and tokenizer...")
# model_gpt2 = GPT2LMHeadModel.from_pretrained('/data/Content Moderation/gpt2').to(device)
# tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('/data/Content Moderation/gpt2')
# print("GPT-2 model and tokenizer loaded successfully.")

# def load_jsonl(filename):
#     """
#     Read a JSONL file, parsing each line as a JSON object.

#     :param filename: Path to the JSONL file
#     :return: Generator that yields parsed JSON objects line by line
#     """
#     print(f"Starting to load JSONL file: {filename}")
#     try:
#         with open(filename, "r") as file:
#             for line in file:
#                 yield json.loads(line.strip())
#         print("JSONL file loaded successfully.")
#     except Exception as e:
#         print(f"Error loading JSONL file: {e}")
#         raise e

# def calculate_entropy(logits):
#     """
#     Calculate the entropy of the logits.
#     :param logits: The logits output from the model
#     :return: The entropy value
#     """
#     probabilities = torch.softmax(logits, dim=-1)
#     entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)
#     return entropy.mean().item()

# def found_key_words(claims):
#     """
#     Use the NER predictor to perform batch predictions on sentences and extract keywords.

#     :param claims: List of sentences
#     :return: List of extracted keywords, each containing keywords and the corresponding sentence
#     """
#     print(f"Starting keyword extraction for {len(claims)} sentences.")
#     try:
#         all_ent_res = predictor_ner.predict_batch_json(
#             inputs=[{"sentence": text} for text in claims]
#         )
#     except Exception as e:
#         print(f"Error during NER prediction: {e}")
#         raise e

#     all_keywords = []
#     for i in range(len(claims)):
#         claim = claims[i]
#         ent_res = all_ent_res[i]

#         key_words = {"noun": [], "claim": claim, "subject": [], "entity": []}
#         all_ents = extract_entity_allennlp(ent_res["words"], ent_res["tags"])
#         key_words["entity"].extend(all_ents)
#         key_words = {"keywords": key_words, "sentence": claim}

#         # Calculate entropy using GPT-2 model
#         inputs = tokenizer_gpt2(claim, return_tensors="pt").to(device)
#         with torch.no_grad():
#             outputs = model_gpt2(**inputs)
#             logits = outputs.logits
#             avg_entropy = calculate_entropy(logits)
#         key_words["entropy"] = avg_entropy

#         all_keywords.append(key_words)

#         if (i + 1) % 100 == 0:
#             print(f"Processed {i + 1} sentences.")

#     print("Keyword extraction completed.")
#     return all_keywords

# def analyze_document(doc):
#     """
#     Analyze a document by splitting it into sentences and extracting keywords.

#     :param doc: Document content as a string
#     :return: List containing keywords
#     """
#     if not doc:
#         print("Document is empty. Skipping analysis.")
#         return []

#     print("Starting document analysis.")
#     sens = nltk.sent_tokenize(doc)
#     resplit_sens = []
#     for sen in sens:
#         resplit_sens += [s.strip() for s in sen.split("\n") if s.strip() != ""]
#     sens = resplit_sens
#     print(f"After splitting, obtained {len(sens)} sentences.")

#     try:
#         all_keywords = found_key_words(sens)
#     except Exception as e:
#         print(f"Error during document analysis: {e}")
#         all_keywords = []

#     print("Document analysis completed.")
#     return all_keywords

# def extract_entity_allennlp(words, tags):
#     """
#     Extract entities based on AllenNLP's NER tags.

#     :param words: List of words in the sentence
#     :param tags: List of NER tags corresponding to each word
#     :return: List of extracted entities
#     """
#     assert len(words) == len(tags), "Number of words and tags do not match."
#     e_list_cache = []
#     e_list_final = []
#     start_index = None

#     for i in range(len(tags)):
#         tag = tags[i]
#         if tag != "O":
#             if tag.startswith("B"):
#                 start_index = i
#                 e_list_cache = [re.sub(r"[^a-zA-Z0-9,.'-/!?]+", "", words[i])]
#             elif tag.startswith("I") and start_index is not None:
#                 e_list_cache.append(re.sub(r"[^a-zA-Z0-9,.'-/!?]+", "", words[i]))
#             elif tag.startswith("L") and start_index is not None:
#                 e_list_cache.append(re.sub(r"[^a-zA-Z0-9,.'-/!?]+", "", words[i]))
#                 entity = " ".join(e_list_cache)
#                 e_list_final.append(entity)
#                 e_list_cache = []
#                 start_index = None
#             elif tag.startswith("U"):
#                 cword = re.sub(r"[^a-zA-Z0-9,.'-/!?]+", "", words[i])
#                 e_list_final.append(cword)
#         else:
#             start_index = None

#     return e_list_final

# # Set up command-line argument parsing
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--raw_dir", 
#     type=str, 
#     required=True, 
#     help="The path of raw dataset."
# )
# parser.add_argument(
#     "--output_dir",
#     type=str,
#     default="",
#     help="The path to output dataset with keywords.",
# )
# args = parser.parse_args()

# if __name__ == "__main__":
#     print("Program started.")
    
#     if args.output_dir == "":
#         args.output_dir = args.raw_dir.replace(".jsonl", "_kws.jsonl")
    
#     print(f"Using output path: {args.output_dir}")

#     print("Loading the raw JSONL file.")
#     try:
#         file = list(load_jsonl(args.raw_dir))
#         print(f"Successfully loaded {len(file)} records.")
#     except Exception as e:
#         print(f"Error loading the raw file: {e}")
#         exit(1)

#     print(f"Starting to process and write to output file: {args.output_dir}")
#     try:
#         with open(args.output_dir, "w", encoding="utf-8") as out_file:
#             for idx, line in enumerate(tqdm(file, desc="Processing Documents")):
#                 doc = line.get("article", "")
#                 if not doc:
#                     print(f"Record {idx} is missing the 'article' field, skipping.")
#                     continue

#                 keywords = analyze_document(doc)
#                 line["information"] = {"keywords": keywords, "entropy": [kw["entropy"] for kw in keywords]}

#                 out_file.write(json.dumps(line, ensure_ascii=False) + "\n")

#                 if (idx + 1) % 100 == 0:
#                     print(f"Processed {idx + 1} records.")
        
#         print("All records processed successfully.")
    
#     except Exception as e:
#         print(f"Error processing documents or writing to the output file: {e}")
#         exit(1)

#     print("Program finished successfully.")
import json
import nltk
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import re
import argparse
import os
import requests
from textblob import TextBlob  # 导入 TextBlob

print(nltk.data.path)

# 添加 NLTK 数据路径
nltk.data.path.append("/home/yyh/nltk_data")

# 测试加载 punkt 数据包
try:
    nltk.sent_tokenize("This is a test sentence.")
    print("NLTK punkt 数据包加载成功。")
except Exception as e:
    print(f"加载 punkt 数据包时出错: {e}")

# 定义NLTK下载进度显示的函数
def download_nltk_data(package):
    """
    Download NLTK data with progress.
    :param package: The NLTK package to download
    """
    nltk_data_url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/{package}.zip"
    save_path = os.path.join(nltk.data.find('tokenizers'), f"{package}.zip")

    print(f"Downloading NLTK package: {package}")
    if os.path.exists(save_path):
        print(f"{package} already exists at {save_path}, skipping download.")
        return save_path

    try:
        response = requests.get(nltk_data_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(save_path, 'wb') as file, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {package}"
        ) as progress_bar:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        print(f"{package} download completed and saved to {save_path}.")
    except Exception as e:
        print(f"Error downloading {package}: {e}")
        raise e

# 确保下载必要的NLTK数据
download_nltk_data('punkt_tab')

# Define a function to download the model and display download progress
def download_model(url, save_path):
    """
    Download the model file and display download progress.

    :param url: The URL to download the model from
    :param save_path: The path to save the downloaded model
    """
    print(f"Starting model download from: {url}")
    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}, skipping download.")
        return save_path

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        print("Successfully connected to the model download URL.")

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(save_path, 'wb') as file, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc="Downloading Model"
        ) as progress_bar:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        print(f"Model download completed and saved to {save_path}.")
    except Exception as e:
        print(f"Error downloading the model: {e}")
        raise e

    return save_path

# Download the model and initialize the predictor
model_url = "https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz"
model_filename = "ner-elmo.2021-02-12.tar.gz"
model_path = os.path.join(os.path.dirname(__file__), model_filename)

print("Initializing model download and predictor.")
download_model(model_url, model_path)

print("Initializing AllenNLP NER Predictor.")
try:
    predictor_ner = Predictor.from_path(
        model_path,
        cuda_device=0,  # 使用GPU设备0，设置为-1以使用CPU
    )
    print("NER Predictor initialized successfully.")
except Exception as e:
    print(f"Error initializing NER Predictor: {e}")
    raise e

def load_jsonl(filename):
    """
    Read a JSONL file, parsing each line as a JSON object.

    :param filename: Path to the JSONL file
    :return: Generator that yields parsed JSON objects line by line
    """
    print(f"Starting to load JSONL file: {filename}")
    try:
        with open(filename, "r") as file:
            for line in file:
                yield json.loads(line.strip())
        print("JSONL file loaded successfully.")
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        raise e

def found_key_words(claims):
    """
    Use the NER predictor to perform batch predictions on sentences and extract keywords.

    :param claims: List of sentences
    :return: List of extracted keywords, each containing keywords and the corresponding sentence
    """
    print(f"Starting keyword extraction for {len(claims)} sentences.")
    try:
        all_ent_res = predictor_ner.predict_batch_json(
            inputs=[{"sentence": text} for text in claims]
        )
    except Exception as e:
        print(f"Error during NER prediction: {e}")
        raise e

    all_keywords = []
    for i in range(len(claims)):
        claim = claims[i]
        ent_res = all_ent_res[i]

        key_words = {"noun": [], "claim": claim, "subject": [], "entity": []}
        all_ents = extract_entity_allennlp(ent_res["words"], ent_res["tags"])
        key_words["entity"].extend(all_ents)
        key_words = {"keywords": key_words, "sentence": claim}

        all_keywords.append(key_words)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} sentences.")

    print("Keyword extraction completed.")
    return all_keywords

def correct_spelling(text, proper_nouns=None):
    """
    Correct spelling errors in the input text using TextBlob.

    :param text: The input text to correct
    :param proper_nouns: A set of proper nouns to exclude from correction
    :return: The corrected text
    """
    blob = TextBlob(text)
    corrected_blob = blob.correct()

    # 如果有专有名词，需要将它们恢复到原始形式
    if proper_nouns:
        corrected_text = str(corrected_blob)
        for noun in proper_nouns:
            # 使用正则表达式匹配独立的单词，忽略大小写
            pattern = r'\b' + re.escape(noun) + r'\b'
            corrected_text = re.sub(pattern, noun, corrected_text, flags=re.IGNORECASE)
        return corrected_text
    else:
        return str(corrected_blob)

def analyze_document(doc, proper_nouns=None):
    """
    Analyze a document by correcting spelling, splitting into sentences, and extracting keywords.

    :param doc: Document content as a string
    :param proper_nouns: A set of proper nouns to exclude from correction
    :return: List containing keywords
    """
    if not doc:
        print("Document is empty. Skipping analysis.")
        return []

    print("Starting document analysis.")

    # 拼写纠错
    corrected_doc = correct_spelling(doc, proper_nouns)
    print("Spelling correction completed.")

    # 句子分割
    sens = nltk.sent_tokenize(corrected_doc)
    resplit_sens = []
    for sen in sens:
        resplit_sens += [s.strip() for s in sen.split("\n") if s.strip() != ""]
    sens = resplit_sens
    print(f"After splitting, obtained {len(sens)} sentences.")

    print("Starting keyword extraction.")
    try:
        all_keywords = found_key_words(sens)
    except Exception as e:
        print(f"Error during document analysis: {e}")
        all_keywords = []

    print("Document analysis completed.")
    return all_keywords

def extract_entity_allennlp(words, tags):
    """
    Extract entities based on AllenNLP's NER tags.

    :param words: List of words in the sentence
    :param tags: List of NER tags corresponding to each word
    :return: List of extracted entities
    """
    assert len(words) == len(tags), "Number of words and tags do not match."
    e_list_cache = []
    e_list_final = []
    start_index = None

    for i in range(len(tags)):
        tag = tags[i]
        if tag != "O":
            if tag.startswith("B"):
                start_index = i
                e_list_cache = [re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[i])]
            elif tag.startswith("I") and start_index is not None:
                e_list_cache.append(re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[i]))
            elif tag.startswith("L") and start_index is not None:
                e_list_cache.append(re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[i]))
                entity = " ".join(e_list_cache)
                e_list_final.append(entity)
                e_list_cache = []
                start_index = None
            elif tag.startswith("U"):
                cword = re.sub(r"[^a-zA-Z0-9,.\'-/!?]+", "", words[i])
                e_list_final.append(cword)
        else:
            start_index = None

    return e_list_final

# Set up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw_dir", 
    type=str, 
    required=True, 
    help="The path of raw dataset."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="The path to output dataset with keywords.",
)
args = parser.parse_args()

if __name__ == "__main__":
    print("Program started.")
    
    if args.output_dir == "":
        args.output_dir = args.raw_dir.replace(".jsonl", "_kws.jsonl")
    
    print(f"Using output path: {args.output_dir}")

    # 加载专有名词词典（可选）
    proper_nouns = set()
    proper_nouns_file = "proper_nouns.txt"  # 假设您有一个包含专有名词的文本文件
    if os.path.exists(proper_nouns_file):
        with open(proper_nouns_file, "r") as pn_file:
            for line in pn_file:
                proper_nouns.add(line.strip())
        print(f"Loaded {len(proper_nouns)} proper nouns.")
    else:
        print("Proper nouns file not found, proceeding without it.")

    print("Loading the raw JSONL file.")
    try:
        file = list(load_jsonl(args.raw_dir))
        print(f"Successfully loaded {len(file)} records.")
    except Exception as e:
        print(f"Error loading the raw file: {e}")
        exit(1)

    print(f"Starting to process and write to output file: {args.output_dir}")
    try:
        with open(args.output_dir, "w", encoding="utf8") as outf:
            for idx, line in enumerate(tqdm(file, desc="Processing Documents")):
                doc = line.get("article", "")
                if not doc:
                    print(f"Record {idx} is missing the 'article' field, skipping.")
                    continue

                keywords = analyze_document(doc, proper_nouns)
                line["information"] = {"keywords": keywords}

                outf.write(json.dumps(line, ensure_ascii=False) + "\n")

                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1} records.")
        
        print("All records processed successfully.")
    
    except Exception as e:
        print(f"Error processing documents or writing to the output file: {e}")
        exit(1)

    print("Program finished successfully.")


