import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

# tokenizer関数をlambdaではなく普通に定義
def custom_tokenizer(text):
    return text.split(", ")

# キャプションファイルのフォルダ
caption_dir = "./img"

# img_xxxxx.txt からテキストを読み込む
file_to_text = {}
for fname in os.listdir(caption_dir):
    if fname.endswith(".txt"):
        with open(os.path.join(caption_dir, fname), "r", encoding="utf-8") as f:
            text = f.read().strip()
            file_to_text[fname.replace(".txt", ".png")] = text

# TF-IDFベクトル化
vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False)
corpus = list(file_to_text.values())
tfidf_matrix = vectorizer.fit_transform(corpus)

# 保存
with open("caption_vectors.pkl", "wb") as f:
    pickle.dump((list(file_to_text.keys()), tfidf_matrix, vectorizer), f)
