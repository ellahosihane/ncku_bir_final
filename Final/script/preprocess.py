import pickle
import os
import string
from typing import Counter
import nltk
from nltk.corpus.reader import tagged
from nltk.util import pr
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from gensim import corpora
from gensim.corpora import Dictionary
import gensim
from gensim.models import CoherenceModel
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from gensim.models import Word2Vec
import pandas as pd


if __name__=='__main__':
    # load data
    # docPath  = '../Data/Full.pkl'
    # with open(docPath, 'rb') as fpick:
    #     pmDoc = pickle.load(fpick)
    # fulltitle = []
    # for title, article in pmDoc.items():
    #     fulltitle.append(title)
    # print(fulltitle)
    # with open('../Data/FullTitle.pkl', 'wb')as fpick:
    #     pickle.dump(fulltitle, fpick)
    # load data
    docPath  = 'Final/Data/Full.pkl'
    with open(docPath, 'rb') as fpick:
        pmDoc = pickle.load(fpick)
    fulltext = []
    for title, article in pmDoc.items():
        context = title+" "
        for label, text in article.items():
            context += text
        fulltext.append(context)

    # 清理
    fulltext = ' '.join(fulltext)
    sentences = sent_tokenize(fulltext)
    sentences = [re.sub(r'[^a-z0-9|^-]', ' ', sent.lower()) for sent in sentences]
    stop = stopwords.words('english')
    cleantext = []
    for sent in tqdm(sentences):
        words = [word for word in sent.split()]
        cleanwords = [word for word in words if word not in stop]
        cleantext.append(cleanwords)
    
    # print(cleantext)
    id2word = corpora.Dictionary(cleantext)
    # print(id2word)
    # print(id2word)
    id2word.save_as_text("dictionary")                   # save dict
    corpus = [' '.join(line) for line in cleantext]   # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in cleantext]
    # print(corpus)
    # word2vec训练词向量
    model = Word2Vec(cleantext, vector_size=200, window=5, min_count=1, seed=1, workers=4)
    model.save('word2vec.model')
    print(model.wv.most_similar('diagnosis'))
    # 加载模型得出词向量
    # vocab = list(model.wv.key_to_index)
    # wv = model.wv[vocab]  # 所有分词对应词向量
    # tsne = TSNE(n_components=2)
    # X_tsne = tsne.fit_transform(wv)
    # df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(df['x'], df['y'])
    # for word, pos in df.iterrows():
    #     ax.annotate(word, pos)
    # plt.savefig('test.png')

