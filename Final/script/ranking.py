from json import load
import pickle
from gensim import models
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
from nltk.util import pr
import spacy
from spacy import vocab
from tqdm import tqdm
from gensim import corpora
from gensim.models import Word2Vec
import numpy as np

def loadText(file):
    with open(file, 'rb') as fpick:
        pmDoc = pickle.load(fpick)
    fulltext = []
    for title, article in pmDoc.items():
        context = title+" "
        for label, text in article.items():
            context += text
        fulltext.append(context)
    return fulltext

def process(fulltext):
    sentences = [sent_tokenize(article) for article in fulltext]
    fulltext = [re.sub(r'[^a-z0-9|^-|^\']', ' ', text.lower()) for text in fulltext]
    fulltext = [word_tokenize(article) for article in fulltext]
    stop = stopwords.words('english')
    cleantext = []
    nlp = spacy.load('en_core_web_sm')
    for article in tqdm(fulltext):
        cleanwords = [word for word in article if word not in stop]
        lemmatext = [token.lemma_ for token in list(nlp(' '.join(cleanwords)))]
        cleantext.append(lemmatext) 
    return cleantext

def load_art(file):
    with open(file, 'rb') as fpick:
        pmDoc = pickle.load(fpick)
    fulltext = {}
    for title, article in pmDoc.items():
        context = ""
        for label, text in article.items():
            context += text
        fulltext[title] = context
    return fulltext

def process_sent(article):
    article = sent_tokenize(article)
    fulltext = [re.sub(r'[^a-z0-9|^-|^\']', ' ', text.lower()) for text in article]
    fulltext = [word_tokenize(article) for article in fulltext]
    stop = stopwords.words('english')
    cleantext = []
    nlp = spacy.load('en_core_web_sm')
    for text in fulltext:
        cleanwords = [word for word in text if word not in stop]
        lemmatext = [token.lemma_ for token in list(nlp(' '.join(cleanwords)))]
        cleantext.append(lemmatext)
    return cleantext

def get_lemma(fulltext):
    sentences = [sent_tokenize(article) for article in fulltext]
    fulltext = [re.sub(r'[^a-z0-9|^-|^\']', ' ', text.lower()) for text in fulltext]
    lemmadoc = []
    nlp = spacy.load('en_core_web_sm')
    for article in tqdm(fulltext):
        lemmatext = [token.lemma_ for token in list(nlp(article))]
        lemmadoc.append(lemmatext) 
    return lemmadoc

if __name__=='__main__':
    docPath  = '../Data'
    # articles = load_art(f'{docPath}/Full.pkl')
    # cleanArt = {}
    # for title, text in tqdm(articles.items()):
    #     cleantext = process_sent(text)
    #     cleanArt[title] = cleantext
    # print(cleantext)

    # model = Word2Vec.load('word2vec.model')
    # for title, text in tqdm(cleantext.items()):
    #     embedding = [model.wv.word_vec(word) for word in text]



    fulltext = loadText(f'{docPath}/Full.pkl')
    lemmas = get_lemma(fulltext) 
    # # preprocess
    # cleantext = process(fulltext)
    # #save processed data
    # with open(f'{docPath}/Full_processed.pkl', 'wb')as fpick:
    #     pickle.dump(cleantext, fpick)
    # # save processed data
    with open(f'{docPath}/Full_processed.pkl', 'rb')as fpick:
        cleantext = pickle.load(fpick)

    # id2word = corpora.Dictionary(cleantext)
    # id2word.save_as_text(f"{docPath}/dictionary")
    # corpus = [' '.join(line) for line in cleantext]   # Term Document Frequency
    # corpus = [id2word.doc2bow(text) for text in cleantext]
    # print(corpus)
    # model = Word2Vec(cleantext, vector_size=300, window=5, min_count=1, seed=1, workers=4, sg=1)
    
    model = Word2Vec.load('word2vec.model')
    
    # lemmadoc = get_lemma(fulltext)
    # with open(f'{docPath}/Full_lemma.pkl', 'wb')as fpick:
    #     pickle.dump(lemmadoc, fpick)
    with open(f'{docPath}/Full_lemma.pkl', 'rb')as fpick:
        lemmadoc = pickle.load(fpick)

    # docVec = []
    # voc = model.wv.index_to_key
    # for lemma in tqdm(lemmadoc):
    #     embedding = []
    #     for word in lemma:
    #         if word in voc:
    #             embedding.append(model.wv.word_vec(word))
    #         else:
    #             embedding.append(np.zeros(300))
        
    #     docVec.append(np.mean(embedding, axis=0))
    
    # with open(f'{docPath}/Doc2Vec.pkl', 'wb') as fpick:
    #     pickle.dump(docVec, fpick)
    
    with open(f'{docPath}/Doc2Vec.pkl', 'rb') as fpick:
        docVec = pickle.load(fpick)
    # print(docVec)
    # # print(docVec[-1])

    # word = model.wv.word_vec('disease')
    # # print(word)
    # result = np.dot(word, docVec[-1])/(np.linalg.norm(word)*np.linalg.norm(docVec[-1]))
    # print(result)

    # word2doc = {}
    # for word in tqdm(voc):
    #     word2doc[word] = []
    #     wordVec = model.wv.word_vec(word)
    #     for doc in docVec:
    #         similar = np.dot(wordVec, doc)/(np.linalg.norm(wordVec)*np.linalg.norm(doc))
    #         word2doc[word].append(similar)
    # print(word2doc)

    wordVec = model.wv.word_vec('fever')
    rank = []
    for doc in docVec:
        similar = np.dot(wordVec, doc)/(np.linalg.norm(wordVec)*np.linalg.norm(doc))
        rank.append(similar)
    # print(rank.sort()[:10])
    rank_sim = sorted(rank, reverse=True)
    rank_doc = np.argsort(-1*np.array(rank))
    for doc in rank_doc[:10]:
        print(fulltext[doc])

    


