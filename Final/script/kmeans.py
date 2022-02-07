from cProfile import label
from this import d
from gensim.models import Word2Vec
import nltk
from nltk.cluster import KMeansClusterer
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import spacy
from tqdm import tqdm
import re
from nltk.corpus import stopwords
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

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

if __name__ == '__main__':
####################### TF-IDF ################################
    docPath  = '../Data'
    with open(f'{docPath}/Full_processed.pkl', 'rb')as fpick:
        texts = pickle.load(fpick)
    with open(f'{docPath}/FullTitle.pkl', 'rb')as fpick:
        titles= pickle.load(fpick)
    id2word = corpora.Dictionary(texts)     # Create Dictionary
    id2word.save_as_text("dictionary")                   # save dict
    corpus = [id2word.doc2bow(text) for text in texts]   # Term Document Frequency
    corpus = [' '.join(line) for line in texts]
    vectorizer_tfidf = TfidfVectorizer(max_df=0.5, min_df=2, use_idf=True)
    X_tfidf = vectorizer_tfidf.fit_transform(corpus)
    svd = TruncatedSVD(100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_tfidf_lsa = lsa.fit_transform(X_tfidf)
    print(X_tfidf_lsa.shape)
    # kclusterer = KMeansClusterer(10, distance=nltk.cluster.util.cosine_distance, repeats=25)
    # assigned_clusters = kclusterer.cluster(X_tfidf_lsa, assign_clusters=True)
    km = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1, verbose=False)
    km_X_tfidf_lsa = km.fit(X_tfidf_lsa)
    print("Top terms per cluster:")
    original_space_centroids = svd.inverse_transform(km_X_tfidf_lsa.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer_tfidf.get_feature_names()
    # pickle.dump(km, open("kmeans.pkl", "wb"))

    with open("kmeans.pkl", "rb")as fpick:
        km = pickle.load(fpick)

    kcluster = km.predict(X_tfidf_lsa)
    # assigned_clusters = kclusterer.cluster(X_tfidf_lsa, assign_clusters=True)
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y=model.fit_transform(X_tfidf_lsa)
    kmeans_dict = {}
    for i, clu in enumerate(kcluster):
        if clu in kmeans_dict:
            kmeans_dict[clu].append([Y[i][0], Y[i][1], titles[i]])
        else:
            kmeans_dict[clu] = []
            kmeans_dict[clu].append([Y[i][0], Y[i][1], titles[i]])
    
    # with open(f'{docPath}/Kmeans_result.pkl', 'wb') as fpick:
    #     pickle.dump(kmeans_dict, fpick)
    # print(kmeans_dict[0])

    
    distortions = []
    K = range(1,12)

    for cluster_size in K:
        kmeans = KMeans(n_clusters=cluster_size, init='k-means++')
        kmeans = kmeans.fit(X_tfidf_lsa)
        distortions.append(kmeans.inertia_)
        
    df = pd.DataFrame({'Clusters': K, 'Distortions': distortions})
    fig = (px.line(df, x='Clusters', y='Distortions', template='seaborn')).update_traces(mode='lines+markers')
    fig.write_image("SSE.png")

    # for j in range(len(titles)):
    #     print ("%s %s" % (kcluster[j],  titles[j]))

    # labels = []
    # for i in range(5):
    #     print("Cluster %d:" % (i+1), end='')
    #     clu = f"Cluster {i+1}: "
    #     for ind in order_centroids[i, :10]:
    #         print(' %s' % terms[ind], end='')
    #         clu += f'{terms[ind]}, '
    #     print()
    #     labels.append(clu[:-2])
    # clus = [labels[i] for i in kcluster]
    # fig = px.scatter(Y, x = Y[:,0], y=Y[:,1],
    #             color = clus, opacity = 0.8)
    # fig.show()
####################### TF-IDF ################################

####################### Word2Vec ################################
    # docPath  = '../Data'
    # with open(f'{docPath}/Full_processed.pkl', 'rb')as fpick:
    #     cleantext = pickle.load(fpick)
    # model = Word2Vec.load('word2vec.model')
    
    # keys = model.wv.index_to_key
    # wordvector=[]
    # for key in keys:
    #     wordvector.append(model.wv.word_vec(key))
    # km = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1, verbose=False)
    # km_skgram = km.fit(wordvector)
    # labels= km.labels_
    
    # classCollects={}
    # for i in range(len(keys)):
    #     if labels[i] in classCollects.keys():
    #         classCollects[labels[i]].append(keys[i])
    #     else:
    #         classCollects[labels[i]]=[keys[i]]
    # for voc in km.cluster_centers_:
####################### Word2Vec ################################
    # with open(f'{docPath}/FullTitle.pkl', 'rb')as fpick:
    #     titles = pickle.load(fpick)
    # model = Word2Vec.load('word2vec.model')
    # with open(f'{docPath}/Doc2Vec.pkl', 'rb') as fpick:
    #     docVec = pickle.load(fpick)
    # NUM_CLUSTERS=10
    # kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
    # assigned_clusters = kclusterer.cluster(docVec, assign_clusters=True)
        
    # kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
    # kmeans.fit(docVec)
    
    # labels = kmeans.labels_
    # centroids = kmeans.cluster_centers_
    
    # model = TSNE(n_components=2, random_state=0)
    # np.set_printoptions(suppress=True)
    
    # Y=model.fit_transform(docVec)
    
    
    # plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)
    
    
    # for j in range(len(titles)):    
    #     plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
    #     print ("%s %s" % (assigned_clusters[j],  titles[j]))
    
    
    # plt.show()