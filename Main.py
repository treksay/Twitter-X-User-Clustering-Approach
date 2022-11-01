# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:36:13 202288

@author: Tarek
"""
import pyLDAvis.gensim  # don't skip this
import pyLDAvis
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import gensim
import random
import pandas as pd
import numpy
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import re
import time
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

stopwords_list = stopwords.words('english')
stopwords_list.extend(['from', 'subject', 're', 'edu', 'use'])
sw = stopwords.words('english')

# Gensim

# spacy for lemmatization

# Plotting tools


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in sw] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def weight(u, w, adj):
    # a function to return the weight between u and v
    return adj[u][w]


@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()


def onlyEnglishTweets(dataset):
    nlp = spacy.load('en_core_web_sm')
    #Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe('language_detector', last=True)
    i = 0
    while i < len(dataset):
        Tweet = dataset[i][2]
        doc = nlp(Tweet)
        detect_language = doc._.language

        if detect_language["language"] != 'en':
            del dataset[i]

            continue
        i = i+1
    return dataset


def column(matrix, i):
    return [row[i] for row in matrix]


def struct_similarity(vcols, wcols, adj, x, y):
    """ Compute the similartiy normalized on geometric mean of vertices"""
    # count the similar rows for unioning edges
    count = [index for index in wcols if (index in vcols)]
    # geomean
    # need to account for vertex itself, add 2(1 for each vertex)
    # ans = (len(count) +2) / (((vcols.size+1)*(wcols.size+1)) ** .5)
    w = weight(x, y, adj)
    ans = (w/(w+1))*(len(count) + 2) / (((vcols.size+1)*(wcols.size+1)) ** .5)
    return ans


def neighborhood(G, vertex_v, eps, adj, u):
    """ Returns the neighbors, as well as all the connected vertices """
    N = deque()
    vcols = vertex_v.tocoo().col
    # check the similarity for each connected vertex
    # test = vertex_v.todense()
    for index in vcols:
        wcols = G[index, :].tocoo().col
        ss = struct_similarity(vcols, wcols, adj, u, index)
        if ss > eps:
            N.append(index)
    return N, vcols


def show_graph_with_labels(adjacency_matrix):

    rows, cols = numpy.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=50, with_labels=False)
    plt.show()


def scan(G, adj, size, eps=0.3, mu=1):
    """
    Vertex Structure = sum of row + itself(1)
    Structural Similarity is the geometric mean of the 2Vertex size of structure
    """

    #v = G.shape[0]
    # All vertices are labeled as unclassified(-1)
    vertex_labels = -numpy.ones(size)
    # start with a neg core(every new core we incr by 1)
    cluster_id = -1
    for vertex in range(size):
        N, vcols = neighborhood(G, G[vertex, :], eps, adj, vertex)
        # must include vertex itself
        N.appendleft(vertex)
        if len(N) >= mu:
            # print "we have a cluster at: %d ,with length %d " % (vertex, len(N))
            # gen a new cluster id (0 indexed)
            cluster_id += 1
            while N:
                y = N.pop()
                R, ycols = neighborhood(G, G[y, :], eps, adj, vertex)
                # include itself
                R.appendleft(y)
                # (struct reachable) check core and if y is connected to vertex
                if len(R) >= mu and y in vcols:
                    # print "we have a structure Reachable at: %d ,with length %d " % (y, len(R))
                    while R:
                        r = R.pop()
                        label = vertex_labels[r]
                        # if unclassified or non-member
                        if (label == -1) or (label == 0):
                            vertex_labels[r] = cluster_id
                        # unclassified ??
                        if label == -1:
                            N.appendleft(r)
        else:
            vertex_labels[vertex] = 0

    # classify non-members
    for index in numpy.where(vertex_labels == 0)[0]:
        ncols = G[index, :].tocoo().col
        if len(ncols) >= 2:
            # mark as a hub
            vertex_labels[index] = -2
            continue

        else:
            # mark as outlier
            vertex_labels[index] = -3
            continue

    return vertex_labels


def getUsers(data):
    # This function creates a list with unique users
    users = []
    for index in data:
        if not(index[1] in users):
            users.append(index[1])
    return users


def getTweets(user, dataset):
    # This function returns a list of tweets for a given user
    tweets = []
    for index in range(len(dataset)):
        if dataset[index][1] == user:
            tweets.append(dataset[index][2])
    return tweets


def repliesTo(tweet1, tweet2):
    s1 = tweet1[3]
    list_of_replies = []
    list_of_replies2 = []
    nbReplies = 0
    if (s1 != ''):
        if (type(tweet1[3]) == str):  # s is string contains ids of replies
            list_of_replies = list(map(int, s1.split(',')))
        else:
            if (not(numpy.isnan(s1))):
                # s is integer that contains one reply
                list_of_replies.append(s1)
        s2 = tweet2[3]
        if (s2 != ''):
            if (type(tweet2[3]) == str):  # s is string contains ids of replies
                try:
                    list_of_replies2 = list(map(int, s2.split(',')))
                except:
                    print(f'tweet: {tweet2}')
                    print(f'String: {s2}')
            else:
                if (not(numpy.isnan(s2))):
                    # s is integer that contains one reply
                    list_of_replies2.append(s2)
        l1 = len(list_of_replies)
        l2 = len(list_of_replies2)
        if ((l1 > 0) & (l2 > 0)):
            for i in list_of_replies2:
                if(i == tweet1[0]):
                    nbReplies = nbReplies + 1
            for i in list_of_replies:
                if(i == tweet2[0]):
                    nbReplies = nbReplies + 1
    return nbReplies


def calculateReplies(user1, user2, dataset):
    tweets1 = []
    tweets2 = []
    replies = 0
    for index in range(len(dataset)):
        if dataset[index][1] == user1:
            tweets1.append(dataset[index])
        if dataset[index][1] == user2:
            tweets2.append(dataset[index])
    for index in tweets1:
        for index2 in tweets2:
            replies = replies + repliesTo(index, index2)
    return replies


def fillMatrix(dataset, users):
    size = len(users)
    matrix = numpy.zeros((size, size))
    for i in range(len(users)):
        for j in range(len(users)):
            if (i != j):
                replies = calculateReplies(users[i], users[j], dataset)
                matrix[i][j] = replies
            matrix[i][i] = 0
    return matrix


def extractHashtags(tweets):
    hashtags = []

    for tweet in tweets:
        hashtag = re.findall(r"#(\w+)", tweet)
        hashtags = hashtags + hashtag
    return hashtags


def cosine(d1, d2):
    # A function to calculate the Cosine similarity betweet 2 documents
    # d1 has users
    # d2 has users

    # Extract Hashtags from each document

    hashtags1 = list(set(d1))
    hashtags2 = list(set(d2))
    # ☻print(hashtags1)
    # print(hashtags2)

    if ((len(hashtags1) > 0) & (len(hashtags2) > 0)):
        #X_list = word_tokenize(hashtags1)
        # Y_list = word_tokenize(hashtag s2)

        # sw contains the list of stopwords

        l1 = []
        l2 = []

        # remove stop words from the string
        X_set = {w for w in hashtags1 if not w in sw}
        Y_set = {w for w in hashtags2 if not w in sw}

        # form a set containing keywords of both strings
        rvector = X_set.union(Y_set)
        for w in rvector:
            if w in X_set:
                l1.append(1)  # create a vector
            else:
                l1.append(0)
            if w in Y_set:
                l2.append(1)
            else:
                l2.append(0)
        c = 0
        fl = float((sum(l1)*sum(l2))**0.5)
        if fl <= 0:
            return 0
        # cosine formula
        for i in range(len(rvector)):
            c += l1[i]*l2[i]

        cosine = c / fl

    else:
        cosine = 0

    return cosine


def centroid(cluster):
    mylist = list(dict.fromkeys(cluster))
    return mylist


def CombineSimilarClusters(users, scan, dataset):

    userDocuments = []
    j = 0
    uniqueuserDocuments = []
    maximum = max(scan)
    while j < len(users):
        if ((scan[j] >= 0) & (not (scan[j] in uniqueuserDocuments))):
            uniqueuserDocuments.append(scan[j])
        else:
            if not (scan[j] in uniqueuserDocuments):  # Add Hubs and Outliers
                maximum = maximum + 1
                uniqueuserDocuments.append(maximum)
                scan[j] = maximum
        j = j+1

    for i in range(len(uniqueuserDocuments)):
        usersPeruserDocuments = []
        for k in range(len(users)):
            if scan[k] == uniqueuserDocuments[i]:
                usersPeruserDocuments.append(users[k])
        userDocuments.append(usersPeruserDocuments)

    uniqueDocuments = []
    documents = []

    for i in range(len(scan)):
        if ((not(scan[i] in uniqueDocuments))):
            uniqueDocuments.append(scan[i])
        #i = i+1

    for i in range(len(uniqueDocuments)):
        hashtagsPerDocument = []
        for j in range(len(users)):
            if scan[j] == uniqueDocuments[i]:
                tweets = getTweets(users[j], dataset)
                hashtagsList = extractHashtags(tweets)
                if(len(hashtagsList) > 0):
                    hashtagsPerDocument = hashtagsPerDocument + hashtagsList
                    hashtagsPerDocument = list(dict.fromkeys(
                        hashtagsPerDocument))  # Remove duplicates
        documents.append(hashtagsPerDocument)

    # Single-pass clustering Technique
    threshold = 0.5  # The threshold of cosine similarity
    clusters = []  # The list of the clusters
    uclusters = []  # The convenient list of the clusters but for users IDs
    clusterRep = []  # the representative for each cluster
    cluster = []  # Contains the documents of each cluster
    ucluster = []
    import copy
    # Step 1:
    if len(documents) > 0:
        d = copy.deepcopy(documents[0])
    else:
        return(["Dataset missing valid data! error: Hashtags"])
    ud = copy.deepcopy(userDocuments[0])
    cluster = d
    ucluster = ud
    clusters.append(cluster)
    uclusters.append(ucluster)
    clusterRep.append(d)
    hashtaglessUsers = []

    # Step 2:  Iteration through all the document to merge them
    for i in range(1, len(documents)):
        if i == 13:
            print("Here!")
        d = copy.deepcopy(documents[i])
        if len(d) > 0:
            ud = copy.deepcopy(userDocuments[i])
            Smax = 0
            SmaxIndex = -1
            for c in range(len(clusters)):
                cos = cosine(clusterRep[c], d)
                if cos > Smax:
                    Smax = cos
                    SmaxIndex = c

            if Smax > threshold:
                cluster = copy.deepcopy(d)
                ucluster = copy.deepcopy(ud)
                clusters[SmaxIndex] += cluster
                clusters[SmaxIndex] = list(dict.fromkeys(clusters[SmaxIndex]))
                uclusters[SmaxIndex] += ucluster
                # Calculate the new cluster Rep
                clusterRep[SmaxIndex] = centroid(clusters[SmaxIndex])

            else:
                cluster = copy.deepcopy(d)
                clusters.append(cluster)
                clusterRep.append(d)
                ucluster = copy.deepcopy(ud)
                uclusters.append(ucluster)
        else:
            hashtaglessUsers.append([users[i]])

    uclusters = uclusters + hashtaglessUsers
    uclusters = removeDuplicates(uclusters)
    return uclusters


def removeDuplicates(mylist):
    i = 0
    while i < len(mylist):
        vector = mylist[i]
        newlist = mylist[i+1:]
        j = 0
        while j < len(mylist):
            if vector in newlist:
                ind = newlist.index(vector)
                newlist.pop(ind)
                mylist.pop(ind+1)
            j = j+1
        i = i+1
    return mylist


def getUsersEmptyCluster(data):
    # This function creates a list with unique users and assigns -1 to their cluster collumn
    users = []
    for i in range(len(data)):
        userNames = [a_tuple[0] for a_tuple in users]
        if not(data['user_name'][i] in userNames):
            user = [data['user_name'][i], -1]
            users.append(user)
    return users


def mostFrequentWords(tweets, x):
    # This method extracts the most frequent words used in a set of tweets
    from collections import Counter
    string = ' '.join([str(item) for item in tweets])
    string = re.sub("#[A-Za-z0-9_]+", "",  string)

    words = string.split()
    cleanWords = []
    for r in words:
        r = r.lower()
        if (not (r in stopwords_list)):
            r = re.sub(r'[^\w]', '', r)
            if len(r) > 0:
                cleanWords.append(r)

    cter = Counter(cleanWords)
    most_occur = cter.most_common(x)

    mfw = [index[0] for index in most_occur]
    return mfw


def cosinex(d1, d2):
    # A function to calculate the Cosine similarity betweet 2 documents
    # d1 has users
    # d2 has users

    # Extract Hashtags from each document

    # ☻print(hashtags1)
    # print(hashtags2)

    if ((len(d1) > 0) & (len(d2) > 0)):
        #X_list = word_tokenize(hashtags1)
        # Y_list = word_tokenize(hashtag s2)

        # sw contains the list of stopwords

        l1 = []
        l2 = []

        X_set = {tuple(str(w)) for w in d1}
        Y_set = {tuple(str(w)) for w in d2}

        # remove stop words from the string

        # form a set containing keywords of both strings
        rvector = X_set.union(Y_set)
        for w in rvector:
            if w in X_set:
                l1.append(1)  # create a vector
            else:
                l1.append(0)
            if w in Y_set:
                l2.append(1)
            else:
                l2.append(0)
        c = 0
        fl = float((sum(l1)*sum(l2))**0.5)
        if fl <= 0:
            return 0
        # cosine formula
        for i in range(len(rvector)):
            c += l1[i]*l2[i]

        cosine = c / fl

    else:
        cosine = 0

    return cosine


def cosine_specific(d1, d2):
    # A function to calculate the Cosine similarity betweet 2 documents
    # d1 has users
    # d2 has users

    l1 = []
    l2 = []
    X_set = set(d1)
    Y_set = set(d2)
    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
        c += l1[i]*l2[i]

    if ((len(l1) != 0) & (len(l2) != 0)):
        cosine = c / float((sum(l1)*sum(l2))**0.5)
    else:
        cosine = 0

    return cosine


def check(S):
    for i in range(len(S)):
        for j in range(i, len(S)):
            if S[j] == i:
                S[i] = i
    return S


def clean(dataset):
    for i in range(len(dataset)):
        tweet = dataset[i][2]
        dataset[i][2] = re.sub("#[A-Za-z0-9_]+", "", tweet)
    return dataset


def show_graph_with_labels(adjacency_matrix):

    rows, cols = numpy.where(adjacency_matrix > 0)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=50, with_labels=False)
    plt.show()


if __name__ == '__main__':
    # Parameters
    #size = 250    # Dataset Size
    nbTopics = 4   # Number of output topics in LDA
    x = 20   # Key Words Number

    t1_start = time.process_time()
    print('Begin')

    df = pd.read_csv('MovieTopics.csv')
    #df = df[0:size]
    dataset = df.values.tolist()
    dataset = onlyEnglishTweets(dataset)
    users = getUsers(dataset)
    adj = fillMatrix(dataset, users)
    show_graph_with_labels(adj)
    G = csr_matrix(adj)
    l = len(users)
    t1_stop = time.process_time()
    PT = t1_stop-t1_start

    print(f'Process Time after matrix : {PT} Seconds')

    print('1 - Structural Clustering Phase')
    S = scan(G, adj, l, .5, 1,)
    S = check(S)

    # Users are now clustered based on their structure

    # Clustering based on their Hashtags :

    print('2 - Content Clustering Phase: Hashtag Usage')
    cleanClusters = CombineSimilarClusters(users, S, dataset)

    # Users are now clustered based on their structure and then their hashtag usage

    # Creation of Pseudousers and Tweets Documents
    print('3 - Generating Pseudo-Users')
    Pseudousers = []
    for i in range(len(cleanClusters)):
        line = []
        tweets = []
        for j in range(len(cleanClusters[i])):
            line.append(cleanClusters[i][j])
        for k in range(len(line)):
            Usertweets = getTweets(line[k], dataset)
            tweets = tweets + Usertweets
        Pseudousers.append([i, tweets])

    # Clustering via Topic Modeling

    # Concat Tweets by PseudoUsers

    dataTweets = {'PU': [], 'text': []}
    for i in range(len(Pseudousers)):
        for j in range(len(Pseudousers[i][1])):
            dataTweets['PU'].append(Pseudousers[i][0])
            tw = re.sub("#[A-Za-z0-9_]+", "", Pseudousers[i][1][j])
            dataTweets['text'].append(tw)
    df2 = pd.DataFrame(dataTweets)

    # LDA

    # Clean Original Seeds

    data = df.text.values.tolist()
    datasetCLEAN = clean(dataset)

    # Run LDA gensim
    print('4 - Topic Modeling: LDA')

    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    data_words = list(sent_to_words(datasetCLEAN))

    # Build the bigram and trigram models
    # higher threshold fewer phrases.
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=[
                                    'NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=nbTopics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=30,
                                                alpha='auto',
                                                per_word_topics=True)

    doc_lda = lda_model[corpus]

    # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # Compute Perplexity
    # a measure of how good the model is. lower the better.
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))

    topics = []
    for index, topic in lda_model.show_topics(formatted=False, num_words=x):
        l = [w[0] for w in topic]
        print(l)
        topics.append(l)

    clusters = []
    for i in range(nbTopics):
        clusters.append([])

    for u in range(len(Pseudousers)):

        '''tweets = []
        for i in range(len(dataset)):
            if users[u][0] == dataset['user_name'][i]:
                tweets.append(dataset['text'][i])'''
        tweets = Pseudousers[u][1]

        # We now have tweets for user u
        # We have to test if user tweets contain > 20 words at least
        mfw = mostFrequentWords(tweets, x)

        ind = -1
        Max = 0

        for j in range(len(topics)):
            topicW = topics[j]

            if(len(mfw) < x):
                topicW = topicW[:len(mfw)]
            cos = cosine_specific(topicW, mfw)

            if cos > Max:

                Max = cos
                ind = j
        #print(f'pseudo = {u}, topic = {ind}, cos = {Max}')
        clusters[ind].append(Pseudousers[u][0])

    # Cleaning the empty clusters

    i = 0
    while (i < len(clusters)):
        if len(clusters[i]) == 0:
            del clusters[i]
            continue
        i = i+1

    '''    
    t1_stop = time.process_time()    
    PT = t1_stop-t1_start
    print(f'Process Time : {PT} Seconds') '''

    FinalClusters = []
    for i in range(len(clusters)):
        FinalClusters.append(cleanClusters[clusters[i][0]])
        for j in range(1, len(clusters[i])):
            if (len(FinalClusters) != 0):  # & (len(FinalClusters[i]) != 0):
                FinalClusters[i] = FinalClusters[i] + \
                    cleanClusters[clusters[i][j]]
            else:
                FinalClusters.append(cleanClusters[clusters[i][j]])
    