import re
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


def create_vertices(line):
    """ Read one line from review file and split it into enumerated review id and review sentences tuple"""
    review = line.split("\t")
    review_id = review[0]
    sentences = review[5].split(".")
    result = []
    for idx, sent in enumerate(sentences):
        sent_id = review_id + '_' + str(idx)
        sent_len = len(sent.split(" "))
        if 10 < sent_len < 30:
            result.append((sent_id, sent))
    return result


def clean_vertex(line):
    """ Take review id and sentence tuple and clean it to (review_id, word list) """
    rev_id, sent = line[0], line[1]
    lmtz = WordNetLemmatizer()
    sw = stopwords.words('english')
    words = re.findall(r'[a-zA-Z]+', sent)
    words = [lmtz.lemmatize(w.lower()) for w in words if w.lower() not in sw]
    words = [w for w in words if len(w) > 3]
    return rev_id, words


def create_adjlist(vertex, allvertices):
    """
    Take a (review_id,sentence) 'RS1' and all other (review_id, sentence) [RS_all] and create adjacency list for 'RS1'
    by measuring similarity between two 'RS1' and all other reviews
    Returns review_id, {other_review_ids : similarity_weights}
    """
    k,v = vertex[0],vertex[1]
    edgedict = {}
    for u in allvertices:
        edge = sim_measure(vertex, u)
        if edge is not None:
            edgedict[edge[0]] = edge[1]
    return (k, edgedict)


def sim_measure(input1, input2):
    """
    Measure similarity between two (review_id,words) and returns a value
    similarity = (count_of_common_words_S1S2)/ (1 + log2(len(S1)) + log2(len(S2)))
    """
    k1,v1 = input1[0],input1[1]
    k2,v2 = input2[0],input2[1]
    if k1 != k2: #To skip measuring between same sentence
        Wk = len(set(v1).intersection(v2))
        logSlen = np.log2(len(v1)) + np.log2(len(v2))
        simval = Wk/(logSlen + 1)
        if simval != 0:
            return (k2, simval)
