__author__ = 'prem'

import numpy
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re


def term_freq(k, wordlist):
    """
    takes review_id and wordlist and computes the term frequency
    """
    tf = dict()
    for term in wordlist:
        tf[term] = tf.get(term, 0.0) + 1.0
    return k, tf

def idf(n, docfreq):
    """ Compute the IDF """
    return numpy.log10(numpy.reciprocal(docfreq) * n)


def read_doc(line):
    """ Read one line from review file and split it into Multiple lines and convert it into wordlist for each line
        Note: Removed sentences with less than 6 words and words with less than 4 characters """
    lmtz = WordNetLemmatizer()
    sw = stopwords.words('english')
    review = line.split("\t")
    review_id = review[0]
    sentences = review[5].split(".")
    result = []
    for idx, sent in enumerate(sentences):
        sent_id = review_id + '_' + str(idx)
        words = re.findall(r'[a-zA-Z]+', sent)
        words = [lmtz.lemmatize(w.lower()) for w in words if w.lower() not in sw]
        words = [w for w in words if len(w) > 3]
        if len(words) < 6: continue
        result.append((sent_id, words))
    return result


# This is used to keep the reivew_id and original sentences from reviews
def read_reviews(line):
    """ Read one line from review file and split it into enumerated review id and review sentences tuple"""
    review = line.split("\t")
    review_id = review[0]
    sentences = review[5].split(".")
    result = []
    for idx, sent in enumerate(sentences):
        sent_id = review_id + '_' + str(idx)
        result.append((sent_id, sent))
    return result

def extract_sentences(VT, reviews, columnheader, k=10, n=3):
    """
    Returns a list of summary from VT matrix
    :param VT: Right Singular Matrix of SVD
    :param reviews: reviews RDD <reviewid, sentence>
    :param columnheader: reivew id
    :param k: no of concepts(rows in VT)
    :param n: no of review per concept
    """
    keysentences = []
    # for idxs in numpy.argpartition(VT[:k,:], -n, 1)[:,-n:]:
    for idxs in numpy.fliplr(VT[:k,:].argsort()[:,-n:]):
        for idx in idxs:
            keysentences.append(reviews.lookup(columnheader[idx]))
    return keysentences

def extract_keywords(VT, rowheader, k = 10, n = 5):
    concepts = []
    for idxs in numpy.fliplr(VT[:k,:].argsort()[:,-n:]):
        keywords = []
        for idx in idxs:
            keywords.append(rowheader[idx])
        concepts.append(keywords)
    return concepts