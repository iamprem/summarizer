__author__ = 'prem'


import numpy
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re

def read_doc_old(line):
    """ Read one line from review file and split it into required format"""
    lmtz = WordNetLemmatizer()
    docwords = line.split("\t")
    review_id = docwords[0]
    wordlist = [];
    for word in re.findall(r'[a-zA-Z]+', docwords[5]):
        word = lmtz.lemmatize(word.lower())
        wordlist.append(word)
    sw = stopwords.words('english')
    wordlist = [w for w in wordlist if w not in sw]
    return (review_id, wordlist)

def term_freq(tuple):
    """
    takes doc object and computes the term frequency
    """
    k = tuple[0]
    wordlist = tuple[1]
    tf = dict()
    for term in wordlist:
        tf[term] = tf.get(term, 0.0) + 1.0
    return (k, tf)

def idf(N, docfreq):
    """ Compute the IDF """
    return numpy.log10(numpy.reciprocal(docfreq) * (N))


def read_doc(line):
    """ Read one line from review file and split it into Multiple lines and convert it into wordlist for each line
        Note: Removed sentences with less than 6 words and words with less than 4 characters """
    lmtz = WordNetLemmatizer()
    sw = stopwords.words('english')
    review = line.split("\t")
    review_id = review[0]
    sentences = review[5].split(".")
    result = [];
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
    """ Read one line from review file and split it into enumerated review id and review senctences tuple"""
    review = line.split("\t")
    review_id = review[0]
    sentences = review[5].split(".")
    result = [];
    for idx, sent in enumerate(sentences):
        sent_id = review_id + '_' + str(idx)
        result.append((sent_id, sent))
    return result

def extract_summary(VT, reviews, columnheader, k = 10, n = 3):
    """
    Returns a list of summary from VT matrix
    :param VT: Right Singular Matrix of SVD
    :param reviews: reviews RDD <reviewid, sentence>
    :param columnheader: reivew id
    :param k: no of concepts(rows in VT)
    :param n: no of review per concept
    """
    summary = [];
    for idxs in numpy.argpartition(VT[:k,:], -n, 1)[:,-n:]:
        for idx in idxs:
            summary.append(reviews.lookup(columnheader[idx]))
    return summary