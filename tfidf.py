import numpy

__author__ = 'prem'
from collections import Counter


class TfIdf:
    doc_ids = []
    vocabulary = {}
    a = 0

    @staticmethod
    def read_document(oneline):
        """
        Take a single line or review from a document and
        convert it to (doc_id,[word_list])
        Note first word should be doc_id
        :return:
        """
        docwords = oneline.split(" ")
        doc_id = docwords[0]
        wordlist = docwords[1:]
        return [{'doc_id': doc_id}, {'wordlist': wordlist}]

    @staticmethod
    def term_freq(doc):
        """
        takes doc object and computes the term frequency
        :param doc: doc is a list of dictionary
        :return: returns a new list appended with termfreq dictionary
        """
        tf = dict()
        for term in doc[1]['wordlist']:
            tf[term] = tf.get(term, 0.0) + 1.0
        return doc + [{'termfreq': tf}]

    @staticmethod
    def get_doc_id(doc):
        """
        Get the doc id of a document
        :param doc:
        :return:
        """
        return doc[0]['doc_id']

    @staticmethod
    def get_vocabulary(x,y):
        vocset = []
        print x[2]['termfreq']
        # for term in x[2]['termfreq'].keys():
        #     vocset.append(term)
        # for term in y[2]['termfreq'].keys():
        #     vocset.append(term)
        return x

    @staticmethod
    def idf(N, docfreq):
        """
        Compute the IDF
        :param N:
        :param docfreq:
        :return:
        """
        return numpy.log10(numpy.reciprocal(docfreq) * (N))

