__author__ = 'prem'
from pyspark import SparkContext
from tfidf import TfIdf
import numpy

sc = SparkContext()

# Read the documents, each document is in one single line or input file
documents = sc.textFile("/home/prem/Desktop/temp/test_inputs/filefull").map(lambda doc: TfIdf.read_document(doc))
tf = documents.map(lambda doc: TfIdf.term_freq(doc))

# To get the doc ids
doc_ids = documents.map(lambda doc: TfIdf.get_doc_id(doc))

# To get the vocabulary of the documents
vocabulary = tf.map(lambda x: x[2]['termfreq'].keys()).reduce(lambda x,y: x + y)
vocabulary = numpy.unique(vocabulary)

def termfreqmatrix(doc):
    return [doc[2]['termfreq'].get(word,0) for word in vocabulary]
def docfreqmatrix(doc):
    return [ 1.0 if (doc[2]['termfreq'].get(word,0) > 0) else 0. for word in vocabulary]

#Create Term Frequency matrix
tfmatrix = tf.map(lambda doc: termfreqmatrix(doc))

#Create Doc Frequency vector
dfvector = tf.map(lambda doc: docfreqmatrix(doc)).reduce(lambda x,y : numpy.array(x) + numpy.array(y))

print tfmatrix.collect()
print dfvector
print doc_ids.collect()
print vocabulary

# Preparing the matrices(tfidf from tf matrix and idf vector)
tfmatrix = numpy.array(numpy.transpose(tfmatrix.collect()))
idfvector = TfIdf.idf(doc_ids.count(), dfvector)
idfvector = numpy.array(numpy.transpose(idfvector))
idfvector = numpy.reshape(idfvector, (-1,1))
tfidfMatrix = tfmatrix * idfvector

print idfvector
print tfidfMatrix

# Singular Value Decomposition on the tfidf matrix
U, S, VT = numpy.linalg.svd(tfidfMatrix, full_matrices=0)
print U
print S
print VT.T