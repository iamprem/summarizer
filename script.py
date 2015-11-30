__author__ = 'prem'
from pyspark import SparkContext
import numpy
import tfidf2

sc = SparkContext()

# Read the documents, each document is in one single line or input file
# documents = sc.textFile("data/reviews/B00HWMPSK6.txt").map(lambda doc: TfIdf.read_document(doc))
# documents = sc.textFile("data/reviews/B00HWMPSK6.txt").map(lambda line: tfidf2.read_doc_old(line)) #Takes full review as one sentence
documents = sc.textFile("data/reviews/B00HWMPSK6.txt").flatMap(lambda review: tfidf2.read_doc(review)) #Takes individual sentence from each review
reviews = sc.textFile("data/reviews/B00HWMPSK6.txt").flatMap(lambda review: tfidf2.read_reviews(review))



tf = documents.map(lambda tuple: tfidf2.term_freq(tuple))
# tf = documents.map(lambda doc: TfIdf.term_freq(doc))

# To get the doc ids
# doc_ids = documents.map(lambda doc: TfIdf.get_doc_id(doc))

# To get the vocabulary of the documents
vocabulary = tf.map(lambda tuple: tuple[1].keys()).reduce(lambda x,y: x + y)
# vocabulary = tf.map(lambda x: x[2]['termfreq'].keys()).reduce(lambda x,y: x + y)
vocabulary = numpy.unique(vocabulary)

# def termfreqmatrix(doc):
#     return [doc[2]['termfreq'].get(word,0) for word in vocabulary]
def termfreqmatrix(tfdict):
    return [tfdict.get(word,0) for word in vocabulary]
# def docfreqmatrix(doc):
#     return [ 1.0 if (doc[2]['termfreq'].get(word,0) > 0) else 0. for word in vocabulary]
def docfreqmatrix(tfdict):
    return [ 1.0 if (tfdict.get(word,0) > 0) else 0. for word in vocabulary]

#Create Doc Frequency vector
# dfvector = tf.map(lambda doc: docfreqmatrix(doc)).reduce(lambda x,y : numpy.array(x) + numpy.array(y))
dfvector = tf.map(lambda tuple: docfreqmatrix(tuple[1])).reduce(lambda x,y : numpy.array(x) + numpy.array(y))

#Create Term Frequency matrix
# tfmatrix = tf.map(lambda doc: termfreqmatrix(doc))
tf = tf.map(lambda tuple: (tuple[0], termfreqmatrix(tuple[1]))).sortByKey()
columnheader = tf.keys().collect()
rowheader = vocabulary
tfmatrix = tf.values();



# print 'Term Freq matrix\n' + str(tfmatrix.collect())
# print 'Doc Freq vector\n' + str(dfvector)
# print 'Vocabulary - Row Header\n' + str(vocabulary)
# print 'Review Ids - Col Header\n' + str(columnheader)

# Preparing the matrices(tfidf from tf matrix and idf vector)
tfmatrix = numpy.array(numpy.transpose(tfmatrix.collect()))
idfvector = tfidf2.idf(len(columnheader), dfvector)
idfvector = numpy.array(numpy.transpose(idfvector))
idfvector = numpy.reshape(idfvector, (-1,1))
tfidfMatrix = tfmatrix * idfvector

# print 'IDF Vector\n' + str(idfvector)
# print 'TFIDF Matrix\n' + str(tfidfMatrix)

# Singular Value Decomposition on the tfidf matrix
U, S, VT = numpy.linalg.svd(tfidfMatrix, full_matrices=0)
# print U
# print S
# print VT

# Extracting the key sentences for summary
# print "Max values\n" + str(numpy.amax(VT,1))
# print "Index\n" + str(numpy.argmax(VT,1))
# print str(LA.norm(VT, axis= 1))
# docs = doc_ids.collect();
reviews.cache()
summary = tfidf2.extract_summary(VT,reviews,columnheader)
print summary #Final Summary