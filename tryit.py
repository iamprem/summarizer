from pyspark.mllib.common import callMLlibFunc

__author__ = 'prem'

from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF, IDFModel
from pyspark.mllib.feature import IDF
from pyspark.mllib.linalg import (
    Vector, Vectors, DenseVector, SparseVector, _convert_to_vector)
from pyspark.mllib.linalg import (Matrix, Matrices, SparseMatrix)


sc = SparkContext()

# Load documents (one per line).
documents = sc.textFile("data/file0").map(lambda line: line.split(" "))

hashingTF = HashingTF()
tf = hashingTF.transform(documents)
tf.cache()

temp = SparseVector(4,{'1' : 1.0,'2':1.0,'3':1.0,'4':1.0})
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

tfidf1 = idf.transform(temp)


print tfidf.collect()
print tfidf1

def myfun(sparseVector):
    jmodel = callMLlibFunc("fitIDF", 0, _convert_to_vector(sparseVector))
    return IDFModel(jmodel)

a = myfun(temp)
b = a.transform(temp)
print a
print b