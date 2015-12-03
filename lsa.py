import sys
from pyspark import SparkContext
import numpy
import tfidf2

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: lsa <-s or -w> <reveiwfile>"
        print >> sys.stderr, "\t<-s or -w> \t- Extract key sentences or Key words(Transpose of TF-IDF to compute)\n" \
                             "\t<reveiwfile> \t\t- Review file in a specific format(Check README for the format of the file)"
        exit(-1)
    sc = SparkContext(appName= 'LSA')#, pyFiles= ['tfidf2.py', 'trhelp.py'])
    file = sys.argv[2]
    flag = sys.argv[1]

    # Takes individual sentence from each review
    documents = sc.textFile(file).flatMap(lambda review: tfidf2.read_doc(review))
    reviews = sc.textFile(file).flatMap(lambda review: tfidf2.read_reviews(review))

    # Term Frequency
    tf = documents.map(lambda (k, wordlist): tfidf2.term_freq(k, wordlist))

    # Get the vocabulary of the documents
    vocabulary = tf.map(lambda tuple: tuple[1].keys()).reduce(lambda x,y: x + y)
    vocabulary = numpy.unique(vocabulary)

    def termfreqmatrix(tfdict):
        return [tfdict.get(word, 0) for word in vocabulary]

    def docfreqmatrix(tfdict):
        return [ 1.0 if (tfdict.get(word, 0) > 0) else 0. for word in vocabulary]

    #Create Doc Frequency vector
    dfvector = tf.map(lambda tuple: docfreqmatrix(tuple[1])).reduce(lambda x, y: numpy.array(x) + numpy.array(y))

    #Create Term Frequency matrix
    tf = tf.map(lambda (rev_id, tfdict): (rev_id, termfreqmatrix(tfdict))).sortByKey()
    tfmatrix = tf.values()
    columnheader = tf.keys().collect()
    rowheader = vocabulary

    # Preparing the matrices(tfidf from tf matrix and idf vector)
    tfmatrix = numpy.array(numpy.transpose(tfmatrix.collect()))
    idfvector = tfidf2.idf(len(columnheader), dfvector)
    idfvector = numpy.array(numpy.transpose(idfvector))
    idfvector = numpy.reshape(idfvector, (-1,1))
    tfidfMatrix = tfmatrix * idfvector

    reviews.cache()
    # Singular Value Decomposition on the tfidf matrix
    if flag == '-s':
        # Summary Sentences - Extraction
        U, S, VT = numpy.linalg.svd(tfidfMatrix, full_matrices=0)
        concepts = tfidf2.extract_sentences(VT,reviews,columnheader)
        for i,concept in enumerate(concepts):
            for j,sent in enumerate(concept):
                print '[Concept '+str(i+1)+'][Sentence '+str(j+1)+'] :\t'+str(sent) #Final Summary
            print '\n'
    elif flag == '-w':
        # Summary Keywords - Abstraction
        U, S, VT = numpy.linalg.svd(tfidfMatrix.T, full_matrices=0)
        concepts = tfidf2.extract_keywords(VT, rowheader)
        for i,concept in enumerate(concepts):
            print '[Concept '+str(i+1)+'] :\t'+str(concept)
