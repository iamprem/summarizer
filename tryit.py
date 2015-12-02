import sys
import dataprep
import trhelp

__author__ = 'prem'

from pyspark import SparkContext

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print >> sys.stderr, "Usage: textrank <iteration-count> <sentence-count> <reveiwfile>"
        exit(-1)

    sc = SparkContext(appName = 'TextRank')#,master = 'local',  pyFiles = ['trhelp.py', 'dataprep.py'])
    # sentences = sc.textFile("data/textrank_test.txt").flatMap(lambda review: tfidf2.read_doc(review))
    path = sys.argv[3]
    iterations = int(sys.argv[1])
    noofsent = int(sys.argv[2])
    vertices = sc.textFile(path).flatMap(lambda review: dataprep.create_vertices(review))
    wordcloud = vertices.map(lambda l: dataprep.clean_vertex(l))
    vertices.cache()
    vert_cache = wordcloud.collect()
    graph = wordcloud.map(lambda ver: dataprep.create_adjlist(ver, vert_cache)).filter(lambda l: len(l[1]) > 0).cache() #Remove this filter if not much use
    rank = graph.map(lambda (vert, neighbors): (vert, 0.15))

    for i in range(0,iterations):
        contrcollection = graph.join(rank).flatMap(lambda (sent, (neigh_dict, r)): trhelp.contributions(neigh_dict, r))
        rank = contrcollection.reduceByKey(lambda x,y: x+y).mapValues(lambda rank: 0.15 + 0.85 * rank)

    rank = rank.collect()
    result = sorted(rank, key=lambda x: x[1], reverse=True)
    for j in range(0, noofsent):
        print 'Rank: ' + str(round(result[j][1],4)) + '\t\tSentence : ' + str(vertices.lookup(result[j][0]))