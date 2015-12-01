import dataprep
import tfidf2
import trhelp

__author__ = 'prem'

from pyspark import SparkContext


sc = SparkContext()
# sentences = sc.textFile("data/textrank_test.txt").flatMap(lambda review: tfidf2.read_doc(review))
path = "data/reviews/B00HWMPSK6.txt"
vertices = sc.textFile(path).flatMap(lambda review: dataprep.create_vertices(review))
wordcloud = vertices.map(lambda l: dataprep.clean_vertex(l))
vertices.cache()
vert_cache = wordcloud.collect()
graph = wordcloud.map(lambda ver: dataprep.create_adjlist(ver, vert_cache)).filter(lambda l: len(l[1]) > 1) #Remove this filter if not much use
newrank = []
for i in range(0,10):
    newrank = graph.flatMap(lambda v : trhelp.mapper(v)).reduceByKey(lambda x,y: x+y).mapValues(lambda rank: (0.15 + 0.85*rank)).collect()
    rankdict = {x:y for x,y in newrank}
    graph = graph.map(lambda ver: trhelp.update_rank(ver, rankdict))

result = sorted(newrank, key=lambda x: x[1], reverse=True)
for j in range(0, 20):
    print 'Rank: ' + str(result[j][1]) + '\t\tSentence : ' + str(vertices.lookup(result[j][0]))

print '::End::'