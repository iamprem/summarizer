import sys
import dataprep
import trhelp

__author__ = 'prem'

from pyspark import SparkContext

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print >> sys.stderr, "Usage: textrank <iteration-count> <sentence-count> <reveiwfile>"
        print >> sys.stderr, "\t<iteration-count> \t- Number of iteration to avoid long convergence time\n" \
                             "\t<sentence-count> \t- Number of sentences that the final summary should contain\n" \
                             "\t<reveiwfile> \t\t- Review file in a specific format(Check README for the format of the file)"
        exit(-1)

    sc = SparkContext(appName = 'TextRank')#,master = 'local',  pyFiles = ['trhelp.py', 'dataprep.py'])
    path = sys.argv[3]
    iterations = int(sys.argv[1])
    sentcount = int(sys.argv[2])

    # Take all sentences from all review with unique key for all
    revsents = sc.textFile(path).flatMap(lambda review: dataprep.create_vertices(review))

    # Create vertices of graph that contains review_id and wordlist from review sentences
    vertices = revsents.map(lambda l: dataprep.clean_vertex(l))
    revsents = revsents.cache()

    #Construct graph by creating adjacency list for each vertex
    allvertices = vertices.collect()
    graph = vertices.map(lambda ver: dataprep.create_adjlist(ver, allvertices))

    # Filter only vertices that have atleast one node in its adj list.
    # Remove this filter step to save time if the graph is highly connected
    graph = graph.filter(lambda l: len(l[1]) > 0).cache()

    # Set initial rank to all sentences as 0.15
    ranks = graph.map(lambda (vert, neighbors): (vert, 0.15))

    # Iterative TextRank algorithm which will converge after many iterations
    for i in range(0,iterations):
        # Compute contribution by each vertex to its all neighbours and sum the contribution to get the updated rank
        # TextRank Formula: TR(Vi) = (1-d) + d* SUM,Vj in In(Vi) [[Wji * PR(Vj)]/ [SUM, Vk in Out(Vj) [Wjk]]]
        contrcollection = graph.join(ranks).flatMap(lambda (sent, (neigh_dict, r)): trhelp.contributions(neigh_dict, r))
        ranks = contrcollection.reduceByKey(lambda x,y: x+y).mapValues(lambda rank: 0.15 + 0.85 * rank)

    output = []
    # Print the sentences that have higher rank
    finalrank = ranks.collect()
    result = sorted(finalrank, key=lambda x: x[1], reverse=True)
    for j in range(0, sentcount):
        output.append('Rank: ' + str(round(result[j][1],2)) + '\t\tSentence : ' + str(revsents.lookup(result[j][0])))
        print 'Rank: ' + str(round(result[j][1],2)) + '\t\tSentence : ' + str(revsents.lookup(result[j][0]))

    sc.parallelize(output).coalesce(1).saveAsTextFile("output-textrank/")