__author__ = 'prem'

def contributions(neigh_dict, rank):
    """
    Compute the contribution from a node using it's current rank and its neighbours' weight
    :param neigh_dict: A dictionary which contains {neighbour_nodes : weights(computed using similarity)}
    :param rank: current rank of the node
    :return:list of (node, contribution received) from the parent node for which we runnig this method
    """
    result = []
    outweight = sum(neigh_dict.itervalues())
    for key,weight in neigh_dict.iteritems():
        contrib = (rank*weight)/outweight
        result.append((key, contrib))
    return result