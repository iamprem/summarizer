__author__ = 'prem'

def mapper(line):
    v_id,v_dict = line[0],line[1]
    currentrank = v_dict['#currank']
    del v_dict['#currank']
    result = []
    outweight = sum(v_dict.itervalues())
    for key, weight in v_dict.iteritems():
        newrank = (currentrank*weight)/outweight
        result.append((key, newrank))
    return result


def update_rank(line, newranks):
    v_id, v_dict = line[0],line[1]
    v_dict['#currank'] = newranks[v_id]
    return (v_id, v_dict)
