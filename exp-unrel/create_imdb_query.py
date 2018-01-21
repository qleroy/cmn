import numpy as np

def update_imdb_all(imdb_old, query):
    # replace mapped_rels by [(0,1,query[0], query[1], query[2])]
    for idx in range(len(imdb_old)):
        imdb_old[idx]['mapped_rels'] = [(0, 1, query[0], query[1], query[2])]
    return imdb_old

def create_imdb_query(id_query):
    imdb_tst = np.load('exp-visgeno-rel/data/imdb/imdb_tst.npy')
    imdb_unrel = np.load('exp-unrel/data/imdb/imdb_unrel.npy')

    queries = np.load('exp-unrel/queries.npy')
    
    if id_query >=len(queries)-1:
        print('number of queries ' + str(len(queries)))
        return
    query = queries[id_query]
    print("query for retrieval:", query)

    imdb_tst = update_imdb(imdb_tst, query)
    imdb_unrel = update_imdb(imdb_unrel, query)

    print(imdb_tst.shape)
    print(imdb_unrel.shape)
    imdb_tst_unrel = np.concatenate([imdb_tst, imdb_unrel])
    imdb_tst_unrel.shape
    
    print('saving imdb_tst_unrel.npy to exp-unrel/data/imdb/imdb_tst_unrel')
    np.save('exp-unrel/data/imdb/imdb_tst_unrel', imdb_tst_unrel)

if __name__ == "__main__":
    import sys
    if len(sys.argv) is not 2:
        print("Usage: python create_imdb_query.py id_query")
    else:
        # TAKE query
        query = int(sys.argv[1])
        create_imdb_query(query)