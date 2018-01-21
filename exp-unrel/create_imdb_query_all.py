import numpy as np

def update_imdb(imdb_old, queries):
    for idx in range(len(imdb_old)):
        imdb_old[idx]['mapped_rels'] = []
        for idx_query, query in enumerate(queries):
            imdb_old[idx]['mapped_rels'].append((0, 1, query[0], query[1], query[2]))
    return imdb_old

def create_imdb_query():
    imdb_tst = np.load('exp-visgeno-rel/data/imdb/imdb_tst.npy')
    imdb_unrel = np.load('exp-unrel/data/imdb/imdb_unrel.npy')

    queries = np.load('exp-unrel/queries.npy')

    imdb_tst = update_imdb(imdb_tst, queries)
    imdb_unrel = update_imdb(imdb_unrel, queries)

    print(imdb_tst.shape)
    print(imdb_unrel.shape)
    imdb_tst_unrel = np.concatenate([imdb_tst, imdb_unrel])
    imdb_tst_unrel.shape
    
    print('saving imdb_tst_unrel.npy to exp-unrel/data/imdb/imdb_tst_unrel')
    np.save('exp-unrel/data/imdb/imdb_tst_unrel', imdb_tst_unrel)

if __name__ == "__main__":
    import sys
    create_imdb_query()