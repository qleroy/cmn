import numpy as np

scores_queries_file = 'exp-unrel/results/scores_query_17.npy'

scores_queries = np.load(scores_queries_file)

print(scores_queries.shape)
print(scores_queries[0].shape)
print(scores_queries[0])