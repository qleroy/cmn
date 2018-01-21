import numpy as np
import pickle

scores_query = np.load('exp-unrel/results/scores_query_17.npy')

def subj_obj_idx(scores, idx, num_boxes):
    subj_idx = idx // num_boxes
    obj_idx = idx % num_boxes
    return subj_idx, obj_idx

a = []
for s in scores_query:
    num_boxes = int(np.sqrt(len(s[0][0])))
    for idx, ss in enumerate(s[0][0]):
        a.append((ss, s[1], subj_obj_idx(s[0][0], idx, num_boxes)))

b = sorted(a, key=lambda x: -x[0])

file = open('exp-unrel/results/scores_query_17_refactored', 'wb')
pickle.dump(b, file)
file.close()