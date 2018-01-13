from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # using GPU 0

import tensorflow as tf
import numpy as np
import skimage.io
import skimage.transform

from models import visgeno_attention_model, spatial_feat, fastrcnn_vgg_net
from util.visgeno_rel_train.rel_data_reader import DataReader
from util import loss, eval_tools, text_processing

################################################################################
# Parameters
################################################################################

# Model Params
T = 20
num_vocab = 72704
embed_dim = 300
lstm_dim = 1000

# Data Params
imdb_file = './exp-unrel/data/imdb/imdb_tst_unrel.npy'
vocab_file = './word_embedding/vocabulary_72700.txt'
im_mean = visgeno_attention_model.fastrcnn_vgg_net.channel_mean

# Snapshot Params
model_file = './downloaded_models/visgeno_attbilstm_strong_iter_360000.tfmodel'

result_file = './exp-visgeno-rel/results/visgeno_attbilstm_strong_iter_360000_tst_pair.txt'

################################################################################
# Network
################################################################################

im_batch = tf.placeholder(tf.float32, [1, None, None, 3])
bbox_batch = tf.placeholder(tf.float32, [None, 5])
spatial_batch = tf.placeholder(tf.float32, [None, 5])
text_seq_batch = tf.placeholder(tf.int32, [T, None])

scores = visgeno_attention_model.visgeno_attbilstm_net(im_batch, bbox_batch, spatial_batch,
    text_seq_batch, num_vocab, embed_dim, lstm_dim, False, False)

np.random.seed(3)
reader = DataReader(imdb_file, vocab_file, im_mean, shuffle=False, max_bbox_num=10000, max_rel_num=10000)

################################################################################
# Snapshot and log
################################################################################

# Snapshot saver
snapshot_saver = tf.train.Saver()

# Start Session
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

# Run Initialization operations
snapshot_saver.restore(sess, model_file)

################################################################################
# Optimization loop
################################################################################

# init scores
scores_query = []
# Run optimization

cpt = 0
for n_iter in range(reader.num_batch):
    batch = reader.read_batch()
    print('\tthis batch: N_lang = %d, N_bbox = %d' %
          (batch['expr_obj1_batch'].shape[1], batch['bbox_batch'].shape[0]))

    # Forward and Backward pass
    scores_val = sess.run(scores,
        feed_dict={
            im_batch            : batch['im_batch'],
            bbox_batch          : batch['bbox_batch'],
            spatial_batch       : batch['spatial_batch'],
            text_seq_batch      : batch['text_seq_batch']
        })

    N_batch, N_box, _, _ = scores_val.shape

    # scores_val has shape [N_batch, N_box, N_box, 1]
    scores_flat = scores_val.reshape((N_batch, N_box*N_box))
    
    scores_query.append((scores_flat, batch['image_id']))

    cpt = cpt + 1
    if cpt % 100:
        np.save('exp-unrel/results/scores_query', np.array(scores_query))

np.save('exp-unrel/results/scores_query', np.array(scores_query))




