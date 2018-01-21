# modules
import numpy as np
import matplotlib.pylab as plt
from skimage.io import imread

import tensorflow as tf
import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''  # not using GPU

import skimage.transform

from models import visgeno_attention_model, spatial_feat, fastrcnn_vgg_net, lstm_net
from util.visgeno_rel_train.rel_data_reader import DataReader
from util import loss, eval_tools, text_processing
from models import modules
from util.visgeno_rel_train.prepare_batch import load_one_batch
from util import text_processing

import pickle

# variables

model = None
ids_objects = pickle.load(open('../visual-genome/ids_objects.npy', 'rb'))

try:
    imdb_trn
except NameError:
    print('load imdb_trn')
    imdb_trn = np.load('exp-visgeno-rel/data/imdb/imdb_trn.npy')
    imdb_trn_ids = [i['image_id'] for i in imdb_trn]
    
try:
    imdb_tst
except NameError:
    print('load imdb_tst')
    imdb_tst = np.load('exp-visgeno-rel/data/imdb/imdb_tst.npy')
    imdb_tst_ids = [i['image_id'] for i in imdb_tst]
    
try:
    imdb_val
except NameError:
    print('load imdb_val')
    imdb_val = np.load('exp-visgeno-rel/data/imdb/imdb_val.npy')
    imdb_val_ids = [i['image_id'] for i in imdb_val]
    
try:
    imdb_unrel
except NameError:
    print('load imdb_unrel')
    imdb_unrel = np.load('exp-unrel/data/imdb/imdb_unrel.npy')
    imdb_unrel_ids = [i['image_id'] for i in imdb_unrel]
    
try:
    imdb_tst_unrel
except NameError:
    print('load imdb_tst_unrel')
    imdb_tst_unrel = np.load('exp-unrel/data/imdb/imdb_tst_unrel.npy')
    imdb_tst_unrel_ids = [i['image_id'] for i in imdb_tst_unrel]

try:
    imdb_ids
except NameError:
    imdb_ids = {'imdb_trn': imdb_trn_ids, 'imdb_tst': imdb_tst_ids, 'imdb_val': imdb_val_ids, 'imdb_unrel': imdb_unrel_ids, 'imdb_tst_unrel': imdb_tst_unrel_ids}
    
try:
    imdbs
except NameError:
    imdbs = {'imdb_trn': imdb_trn, 'imdb_tst': imdb_tst, 'imdb_val': imdb_val, 'imdb_unrel': imdb_unrel, 'imdb_tst_unrel': imdb_tst_unrel}
    
# Model Params    
T = 20
num_vocab = 72704
embed_dim = 300
lstm_dim = 1000

# Data Params
vocab_file = './word_embedding/vocabulary_72700.txt'
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)
vocab_list = [w.strip() for w in open(vocab_file).readlines()]

# Params for load_ont_batch
min_size = 600
max_size = 1000
max_bbox_num = 10000
max_rel_num = 10000
im_mean = visgeno_attention_model.fastrcnn_vgg_net.channel_mean
model_file = 'downloaded_models/visgeno_attbilstm_strong_iter_360000.tfmodel'

class Model():
    
    def __init__(self, gpu=False):
        
        self.scores_computed = False
        self.s_final = None
        self.probs_subj, self.probs_obj, self.probs_rel = None, None, None
        self.s_subj, self.s_obj, self.s_rel = None, None, None
        self.scores_dir = 'scores/'
        
        tf.reset_default_graph()

        # from https://github.com/ronghanghu/cmn
        self.im_batch = tf.placeholder(tf.float32, [1, None, None, 3]) 
        self.bbox_batch = tf.placeholder(tf.float32, [None, 5]) 
        self.spatial_batch = tf.placeholder(tf.float32, [None, 5])
        self.text_seq_batch = tf.placeholder(tf.int32, [T, None])

        self.scores = visgeno_attention_model.visgeno_attbilstm_net(self.im_batch, self.bbox_batch, self.spatial_batch,
                                                               self.text_seq_batch, num_vocab, embed_dim, lstm_dim, False, False)
        
        self.snapshot_saver = tf.train.Saver()

        # Start Session
        if gpu:
            sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        else:
            self.sess = tf.Session()

        # Run Initialization operations
        self.snapshot_saver.restore(self.sess, model_file)
        
        self.batch = None
        
    def load_batch(self, imdb_str, imdb_idx):
        self.imdb_str = imdb_str
        self.imdb_idx = imdb_idx
        self.image_id = id_from_idx(self.imdb_str, self.imdb_idx)
        self.scores_dir = 'scores/' + imdb_str + '/' + str(imdb_idx) + '/'
        if not os.path.isdir(self.scores_dir):
            os.mkdir(self.scores_dir)
            self.scores_computed = False
        else:
            self.load_scores()
            
        try:
            im = imdbs[imdb_str][imdb_idx]
        except:
            print(imdb_str, '[', imdb_idx, ']', 'does not exist')
        self.batch = load_one_batch(im, im_mean, min_size, max_size,
                                   vocab_dict, T, max_bbox_num, max_rel_num)
        
        self.num_rels = self.batch['text_seq_batch'].shape[1]
        self.num_boxes = self.batch['bbox_batch'].shape[0]
        if self.image_id in ids_objects:
            self.objects_names = [name[0] for name in ids_objects[self.image_id]]
        else:
            self.objects_names = ['<UNK>', '<UNK>']
        
    def load_scores(self):
        self.s_final = np.load(self.scores_dir + 's_final' + '.npy')
        self.probs_subj = np.load(self.scores_dir + 'probs_subj' + '.npy')
        self.probs_rel = np.load(self.scores_dir + 'probs_rel' + '.npy')
        self.probs_obj = np.load(self.scores_dir + 'probs_obj' + '.npy')
        self.s_subj = np.load(self.scores_dir + 's_subj' + '.npy')
        self.s_rel = np.load(self.scores_dir + 's_rel' + '.npy')
        self.s_obj = np.load(self.scores_dir + 's_obj' + '.npy')
        self.scores_computed = True
            
    def compute_scores(self):
        if self.batch is None:
            print('load a batch using load_batch')
            return
        if not self.scores_computed:
            self.s_final, s, probs = self.sess.run([self.scores,
                        tf.get_collection("scores"), tf.get_collection("attention_probs")],
                    feed_dict={
                        self.im_batch            : self.batch['im_batch'],
                        self.bbox_batch          : self.batch['bbox_batch'],
                        self.spatial_batch       : self.batch['spatial_batch'],
                        self.text_seq_batch      : self.batch['text_seq_batch']
                    })
            self.probs_subj, self.probs_obj, self.probs_rel = probs[0]
            self.s_subj, self.s_obj, self.s_rel = s[0]
            self.scores_computed = True
            
            np.save(self.scores_dir + 's_final', self.s_final)
            np.save(self.scores_dir + 'probs_subj', self.probs_subj)
            np.save(self.scores_dir + 'probs_rel', self.probs_rel)
            np.save(self.scores_dir + 'probs_obj', self.probs_obj)
            np.save(self.scores_dir + 's_subj', self.s_subj)
            np.save(self.scores_dir + 's_rel', self.s_rel)
            np.save(self.scores_dir + 's_obj', self.s_obj)
                
    def get_attention_weights(self):
        return self.probs_subj, self.probs_rel, self.probs_obj
        
    def get_score_subj(self):
        return self.s_subj
    
    def get_score_rel(self):
        return self.s_rel
    
    def get_score_obj(self):
        return self.s_obj
    
    def get_score(self):
        return self.s_final
    
    def predict(self, idx_expr):
        num_bboxes = self.s_subj.shape[1]
        score_final = self.s_final[idx_expr].reshape(num_bboxes, num_bboxes)
        return np.argmax(score_final)
        
    
    def draw_attention_weights(self, idx_expr, ax=None, num_words=10):
        k = idx_expr
        # from https://github.com/ronghanghu/cmn
        expr = vocab_indices2sentence(self.batch['text_seq_batch'][:, k])
        is_not_pad = self.batch['text_seq_batch'][:, k] > 0
        words = [vocab_list[idx] for idx in self.batch['text_seq_batch'][is_not_pad, k]]

        if ax is None:
            plt.figure(6, 8)
            tick_marks = np.arange(10)
            plt.xticks(tick_marks, words + ['']*(10-len(words)), rotation=90, fontsize=20)
            plt.xticks([0, 1, 2], ['$a_{subj}$', '$a_{rel}$  ', '$a_{obj}$  '], fontsize=28)
            attention_mat = np.hstack((self.probs_subj[is_not_pad, k], self.probs_rel[is_not_pad, k], self.probs_obj[is_not_pad, k])).T
            attention_mat = np.hstack((attention_mat, np.zeros((3, 10-len(words)), attention_mat.dtype)))
            plt.imshow(attention_mat, interpolation='nearest', cmap='Reds')
            plt.colorbar()
        else:
            fig = plt.gcf()
            tick_marks = np.arange(num_words)
            ax.set_xticks(tick_marks)
            print(len(words))
            ax.set_xticklabels(words + ['']*(num_words-len(words)), rotation=90, fontsize=20)    
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['$a_{subj}$', '$a_{rel}$  ', '$a_{obj}$  '], fontsize=28)
            attention_mat = np.hstack((self.probs_subj[is_not_pad, k], self.probs_rel[is_not_pad, k], self.probs_obj[is_not_pad, k])).T
            attention_mat = np.hstack((attention_mat, np.zeros((3, num_words-len(words)), attention_mat.dtype)))
            im = ax.imshow(attention_mat, interpolation='nearest', cmap='Reds')
            fig.colorbar(im)
            
        
    def draw_scores(self, idx_expr, figsize=(50, 25), plot_ticks=True):
        if idx_expr >= self.num_rels:
            print('idx_expr too high for number of relationships')
            return
        fig, ax = plt.subplots(figsize=figsize)
        ax.matshow(self.s_final[idx_expr, :, :, 0], cmap=plt.get_cmap('plasma'))
        if plot_ticks:
            plt.xticks(range(self.num_boxes), self.objects_names, rotation=90);
            plt.yticks(range(self.num_boxes), self.objects_names);
            
    def get_expr_score_subj(self, idx_expr):
        num_bboxes = self.s_subj.shape[1]
        score_subj = self.s_subj[idx_expr].reshape(num_bboxes)
        return score_subj
    
    def get_expr_score_rel(self, idx_expr):
        num_bboxes = self.s_subj.shape[1]
        score_rel = self.s_rel[idx_expr].reshape(num_bboxes, num_bboxes)
        return score_rel
    
    def get_expr_score_obj(self, idx_expr):
        num_bboxes = self.s_subj.shape[1]
        score_obj = self.s_obj[idx_expr].reshape(num_bboxes)
        return score_obj
    
    def draw_expr_score(self, idx_expr, idx_subj, idx_obj, ax=None):
        s_subj, s_rel, s_obj = self.s_subj, self.s_rel, self.s_obj
        num_bboxes = s_subj.shape[1]
        s_subj = s_subj[idx_expr].reshape(num_bboxes)
        s_obj = s_obj[idx_expr].reshape(num_bboxes)
        s_rel = s_rel[idx_expr].reshape(num_bboxes, num_bboxes)
        s_subj = s_subj[idx_subj]
        s_obj = s_obj[idx_obj]
        s_rel = s_rel[idx_subj, idx_obj]
        if ax is None:
            plt.barh(range(3), [s_subj, s_rel, s_obj], color=['#FF0000', 'blue', '#00FF00'])
            plt.yticks(range(3), ['$subj$', '$rel$', '$obj$'], rotation=90, fontsize=20)
            plt.xlabel('$score$', fontsize=20)
            plt.grid()
        else:
            ax.barh(range(3), [s_subj, s_rel, s_obj], color=['#FF0000', 'blue', '#00FF00'])
            ax.set_yticks(range(3))
            ax.set_yticklabels(['$subj$', '$rel$', '$obj$'], rotation=90, fontsize=20)
            ax.set_xlabel('$score$', fontsize=20)
            ax.grid()
            
    def draw_scores_subj(self, idx_expr, figsize=(50, 12), plot_ticks=True):
        if idx_expr >= self.batch['text_seq_batch'].shape[1]:
            print('idx_expr too high for number of relationships')
            return
        fig, ax = plt.subplots(figsize=figsize)
        scores = self.s_subj[idx_expr].reshape(self.num_boxes)
        scores_objects_names = [(s, n) for s, n in zip(scores, self.objects_names)]
        scores_objects_names_sorted = sorted(scores_objects_names, key=lambda x: x[0])
        scores_sorted = [sn[0] for sn in scores_objects_names_sorted]
        objects_names_sorted = [sn[1] for sn in scores_objects_names_sorted]
        plt.bar(range(self.num_boxes), scores_sorted)
        plt.grid()
        if plot_ticks:
            plt.xticks(range(self.num_boxes, 1, -1), objects_names_sorted, rotation=90)
        
    
    def draw_scores_rel(self, idx_expr, figsize=(50, 25), plot_ticks=True):
        if idx_expr >= self.batch['text_seq_batch'].shape[1]:
            print('idx_expr too high for number of relationships')
            return
        fig, ax = plt.subplots(figsize=figsize)
        ax.matshow(self.s_rel[idx_expr, :, :, 0], cmap=plt.get_cmap('plasma'))
        if plot_ticks:
            plt.xticks(range(self.num_boxes), self.objects_names, rotation=90);
            plt.yticks(range(self.num_boxes), self.objects_names);
    
    def draw_scores_obj(self, idx_expr, figsize=(50, 12), plot_ticks=True):
        if idx_expr >= self.batch['text_seq_batch'].shape[1]:
            print('idx_expr too high for number of relationships')
            return
        fig, ax = plt.subplots(figsize=figsize)
        scores = self.s_obj[idx_expr].reshape(self.num_boxes)
        scores_objects_names = [(s, n) for s, n in zip(scores, self.objects_names)]
        scores_objects_names_sorted = sorted(scores_objects_names, key=lambda x: x[0])
        scores_sorted = [sn[0] for sn in scores_objects_names_sorted]
        objects_names_sorted = [sn[1] for sn in scores_objects_names_sorted]
        plt.bar(range(self.num_boxes), scores_sorted)
        plt.grid()
        if plot_ticks:
            plt.xticks(range(self.num_boxes, 1, -1), objects_names_sorted, rotation=90)
    
# from Quentin Leroy
class Input():

    def load(self, im): # im from imdb_{}
        self.bboxes = im['bboxes']
        self.im_path = im['im_path']
        self.image_id = im['image_id']
        self.mapped_rels = im['mapped_rels']
        if 'obj_idx_map' in im:
            self.obj_idx_map = im['obj_idx_map']
        self.im = imread(self.im_path) # load image
        self.num_bboxes = len(self.bboxes) # num bboxes
        self.num_mapped_rels = len(self.mapped_rels) # num relations
        if 'obj_idx_map' in im:
            self.num_obj = len(self.obj_idx_map) # num objects
        self.rels = [Relation(mapped_rel) for mapped_rel in self.mapped_rels] # relations
        if self.image_id in ids_objects:
            self.objects_names = [name[0] for name in ids_objects[self.image_id]]
        else:
            self.objects_names = ['<UNK>', '<UNK>']
        
    def __init__(self, image_id):
        imdb_str, imdb_idx = idx_from_id(image_id)
        im = imdbs[imdb_str][imdb_idx]
        self.load(im)
        
    def __init__(self, imdb_str, imdb_idx):
        im = imdbs[imdb_str][imdb_idx]
        self.load(im)
        
    def draw_box(self, idx_bbox, ax, plot_im=True):
        if plot_im:
            ax.imshow(self.im)
        print_bbox(self.bboxes[idx_bbox], ax, '-', color='#FF0000', linewidth=5)
        ax.set_title(self.objects_names[idx_bbox])
        plt.axis('off')

    def draw_relation(self, idx, ax, plot_im=True, set_title=True, linewidth=5):
        if idx >= len(self.rels):
            print('idx too high for number of relations')
            return
        rel = self.rels[idx]
        b1, b2 = self.bboxes[rel.idx_obj1], self.bboxes[rel.idx_obj2]
        if plot_im:
            ax.imshow(self.im)
        print_bbox(b1, ax, '-', color='#FF0000', linewidth=linewidth)
        print_bbox(b2, ax, '--', color='#00FF00', linewidth=linewidth)
        ax.axis('off')
        if set_title:
            ax.set_title("\"" + rel.__str__() + "\"")
            
    def draw_pair_boxes(self, idx_subj, idx_obj, ax, plot_im=True, set_title=True):
        if idx_subj >= self.num_bboxes:
            print('idx_subj too high for number of bboxes')
            return
        elif idx_obj >= self.num_bboxes:
            print('idx_obj too high for number of bboxes')
            return
        
        b1, b2 = self.bboxes[idx_subj], self.bboxes[idx_obj]
        if plot_im:
            ax.imshow(self.im)
        print_bbox(b1, ax, '-', color='#FF0000')
        print_bbox(b2, ax, '--', color='#00FF00')
        if set_title:
            ax.set_title(self.objects_names[idx_subj] + " -> " + self.objects_names[idx_obj])
        plt.axis('off')

    def draw_prediction(self, idx, prediction, ax):
        if idx >= len(self.rels):
            print('idx too high for number of relations')
            return
        rel = self.rels[idx]
        b1_idx, b2_idx = prediction // len(self.bboxes), prediction % len(self.bboxes)
        b1, b2 = self.bboxes[b1_idx], self.bboxes[b2_idx]
        ax.imshow(self.im)
        print_bbox(b1, ax, '-', color='#FF0000')
        print_bbox(b2, ax, '--', color='#00FF00')
        plt.axis('off')
        ax.set_title(rel.__str__())

    def draw_text(self, idx, ax):
        rel = self.rels[idx]
        b1 = self.bboxes[rel.idx_obj1]
        b1x, b1y = b1[0], b1[1]
        # write text around/above idk the subject
        ax.text(b1x, b1y, rel.__str__(), style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})

    def draw_patchwork(self, ax):
        ax.imshow(self.im)
        if len(self.rels) <= 5:
            for idx_rel, rel in enumerate(self.rels):
                self.draw_relation(idx_rel, ax, set_title=False, plot_im=False, linewidth=1)
                self.draw_text(idx_rel, ax)
        else:
            indices_rels = np.arange(len(self.rels))
            np.random.shuffle(indices_rels)
            print(indices_rels[:5])
            for idx_rel, rel in enumerate(indices_rels[:5]):
                self.draw_relation(idx_rel, ax, set_title=False, plot_im=False, linewidth=1)
                self.draw_text(idx_rel, ax)



class Relation():

    def __init__(self, mapped_rel):
        self.idx_obj1 = mapped_rel[0]
        self.idx_obj2 = mapped_rel[1]
        self.subj = mapped_rel[2]
        self.rel = mapped_rel[3]
        self.obj = mapped_rel[4]

    def __str__(self):
        return self.subj + " " + self.rel + " " + self.obj
        
        
#######################
# Utilities functions #
#######################

# from https://github.com/ronghanghu/cmn
def print_bbox(bboxes, ax, style='r-', color='#00FF00', linewidth=5):
    bboxes = np.array(bboxes).reshape((-1, 4))
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox   
        xmin-=(linewidth+3)
        ymin-=(linewidth+3)
        xmax+=(linewidth+3)
        ymax+=(linewidth+3)
        ax.plot([xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin], style, color=color, linewidth=linewidth)
        
# from https://github.com/ronghanghu/cmn
def vocab_indices2sentence(indices):
    return ' '.join([vocab_list[idx] for idx in indices if idx != 0])
        
# from image_id to idx in one of the arrays (imdb_trn, imdb_tst, imdb_val, imdb_unrel)  
# image_id is either visual genome id, or id given to unrel images in make_imdb_unrel.ipynb (id starting at MAX_VISGENO_ID)
def idx_from_id(image_id):
    imdb_idx = None
    for imdb_str in imdb_ids:
        imdb_id = imdb_ids[imdb_str] # iterate over the ids of imdb_trn, imdb_tst, imdb_val, imdb_unrel...
        if image_id in imdb_id:
            imdb_idx = imdb_id.index(image_id)
            return imdb_str, imdb_idx
        
    # if the function did not return image_id is not in any imdb
    return None, None

# from idx in one of the arrays (imdb_trn, imdb_tst, imdb_val, imdb_unrel) to image_id
# image_id is either visual genome id, or id given to unrel images in make_imdb_unrel.ipynb (id starting at MAX_VISGENO_ID)
def id_from_idx(imdb_str, imdb_idx):
    image_id = imdb_ids[imdb_str][imdb_idx]
    return image_id