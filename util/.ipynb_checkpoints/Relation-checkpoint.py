import skimage
import matplotlib.pylab as plt
import numpy as np

# from https://github.com/ronghanghu/cmn
def print_bbox(bboxes, style='r-', color='#00FF00', linewidth=5):
    """A utility function to help visualizing boxes."""
    bboxes = np.array(bboxes).reshape((-1, 4))
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        xmin-=(linewidth+3)
        ymin-=(linewidth+3)
        xmax+=(linewidth+3)
        ymax+=(linewidth+3)
        plt.plot([xmin, xmax, xmax, xmin, xmin],
                 [ymin, ymin, ymax, ymax, ymin], style, color=color, linewidth=linewidth)
        
# from Quentin Leroy
class Image():
    
    def __init__(self, im):
        self.bboxes = im['bboxes']
        self.im_path = im['im_path']
        self.image_id = im['image_id']
        self.mapped_rels = im['mapped_rels']
        if 'obj_idx_map' in im:
            self.obj_idx_map = im['obj_idx_map']
        # load image
        self.im = skimage.io.imread(self.im_path)
        # num bboxes
        self.num_bboxes = len(self.bboxes)
        # num mapped rels
        self.num_mapped_rels = len(self.mapped_rels)
        # num objects
        if 'obj_idx_map' in im:
            self.num_obj = len(self.obj_idx_map)
        # rels
        self.rels = [Relation(mapped_rel) for mapped_rel in self.mapped_rels]
        
    def draw_relation(self, idx):
        if idx >= len(self.rels):
            print('idx too high for number of relations')
            return
        rel = self.rels[idx]
        b1, b2 = self.bboxes[rel.idx_obj1], self.bboxes[rel.idx_obj2]
        plt.imshow(self.im)
        print_bbox(b1, '-', color='#FF0000')
        print_bbox(b2, '--', color='#00FF00')
        plt.axis('off')
        plt.title(rel.__str__())

class Relation():
    
    def __init__(self, mapped_rel):
        self.idx_obj1 = mapped_rel[0]
        self.idx_obj2 = mapped_rel[1]
        self.subj = mapped_rel[2]
        self.rel = mapped_rel[3]
        self.obj = mapped_rel[4]
        
    def __str__(self):
        return self.subj + " " + self.rel + " " + self.obj
    
