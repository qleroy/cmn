import numpy as np
import matplotlib.pylab as plt

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