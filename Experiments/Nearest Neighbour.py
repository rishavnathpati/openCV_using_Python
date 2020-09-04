import numpy as np
import cv2
from collections import Counter


def nearest_neighbour(A, new_size):

    old_size = A.shape

    row_ratio, col_ratio = new_size[0]/old_size[0], new_size[1]/old_size[1]

    new_row_positions = np.array(range(new_size[0]))+1
    new_col_positions = np.array(range(new_size[1]))+1

    new_row_positions = np.ceil(new_row_positions / row_ratio)
    new_col_positions = np.ceil(new_col_positions / col_ratio)

    row_repeats = np.array(list(Counter(new_row_positions).values()))
    col_repeats = np.array(list(Counter(new_col_positions).values()))
    row_matrix = np.dstack([np.repeat(A[:, i], row_repeats)
                            for i in range(old_size[1])])[0]
    nrow, ncol = row_matrix.shape
    final_matrix = np.stack([np.repeat(row_matrix[i, :], col_repeats)
                             for i in range(nrow)])

    return final_matrix


path = r'D:/Study/Python/openCV/Experiments/DATA/'
img1 = cv2.imread(path+'dog_backpack.png')
new = [1200, 1200]
nearest_neighbour(img1, new)