import numpy as np
import cv2
from collections import Counter


def nearest_neighbour(A, new_size):

    old_size = A.shape

    # calculate row and column ratios
    row_ratio, col_ratio = new_size[0]/old_size[0], new_size[1]/old_size[1]

    # define new pixel row position i
    new_row_positions = np.array(range(new_size[0]))+1
    new_col_positions = np.array(range(new_size[1]))+1

    # normalize and ceil new row and col positions by ratios
    new_row_positions = np.ceil(new_row_positions / row_ratio)
    new_col_positions = np.ceil(new_col_positions / col_ratio)

    # find how many times to repeat each element
    row_repeats = np.array(list(Counter(new_row_positions).values()))
    col_repeats = np.array(list(Counter(new_col_positions).values()))

    # perform column-wise interpolation on the columns of the matrix
    row_matrix = np.dstack([np.repeat(A[:, i], row_repeats)
                            for i in range(old_size[1])])[0]

    # perform column-wise interpolation on the columns of the matrix
    nrow, ncol = row_matrix.shape
    final_matrix = np.stack([np.repeat(row_matrix[i, :], col_repeats)
                             for i in range(nrow)])

    return final_matrix


path = r'D:/Study/Python/openCV/Experiments/DATA/'
img1 = cv2.imread(path+'dog_backpack.png')
new = [1200, 1200]
nearest_neighbour(img1, new)
# print(final)
