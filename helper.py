import numpy as np

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def normalize_matrix(a):
    row_sums = a.sum(axis=0)
    new_matrix = a / np.linalg.norm(row_sums[np.newaxis, :])
    return new_matrix

def normalize_matrix_by_rows(a):
    col_sums = a.sum(axis=1)
    new_matrix = a / col_sums[:, np.newaxis]
    return new_matrix
