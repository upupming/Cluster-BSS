import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import helper

def get_sparse_signals(N, T):
    """
    完全稀疏矩阵
    """
    S_raw = np.random.rand(N, T)
    k_filter = np.random.randint(N, size=T)
    S = np.zeros(S_raw.shape)
    for t in range(T):
        # Filter all except k_filter[t]
        mask = k_filter[t]
        S[mask][t] = S_raw[mask][t]
    return S

def get_output_signals(A, S):
    return np.matmul(A, S)

def do_experiment(exp_no, A, sparse=True, T=100):
    A = np.array(A)
    (_, N) = A.shape

    if sparse:
        S = get_sparse_signals(N, T)
    else:
        S = np.random.rand(N, T)
    A_normalized = helper.normalize_matrix(A)
    A_sorted = A[:,np.lexsort(A_normalized)]
    # print(np.lexsort(A_normalized))
    # print(A_normalized[:,np.lexsort(A_normalized)])

    X = get_output_signals(A_sorted, S)

    # gridsize=(3, 6)
    # fig = plt.figure(figsize=(12, 8))
    # axes = [matplotlib.axes.Axes] * 8
    # # For input signals
    # axes[0] = plt.subplot2grid(gridsize, (0, 0), colspan=3)
    # axes[1] = plt.subplot2grid(gridsize, (0, 3), colspan=3)
    # axes[2] = plt.subplot2grid(gridsize, (1, 0), colspan=2)
    # axes[3] = plt.subplot2grid(gridsize, (1, 2), colspan=2)
    # axes[4] = plt.subplot2grid(gridsize, (1, 4), colspan=2)
    # # For output signals
    # axes[5] = plt.subplot2grid(gridsize, (2, 0), colspan=2)
    # axes[6] = plt.subplot2grid(gridsize, (2, 2), colspan=2)
    # axes[7] = plt.subplot2grid(gridsize, (2, 4), colspan=2)


    # S_normalized = helper.normalize_matrix_by_rows(S)
    # for i in range(5):
    #     axes[i].plot(S_normalized[i], c='y')
    #     axes[i].set_title(f'Input signal {i+1}')
    
    # for i in range(5, 8):
    #     axes[i].plot(X[i-5], c='r')
    #     axes[i].set_title(f'Output signal {i-4}')
    
    # fig.tight_layout()
    # import save_fig as sf
    # sf.save_to_file(f'Input-and-Output-{exp_no}')

    prefix = './results/'
    A_fname = prefix + f'A-{exp_no}.txt'
    A_sorted_fname = prefix + f'A-sorted-{exp_no}.txt'
    S_fname = prefix + f'S-{exp_no}.txt'
    X_fname = prefix + f'X-{exp_no}.txt'

    np.savetxt(A_fname, A)
    np.savetxt(A_sorted_fname, A_sorted)
    np.savetxt(S_fname, S)
    np.savetxt(X_fname, X)

if __name__ == '__main__':
    print('====== Step1: Generating data... ======')

    # do_experiment(1, 
    #     [
    #         [11, 22, 34, 45, 56],
    #         [56, 47, 28, 19, 10],
    #         [55, 64, 32, 72, 21]
    #     ]
    # )

    # do_experiment(2,
    #     [
    #         [5100, 22, 34, 45, 56],
    #         [5600, 47, 28, 19, 10],
    #         [5500, 64, 32, 72, 21]
    #     ]
    # )

    # do_experiment(3,
    #     [
    #         [11, 22, 34, 45, 56],
    #         [56, 47, 28, 19, 10],
    #         [55, 64, 32, 72, 21]
    #     ],
    #     sparse=False
    # )

    # N = 8, M = 5, T = 100
    do_experiment(4,
        [
            [11, 22, 34, 45, 56, 20, 34, 56],
            [56, 47, 28, 19, 10, 34, 54, 65],
            [55, 64, 32, 72, 21, 23, 43, 43],
            [34, 56, 75, 23, 45, 46, 65, 65],
            [45, 67, 87, 45, 67, 34, 34, 12]
        ],
        sparse=True
    )

    # N = 8, M = 6, T = 100
    do_experiment(5,
    [
        [11, 22, 34, 45, 56, 20, 34, 56],
        [56, 47, 28, 19, 10, 34, 54, 65],
        [55, 64, 32, 72, 21, 23, 43, 43],
        [34, 56, 75, 23, 45, 46, 65, 65],
        [45, 67, 87, 45, 67, 34, 34, 12],
        [32, 43, 24, 43, 24, 45, 12, 23]
    ],
    sparse=True
    )

    # N = 8, M = 6, T = 10000
    do_experiment(6,
    [
        [11, 22, 34, 45, 56, 20, 34, 56],
        [56, 47, 28, 19, 10, 34, 54, 65],
        [55, 64, 32, 72, 21, 23, 43, 43],
        [34, 56, 75, 23, 45, 46, 65, 65],
        [45, 67, 87, 45, 67, 34, 34, 12],
        [32, 43, 24, 43, 24, 45, 12, 23]
    ],
    sparse=True,
    T=10000
    )

    # N = 10, M = 8, T = 100
    do_experiment(7,
        [
            [11, 22, 34, 45, 56, 20, 34, 56, 24, 65],
            [56, 47, 28, 19, 10, 34, 54, 65, 87, 24],
            [55, 64, 32, 72, 21, 23, 43, 43, 23, 43],
            [34, 56, 75, 23, 45, 46, 65, 65, 23, 24],
            [45, 67, 87, 45, 67, 34, 34, 12, 35, 13],
            [53, 23, 35, 34, 23, 64, 44, 64, 33, 42],
            [454, 645, 47, 46, 47, 34, 45, 90, 35, 23],
            [452, 675, 82, 42, 64, 34, 43, 54, 22, 53]
        ],
        sparse=True
    )

    # N = 15, M = 12, T = 100
    do_experiment(8,
        [
            [131, 22, 34, 45, 56, 20, 34, 56, 24, 65, 21, 43, 283, 438, 325],
            [56, 47, 28, 19, 100, 34, 54, 65, 87, 24, 453, 123, 4323, 987, 23],
            [355, 64, 32, 72, 21, 23, 43, 43, 23, 43, 32, 43, 32, 32, 32],
            [334, 56, 75, 203, 45, 46, 65, 65, 23, 24, 43, 42, 323, 23, 322],
            [435, 67, 87, 405, 67, 34, 34, 12, 35, 13, 42 ,24, 23, 32, 321],
            [533, 23, 35, 304, 23, 64, 44, 64, 33, 42, 45, 24, 233, 132, 23],
            [454, 645, 47, 46, 47, 34, 45, 90, 35, 23, 35, 35, 232, 23, 2332],
            [452, 675, 82, 42, 64, 34, 43, 54, 22, 53, 35, 43, 322, 43, 212],
            [635, 57, 47, 350, 37, 84, 74, 62, 55, 53, 43 ,244, 43, 33, 342],
            [334, 45, 43, 45, 45, 54, 65, 76, 45, 32, 546, 65, 21, 32, 423],
            [4545, 6455, 474, 446, 472, 343, 425, 920, 335, 233, 34, 36, 434, 434, 243],
            [4523, 63475, 182, 142, 464, 343, 433, 354, 322, 353, 335, 543, 213, 324, 243]
        ],
        sparse=True
    )