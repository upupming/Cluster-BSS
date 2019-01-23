# import data_generator as dg
# import numpy as np
# import matplotlib.pyplot as plt
# import helper
# import matplotlib

# def do_experiment(exp_no, A, sparse=False, fourier_transform=True):
#     if sparse:
#         S = dg.get_sparse_signals(5, 100)
#     else:
#         S = np.random.rand(5, 100)
#     A = np.array(A)
#     A_normalized = helper.normalize_matrix(A)
#     A_sorted = A[:,np.lexsort(A_normalized)]
#     # print(np.lexsort(A_normalized))
#     # print(A_normalized[:,np.lexsort(A_normalized)])

#     X = dg.get_output_signals(A_sorted, S)

#     gridsize=(3, 6)
#     fig = plt.figure(figsize=(12, 8))
#     axes = [matplotlib.axes.Axes] * 8
#     # For input signals
#     axes[0] = plt.subplot2grid(gridsize, (0, 0), colspan=3)
#     axes[1] = plt.subplot2grid(gridsize, (0, 3), colspan=3)
#     axes[2] = plt.subplot2grid(gridsize, (1, 0), colspan=2)
#     axes[3] = plt.subplot2grid(gridsize, (1, 2), colspan=2)
#     axes[4] = plt.subplot2grid(gridsize, (1, 4), colspan=2)
#     # For output signals
#     axes[5] = plt.subplot2grid(gridsize, (2, 0), colspan=2)
#     axes[6] = plt.subplot2grid(gridsize, (2, 2), colspan=2)
#     axes[7] = plt.subplot2grid(gridsize, (2, 4), colspan=2)


#     S_normalized = helper.normalize_matrix_by_rows(S)
#     for i in range(5):
#         axes[i].plot(S_normalized[i], c='y')
#         axes[i].set_title(f'Input signal {i+1}')
    
#     for i in range(5, 8):
#         axes[i].plot(X[i-5], c='r')
#         axes[i].set_title(f'Output signal {i-4}')
    
#     fig.tight_layout()
#     import save_fig as sf
#     sf.save_to_file(f'Input-and-Output-{exp_no}')

#     prefix = './results/'
#     A_fname = prefix + f'A-{exp_no}.txt'
#     A_sorted_fname = prefix + f'A-sorted-{exp_no}.txt'
#     S_fname = prefix + f'S-{exp_no}.txt'
#     X_fname = prefix + f'X-{exp_no}.txt'

#     np.savetxt(A_fname, A)
#     np.savetxt(A_sorted_fname, A_sorted)
#     np.savetxt(S_fname, S)
#     np.savetxt(X_fname, X)


# if __name__ == '__main__':
#     print('====== Step1: Generating data and apply fourier transform... ======')

#     do_experiment(4,
#         [
#             [11, 22, 34, 45, 56],
#             [56, 47, 28, 19, 10],
#             [55, 64, 32, 72, 21]
#         ],
#         sparse=False,
#         fourier_transform=True   
#     )