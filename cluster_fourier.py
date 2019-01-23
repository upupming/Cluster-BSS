import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import array
from scipy.cluster.vq import vq, kmeans
from scipy.optimize import linprog
from scipy.optimize import OptimizeWarning
import warnings
import helper
import cluster

def do_experiment(exp_no, using_fourier_transform=True):
    print(f'Running experiment {exp_no}...' )
    prefix = './results/'
    A_fname = prefix + f'A-{exp_no}.txt'
    A_sorted_fname = prefix + f'A-sorted-{exp_no}.txt'
    S_fname = prefix + f'S-{exp_no}.txt'
    X_fname = prefix + f'X-{exp_no}.txt'
    A = np.loadtxt(A_fname)
    A_sorted = np.loadtxt(A_sorted_fname)
    S = np.loadtxt(S_fname)
    X = np.loadtxt(X_fname)

    assert using_fourier_transform == True
    X_transformed = np.fft.fftshift(
        np.fft.fft(X)
    )

    A_estimated = cluster.k_means(X_transformed, 5, 100)
    A_estimated_sorted = A_estimated[:,np.lexsort(A_estimated)]

    print('A = \n', A)
    print('sorted A = \n', A_sorted)
    print('Estimated A = \n', A_estimated)
    print('Estimated sorted A = \n', A_estimated_sorted)

    normalized_A = helper.normalize_matrix(A)
    normalized_A_sorted = helper.normalize_matrix(A_sorted)
    # normalized_A_estimated = helper.normalize_matrix(A_estimated)
    normalized_A_estimated_sorted = helper.normalize_matrix(A_estimated_sorted)
    print('Normalized sorted A = \n', normalized_A_sorted)
    print('Normalized estimated sorted A = \n', normalized_A_estimated_sorted)

    print('Error rate for A: ', np.linalg.norm(normalized_A_sorted - normalized_A_estimated_sorted)/np.linalg.norm(normalized_A))

    S_estimated_transformed = cluster.cal_S_using_linear_programming(X_transformed, normalized_A_estimated_sorted)
    exit()
    # Normalize
    S = helper.normalize_matrix_by_rows(S)
    
    S_estimated = np.fft.ifft(np.fft.ifftshift(S_estimated_transformed))
    
    S_estimated = helper.normalize_matrix_by_rows(S_estimated)

    gridsize=(2, 6)
    fig = plt.figure(figsize=(12, 6))
    axes = [matplotlib.axes.Axes] * 8
    # For estimated input signals
    axes[0] = plt.subplot2grid(gridsize, (0, 0), colspan=3)
    axes[1] = plt.subplot2grid(gridsize, (0, 3), colspan=3)
    axes[2] = plt.subplot2grid(gridsize, (1, 0), colspan=2)
    axes[3] = plt.subplot2grid(gridsize, (1, 2), colspan=2)
    axes[4] = plt.subplot2grid(gridsize, (1, 4), colspan=2)

    print('Error rate:')
    for i in range(5):

        error_rate = np.linalg.norm(S_estimated[i] - S[i]) / np.linalg.norm(S[i])

        axes[i].plot(S_estimated[i], c='g')
        axes[i].set_title(f'Estimated input signal(Normalized) {i+1} \n Error rate: {error_rate}')
        print(error_rate)
    

    fig_fourier = plt.figure(figsize=(12, 6))
    axes_fourier = [matplotlib.axes.Axes] * 8
    # For estimated input signals
    axes_fourier[0] = plt.subplot2grid(gridsize, (0, 0), colspan=3)
    axes_fourier[1] = plt.subplot2grid(gridsize, (0, 3), colspan=3)
    axes_fourier[2] = plt.subplot2grid(gridsize, (1, 0), colspan=2)
    axes_fourier[3] = plt.subplot2grid(gridsize, (1, 2), colspan=2)
    axes_fourier[4] = plt.subplot2grid(gridsize, (1, 4), colspan=2)

    for i in range(5):
        axes[i].plot(S_estimated_transformed[i], c='g')
        axes[i].set_title(f'Estimated fourier-transformed input signal {i+1}')

    fig.tight_layout()
    fig_fourier.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    print('====== Step2: Clustering using fourier transform... ======')
    do_experiment(1, using_fourier_transform=True)