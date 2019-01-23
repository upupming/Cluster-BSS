import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import array
from scipy.cluster.vq import vq, kmeans
from scipy.optimize import linprog
from scipy.optimize import OptimizeWarning
import warnings
import helper

def k_means(X, N, iter):
    (M, T) = X.shape
    vectors = np.empty((T, M))
    for t in range(T):
        vectors[t] = helper.normalize(X[:,t])
    
    (A_cols, _) = kmeans(vectors, N, iter=iter)
    return A_cols.T

def distance(A, x):
    """
    Assume A_1...N has been normalized
    """
    return np.abs(np.dot(A.T, x))

def cal_S(X, A):
    """
    Assume A has been normalized
    """
    (M, T) = X.shape
    (M, N) = A.shape
    S = np.zeros((N, T))
    for t in range(T):
        print(f'===== At time {t+1} =====')
        x = X[:,t]
        print('x = \n', x)
        indices = np.argsort(
            -distance(A, x)
        )[0:M]
        print('indices = \n', indices)
        print('A[:,indices] = \n', A[:,indices])
        S[:,t][indices] = np.matmul(
            np.linalg.inv(A[:,indices]),
            x
        )
        print('Solve result \n', np.matmul(
            np.linalg.inv(A[:,indices]),
            x
        ))
        print('S[:,t] = \n', S[:,t])
    return S

def cal_S_using_linear_programming(X, A):
    """
    Assume A has been normalized
    """
    (M, T) = X.shape
    (M, N) = A.shape
    S = np.zeros((N, T))
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        try:
            for t in range(T):
                x = X[:,t]
                res = linprog(
                    c=np.ones(N),
                    A_ub=None,
                    b_ub=None,
                    A_eq=A,
                    b_eq=x,
                    options={
                        'tol': 1e-10
                    }
                )
                S[:,t] = res.x
                # print(res)
        except OptimizeWarning as e:
            print('error found:', e)
            exit()
    return S

def do_experiment(exp_no):
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

    (M, N) = A.shape
    (M, T) = X.shape
    print(N)
    A_estimated = k_means(X, N, 100)
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

    S_estimated = cal_S_using_linear_programming(X, normalized_A_estimated_sorted)
    # Normalize
    S = helper.normalize_matrix_by_rows(S)
    S_estimated = helper.normalize_matrix_by_rows(S_estimated)

    # gridsize=(2, 6)
    # fig = plt.figure(figsize=(12, 6))
    # axes = [matplotlib.axes.Axes] * 8
    # # For estimated input signals
    # axes[0] = plt.subplot2grid(gridsize, (0, 0), colspan=3)
    # axes[1] = plt.subplot2grid(gridsize, (0, 3), colspan=3)
    # axes[2] = plt.subplot2grid(gridsize, (1, 0), colspan=2)
    # axes[3] = plt.subplot2grid(gridsize, (1, 2), colspan=2)
    # axes[4] = plt.subplot2grid(gridsize, (1, 4), colspan=2)

    print('Error rate:')
    for i in range(N):

        error_rate = np.linalg.norm(S_estimated[i] - S[i]) / np.linalg.norm(S[i])

        # axes[i].plot(S_estimated[i], c='g')
        # axes[i].set_title(f'Estimated input signal(Normalized) {i+1} \n Error rate: {error_rate}')
        print(error_rate)
    
    # fig.tight_layout()
    # import save_fig as sf
    # sf.save_to_file(f'Estimated-Input-{exp_no}')
    print()

if __name__ == '__main__':
    print('====== Step2: Clustering... ======')
    # do_experiment(1)
    # do_experiment(2)
    do_experiment(4)
    do_experiment(5)
    do_experiment(6)
    do_experiment(7)
    do_experiment(8)