import numpy as np
from scipy.stats import pearsonr, shapiro
from sklearn.linear_model import LinearRegression
from lingam.hsic import hsic_test_gamma
import time
import copy

from find_ancestor import find_loop, wald_eliminate_loop, value_eliminate_loop, ancestor_eliminate_loop


def get_resid_and_coef(X, endog_idx, exog_idcs):
    """Get the residuals and coefficients of the ordinary least squares method"""
    lr = LinearRegression()
    lr.fit(X[:, exog_idcs], X[:, endog_idx])
    resid = X[:, endog_idx] - lr.predict(X[:, exog_idcs])
    return resid, lr.coef_


def is_correlated(a, b, cor_alpha):
    """Estimate that the two variables are linearly correlated using the Pearson's correlation"""
    return pearsonr(a, b)[1] < cor_alpha


def is_independent(X, Y, ind_alpha):
    _, p = hsic_test_gamma(X, Y, bw_method="mdbs")
    return p > ind_alpha


def is_parent(X, M, xj, xi, ind_alpha):
    if len(M[xi] - set([xj])) > 0:
        zi, _ = get_resid_and_coef(X, xi, list(M[xi] - set([xj])))
    else:
        zi = X[:, xi]

    if len(M[xi] & M[xj]) > 0:
        wj, _ = get_resid_and_coef(X, xj, list(M[xi] & M[xj]))
    else:
        wj = X[:, xj]

    # Check if zi and wj are correlated
    return not is_independent(wj, zi, ind_alpha)
    # return is_correlated(wj, zi, ind_alpha)

def extract_parents(X, M, ind_alpha):
    """Extract parents (direct causes) from a set of ancestors"""
    n_features = X.shape[1]
    print("the number of feature: ", n_features)
    P = [set() for i in range(n_features)]

    for xi in range(n_features):
        for xj in M[xi]:
            # Check if xj is the parent of xi
            if is_parent(X, M, xj, xi, ind_alpha):
                P[xi].add(xj)

    return P


def get_resid_to_parent(X, idx, P):
    if len(P[idx]) == 0:
        return X[:, idx]

    resid, _ = get_resid_and_coef(X, idx, list(P[idx]))
    return resid


def estimate_adjacency_matrix(X, P):
    """Estimate adjacency matrix by causal parents and confounders.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.

    Returns
    -------
    self : object
        Returns the instance itself.
    """
    # Check parents
    n_features = X.shape[1]
    B = np.zeros([n_features, n_features], dtype="float64")
    for xi in range(n_features):
        xj_list = list(P[xi])
        xj_list.sort()
        if len(xj_list) == 0:
            continue

        _, coef = get_resid_and_coef(X, xi, xj_list)
        for j, xj in enumerate(xj_list):
            B[xi, xj] = coef[j]

    return B


def change_list_to_set(ancestor_dict_list):
    
    ancestor_dict_set = [set() for i in range(len(ancestor_dict_list))]
    for i in range(len(ancestor_dict_list)):
        ancestor_dict_set[i] = set(ancestor_dict_list[i])
    return ancestor_dict_set


def get_res(data, B, ancestor_list, ind_alpha):
    
    ancestor_dict = change_list_to_set(ancestor_list)
    M = np.zeros_like(B)
    visit = np.zeros_like(B)
    
    N_num = len(M)
    
    P = extract_parents(data, ancestor_dict, ind_alpha)
    M_res_rcd = estimate_adjacency_matrix(data, P)

    # print("******************")
    ###### based on wald test
    time_0 = time.time()
    M_temp_1 = copy.deepcopy(M_res_rcd)
    temp_loop = find_loop(M_temp_1)
    # count = 0
    while(len(temp_loop)):
        M_temp_1 = wald_eliminate_loop(M_temp_1, temp_loop, data)
        temp_loop = find_loop(M_temp_1)
    
    
    ###### based on ancestors
    time_1 = time.time()
    M_temp_2 = copy.deepcopy(M_res_rcd)
    # temp_loop = find_loop(M_temp_2)
    # count = 0
    # while(len(temp_loop)):
    #     # print("loop!")
    #     # M = value_eliminate_loop(M, temp_loop)
    #     if count == 0:
    #         M_temp_2 = ancestor_eliminate_loop(M_temp_2, temp_loop, ancestor_list)
    #     else:
    #         M_temp_2 = value_eliminate_loop(M_temp_2, temp_loop)
    #     # M = wald_eliminate_loop(M, temp_loop, data)
    #     temp_loop = find_loop(M_temp_2)
    #     count += 1
    #     # print("a")

    ###### based on value
    time_2 = time.time()
    M_temp_3 = copy.deepcopy(M_res_rcd)
    temp_loop = find_loop(M_temp_3)
    # count = 0
    while(len(temp_loop)):
        M_temp_3 = value_eliminate_loop(M_temp_3, temp_loop)
        temp_loop = find_loop(M_temp_3)
    time_3 = time.time()

    return [M_temp_1, M_temp_2, M_temp_3, (time_1 - time_0), (time_2 - time_1), (time_3 - time_2)]