import numpy as np
from sklearn.linear_model import LinearRegression
# from lingam.hsic import get_gram_matrix, get_kernel_width, hsic_test_gamma, hsic_teststat
from lingam.utils import predict_adaptive_lasso


# xi = alpha * xj
def get_residual(xi, xj): 
    return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj


def get_residual_sk(xi, xj): 
    reg = LinearRegression(fit_intercept=False)
    reg.fit(xj, xi)
    x_i_hat = reg.predict(xj)
    return xi - x_i_hat


def entropy(u):
    k1 = 79.047
    k2 = 7.4129
    gamma = 0.37457
    return (1 + np.log(2 * np.pi)) / 2 - k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2 - k2 * (np.mean(u * np.exp((-(u ** 2)) / 2))) ** 2


def diff_mutual_info(xi_std, xj_std, ri_j, rj_i):
    return (entropy(xj_std) + entropy(ri_j / np.std(ri_j))) - (entropy(xi_std) + entropy(rj_i / np.std(rj_i)))


def search_causal_order(X, U):
    Uc = U.copy()
    if len(Uc) == 1:
        return Uc[0]
    M_list = []
    for i in Uc:
        M = 0
        for j in U:
            if i != j:
                xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
                ri_j = get_residual(xi_std, xj_std)
                rj_i = get_residual(xj_std, xi_std)
                M += np.min([0, diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
        M_list.append(-1.0 * M)
    return Uc[np.argmax(M_list)]


def myDirectLiNGAM(X):
    n_features = len(X[0])

    U = np.arange(n_features)
    K = list()
    X_ = np.copy(X)
    for _ in range(n_features):
        m = search_causal_order(X_, U)

        for i in U:
            if i != m:
                X_[:, i] = get_residual(X_[:, i], X_[:, m])
        K.append(m)

        U = U[U != m] # 除去U中的m
    causal_order = K

    # print(causal_order)

    B = np.zeros([n_features, n_features], dtype="float64")

    for i in range(1, len(causal_order)):
        target = causal_order[i]
        predictors = causal_order[:i]

        if len(predictors) == 0:
            continue

        B[target, predictors] = predict_adaptive_lasso(X, predictors, target)

    return B