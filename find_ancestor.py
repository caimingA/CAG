import numpy as np
import copy
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import lingam
from lingam.hsic import hsic_test_gamma
# from hyppo.conditional import FCIT
# from hyppo.conditional import KCI
from causallearn.utils.cit import CIT

import lingam_local

import networkx as nx
# from causallearn.search.ConstraintBased.PC import pc
import pandas as pd
import statsmodels.api as sm
import time

import itertools

# from hyppo.independence import Hsic

import warnings
warnings.filterwarnings("ignore")


# xi = alpha * xj
def get_residual(xi, xj):
    return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj


def is_independent(X, Y, alpha):
    _, p = hsic_test_gamma(X, Y, bw_method="mdbs")
    return p > alpha


def is_linear(X, Y, alpha):
    # _, p = stats.spearmanr(X, Y)
    _, p = stats.pearsonr(X, Y)
#     print(p)
    return p < alpha


def is_linear_spearmanr(X, Y, alpha):
    _, p = stats.pearsonr(X, Y)
    # _, p = stats.spearmanr(X, Y)
#     print(p)
    return p < alpha


def is_linear_tau(X, Y, alpha):
    _, p = stats.kendalltau(X, Y)
#     print(p)
    return p < alpha


def make_unfinished(data):
    n_features = len(data[0])
    unfinished_list = list()
    for i in range(n_features):
        for j in range(i+1, n_features):
            unfinished_list.append([i, j])
    return unfinished_list


def quick_ancestor(i, j, ancestor_dict):
    for a in ancestor_dict[i]:
        if j in ancestor_dict[a]:
            ancestor_dict[i].append(j)
            print("quick finding!: ", j, " --> ", i)
            return True

    for a in ancestor_dict[j]:
        if i in ancestor_dict[a]:
            ancestor_dict[j].append(i)
            print("quick finding!: ", i, " --> ", j)
            return True 
    return False


def get_ancestor_loop_HSIC(data, ancestor_dict, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
#     l_alpha = 0.01
#     i_alpha = 0.01
#     i_alpha_U = 0.01
#     p_alpha = 0.001
    num = len(data[0])
    unfinished_list = make_unfinished(data)

    flag_anc = True
    flag_unf = True

    loop_count = 0
    
    while flag_anc or flag_unf:
        flag_anc = False
        flag_unf = False

        unfinished_list_temp = copy.deepcopy(unfinished_list)

        for c in unfinished_list_temp:
            
            i = c[0]
            j = c[1]
            
            flag_temp = False

            flag_temp = quick_ancestor(i, j, ancestor_dict)
            if flag_temp:
                flag_anc = True
                flag_unf = True
                unfinished_list.remove(c)
            else:            
                UC = list(set(ancestor_dict[i]).intersection(set(ancestor_dict[j]))) # 交集
                if len(UC) == 0:                    
                    if is_linear(data[:, i], data[:, j], l_alpha):
                        
                        # xi_std = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
                        # xj_std = (data[:, j] - np.mean(data[:, j])) / np.std(data[:, j])
                        # ri_j = get_residual(xi_std, xj_std) # ri_j = xi - alpha * xj
                        # rj_i = get_residual(xj_std, xi_std) # rj_i = xj - alpha * xi

                        # _, pi_j = hsic_test_gamma(ri_j, xj_std, bw_method="mdbs") # xj -> xi
                        # _, pj_i = hsic_test_gamma(rj_i, xi_std, bw_method="mdbs") # xi -> xj

                        xi_std = data[:, i]
                        xj_std = data[:, j]

                        reg = LinearRegression(fit_intercept=False)

                        # x_data_reshaped = x_data.reshape(-1, 1)
                        # y_data_reshaped = y_data.reshape(-1, 1)
                        
                        res = reg.fit(data[:, j].reshape(-1, 1), data[:, i]) # x, y
                        coef = res.coef_
                        ri_j = data[:, i] - coef * data[:, j] # ri_j = xi - alpha * xj

                        res = reg.fit(data[:, i].reshape(-1, 1), data[:, j]) # x, y
                        coef = res.coef_
                        rj_i = data[:, j] - coef * data[:, i]# rj_i = xj - alpha * xi

                        # print(ri_j.shape)
                        
                        xi_std = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
                        xj_std = (data[:, j] - np.mean(data[:, j])) / np.std(data[:, j])
                        
                        # _, pi_j = Hsic().test(ri_j, xj_std, auto=True) # xj -> xi
                        # _, pj_i = Hsic().test(rj_i, xi_std, auto=True) # xi -> xj


                        if pi_j > i_alpha and pj_i <= i_alpha:
                            if True:
                                ancestor_dict[i].append(j)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True
                            else:
                                continue
                        if pi_j <= i_alpha and pj_i > i_alpha:
                            if True:
                                ancestor_dict[j].append(i)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True                                
                            else:
                                continue
                        if pi_j > i_alpha and pj_i > i_alpha:
                            unfinished_list.remove(c)
                            flag_anc = True
                            flag_unf = True
                            # continue
                        if pi_j <= i_alpha and pj_i <= i_alpha:
                            continue        
                    else:
                        if loop_count >= 0: ##########################
                            flag_anc = True
                            flag_unf = True
                            unfinished_list.remove(c)
                        else:
                            continue
            
                else:
                    X = list()
                    for index in UC:
                        X.append(data[:, index])
                    
                    X = np.array(X).T
                    
                    reg = LinearRegression(fit_intercept=False)
                    res = reg.fit(X, data[:, i])
                    coef = res.coef_
                    # coef = np.linalg.lstsq(X, data[:, i], rcond=None) # X, y
                    yi = data[:, i] - np.dot(coef, X.T)
                    # yi = data[:, i] - reg.predict(X)

                    res = reg.fit(X, data[:, j])
                    coef = res.coef_
                    # coef = np.linalg.lstsq(X, data[:, j], rcond=None) # X, y
                    yj = data[:, j] - np.dot(coef, X.T)
                    # yj = data[:, j] - reg.predict(X)
                    
                    # if not is_independent(yi, yj, 0.01):

                    if is_linear(yi, yj, l_alpha):
                        # yi_std = (yi - np.mean(yi)) / np.std(yi)
                        # yj_std = (yj - np.mean(yj)) / np.std(yj)
                        # ri_j = get_residual(yi_std, yj_std) # ri_j = yi - alpha * yj
                        # rj_i = get_residual(yj_std, yi_std) # rj_i = yj - alpha * yi

                        # _, pi_j = hsic_test_gamma(ri_j, yj_std, bw_method="mdbs") # yj -> yi
                        # _, pj_i = hsic_test_gamma(rj_i, yi_std, bw_method="mdbs") # yi -> yj
#                         _, pi_j = Hsic().test(ri_j, xj_std) # xj -> xi
#                         _, pj_i = Hsic().test(rj_i, xi_std) # xi -> xj
                        
                        yi_std = yi
                        yj_std = yj

                        reg = LinearRegression(fit_intercept=False)

                        res = reg.fit(yj.reshape(-1, 1), yi) # x, y
                        coef = res.coef_
                        ri_j = yi - coef * yj # ri_j = xi - alpha * xj

                        res = reg.fit(yi.reshape(-1, 1), yj) # x, y
                        coef = res.coef_
                        rj_i = yj - coef * yi# rj_i = xj - alpha * xi
                        # _, pi_j = Hsic().test(ri_j, yj_std, auto=True) # yj -> yi
                        # _, pj_i = Hsic().test(rj_i, yi_std, auto=True) # yi -> yj

                        _, pi_j = hsic_test_gamma(ri_j, yj_std, bw_method="mdbs") # yj -> yi
                        _, pj_i = hsic_test_gamma(rj_i, yi_std, bw_method="mdbs") # yi -> yj

                        
                        if pi_j > i_alpha_U and pj_i <= i_alpha_U:
                            if True:
                            # if (pi_j / pj_i < p_alpha) or (pj_i / pi_j < p_alpha):
                                ancestor_dict[i].append(j)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True                                
                                
                            else:
                                continue
                        if pi_j <= i_alpha_U and pj_i > i_alpha_U:
                            if True:    
                            # if (pi_j / pj_i < p_alpha) or (pj_i / pi_j < p_alpha):
                                ancestor_dict[j].append(i)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True                                
                                
                            else:
                                continue
                        if pi_j > i_alpha_U and pj_i > i_alpha_U:
                            # unfinished_list.remove(c)
                            # flag_anc = True
                            # flag_unf = True
                            continue
                        if pi_j <= i_alpha_U and pj_i <= i_alpha_U:
                            continue
                    else:
                        flag_anc = True
                        flag_unf = True
                        unfinished_list.remove(c) 
        loop_count += 1

def get_ancestor_loop_KCI(data, ancestor_dict, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
    num = len(data[0])
    unfinished_list = make_unfinished(data)

    flag_anc = True
    flag_unf = True

    loop_count = 0
    
    while flag_anc or flag_unf:
        flag_anc = False
        flag_unf = False

        unfinished_list_temp = copy.deepcopy(unfinished_list)

        for c in unfinished_list_temp:
            
            i = c[0]
            j = c[1]
            
            flag_temp = False

            flag_temp = quick_ancestor(i, j, ancestor_dict)
            if flag_temp:
                flag_anc = True
                flag_unf = True

                unfinished_list.remove(c)
            else:            
                UC = list(set(ancestor_dict[i]).intersection(set(ancestor_dict[j]))) # 交集
                if len(UC) == 0:                    
                    if is_linear(data[:, i], data[:, j], l_alpha):
                        # xi_std = data[:, i]
                        # xj_std = data[:, j]
                        # xi_std = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
                        # xj_std = (data[:, j] - np.mean(data[:, j])) / np.std(data[:, j])
                        # ri_j = get_residual(xi_std, xj_std) # ri_j = xi - alpha * xj
                        # rj_i = get_residual(xj_std, xi_std) # rj_i = xj - alpha * xi

                        xi_std = data[:, i]
                        xj_std = data[:, j]

                        reg = LinearRegression(fit_intercept=False)

                        res = reg.fit(data[:, j].reshape(-1, 1), data[:, i]) # x, y
                        coef = res.coef_
                        ri_j = data[:, i] - coef * data[:, j] # ri_j = xi - alpha * xj

                        res = reg.fit(data[:, i].reshape(-1, 1), data[:, j]) # x, y
                        coef = res.coef_
                        rj_i = data[:, j] - coef * data[:, i]# rj_i = xj - alpha * xi
                        
                        data_set = np.array([ri_j, xj_std, rj_i, xi_std])
                        
                        kci_obj = CIT(data_set.T, "kci")
                        # pi_j = kci_obj(ri_j, xj_std, [])
                        # pj_i = kci_obj(rj_i, xi_std, [])
                        pi_j = kci_obj(0, 1, [])
                        pj_i = kci_obj(2, 3, [])

                        # print("(", i, ", ", j, ")", "pi_j, pj_i:", pi_j, "----", pj_i)
                        # _, pi_j = hsic_test_gamma(ri_j, xj_std, bw_method="mdbs") # xj -> xi
                        # _, pj_i = hsic_test_gamma(rj_i, xi_std, bw_method="mdbs") # xi -> xj

                        if pi_j > i_alpha and pj_i <= i_alpha:
                            if True:
                                ancestor_dict[i].append(j)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True
                            else:
                                continue
                        if pi_j <= i_alpha and pj_i > i_alpha:
                            if True:
                                ancestor_dict[j].append(i)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True                                
                            else:
                                continue
                        if pi_j > i_alpha and pj_i > i_alpha:
                            unfinished_list.remove(c)
                            flag_anc = True
                            flag_unf = True
                            # continue
                        if pi_j <= i_alpha and pj_i <= i_alpha:
                            continue        
                    else:
                        if loop_count >= 0: ##########################
                            flag_anc = True
                            flag_unf = True
                            unfinished_list.remove(c)
                        else:
                            continue
            
                else:
                    X = list()
                    for index in UC:
                        X.append(data[:, index])
                    
                    X = np.array(X).T
                    
                    reg = LinearRegression(fit_intercept=False)
                    res = reg.fit(X, data[:, i])
                    coef = res.coef_
                    # coef = np.linalg.lstsq(X, data[:, i], rcond=None) # X, y
                    yi = data[:, i] - np.dot(coef, X.T)
                    # yi = data[:, i] - reg.predict(X)

                    res = reg.fit(X, data[:, j])
                    coef = res.coef_
                    # coef = np.linalg.lstsq(X, data[:, j], rcond=None) # X, y
                    yj = data[:, j] - np.dot(coef, X.T)
                    # yj = data[:, j] - reg.predict(X)
                    
                    # if not is_independent(yi, yj, 0.01):

                    if is_linear(yi, yj, l_alpha):
                        # yi_std = yi
                        # yj_std = yj
                        # yi_std = (yi - np.mean(yi)) / np.std(yi)
                        # yj_std = (yj - np.mean(yj)) / np.std(yj)
                        # ri_j = get_residual(yi_std, yj_std) # ri_j = yi - alpha * yj
                        # rj_i = get_residual(yj_std, yi_std) # rj_i = yj - alpha * yi

                        yi_std = yi
                        yj_std = yj

                        reg = LinearRegression(fit_intercept=False)

                        res = reg.fit(yj.reshape(-1, 1), yi) # x, y
                        coef = res.coef_
                        ri_j = yi - coef * yj # ri_j = xi - alpha * xj

                        res = reg.fit(yi.reshape(-1, 1), yj) # x, y
                        coef = res.coef_
                        rj_i = yj - coef * yi# rj_i = xj - alpha * xi
                        # _, pi_j = hsic_test_gamma(ri_j, yj_std, bw_method="mdbs") # yj -> yi
                        # _, pj_i = hsic_test_gamma(rj_i, yi_std, bw_method="mdbs") # yi -> yj

                        
                        data_set = np.array([ri_j, yj_std, rj_i, yi_std])
                        kci_obj = CIT(data_set.T, "kci")
                        pi_j = kci_obj(0, 1, [])
                        pj_i = kci_obj(2, 3, [])
                        # pi_j = kci_obj(ri_j, yj_std, [])
                        # pj_i = kci_obj(rj_i, yi_std, [])
                        # print("(", i, ", ", j, ")", "pi_j, pj_i:", pi_j, "----", pj_i)
#                         _, pi_j = Hsic().test(ri_j, xj_std) # xj -> xi
#                         _, pj_i = Hsic().test(rj_i, xi_std) # xi -> xj
                        

                        if pi_j > i_alpha_U and pj_i <= i_alpha_U:
                            if True:
                            # if (pi_j / pj_i < p_alpha) or (pj_i / pi_j < p_alpha):
                                ancestor_dict[i].append(j)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True                                
                                
                            else:
                                continue
                        if pi_j <= i_alpha_U and pj_i > i_alpha_U:
                            if True:    
                            # if (pi_j / pj_i < p_alpha) or (pj_i / pi_j < p_alpha):
                                ancestor_dict[j].append(i)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True                                
                                
                            else:
                                continue
                        if pi_j > i_alpha_U and pj_i > i_alpha_U:
                            # unfinished_list.remove(c)
                            # flag_anc = True
                            # flag_unf = True
                            continue
                        if pi_j <= i_alpha_U and pj_i <= i_alpha_U:
                            continue
                    else:
                        flag_anc = True
                        flag_unf = True
                        unfinished_list.remove(c) 
        loop_count += 1


def get_ancestor_loop_KCI_2(data, ancestor_dict, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
    num = len(data[0])
    record_matrix = np.zeros((num, num))
    loop_flag = True

    while loop_flag:
        loop_flag = False

        for i in range(0, num):
            for j in range(i + 1, num):
                print(i, " and ", j)
                if record_matrix[i][j] == 0:
                    quick_flag = quick_ancestor(i, j, ancestor_dict)
                    if quick_flag:
                        loop_flag = True
                        record_matrix[i][j] = 1
                        record_matrix[j][i] = 1
                        continue
                    UC = list(set(ancestor_dict[i]).intersection(set(ancestor_dict[j])))
                    if len(UC) == 0:
                        if is_linear(data[:, i], data[:, j], l_alpha):
                            xi_std = data[:, i]
                            xj_std = data[:, j]

                            reg = LinearRegression(
                                copy_X=True
                                , fit_intercept=False
                                ).fit(
                                    data[:, i].reshape(-1, 1)
                                    , data[:, j]
                                    ) # x, y
                            rj_i = data[:, j] - reg.predict(data[:, i].reshape(-1, 1)) # rj_i = xj - alpha * xi

                            reg = LinearRegression(
                                copy_X=True
                                , fit_intercept=False
                                ).fit(
                                    data[:, j].reshape(-1, 1)
                                    , data[:, i]
                                    ) # x, y
                            ri_j = data[:, i] - reg.predict(data[:, j].reshape(-1, 1)) # ri_j = xi - alpha * xj
                            
                            data_set = np.array([ri_j, xj_std, rj_i, xi_std])
                            kci_obj = CIT(data_set.T, "kci")
                    
                            pi_j = kci_obj(0, 1, [])
                            pj_i = kci_obj(2, 3, [])

                            if pi_j > i_alpha and pj_i <= i_alpha:
                                ancestor_dict[i].append(j)
                                loop_flag = True
                                record_matrix[i][j] = 1
                                record_matrix[j][i] = 1
                            if pi_j <= i_alpha and pj_i > i_alpha:
                                ancestor_dict[j].append(i)
                                loop_flag = True    
                                record_matrix[i][j] = 1
                                record_matrix[j][i] = 1
                            if pi_j > i_alpha and pj_i > i_alpha:
                                if pi_j > pi_j:
                                    ancestor_dict[i].append(j)
                                    loop_flag = True
                                    record_matrix[i][j] = 1
                                    record_matrix[j][i] = 1
                                else:
                                    ancestor_dict[j].append(i)
                                    loop_flag = True
                                    record_matrix[i][j] = 1
                                    record_matrix[j][i] = 1
                        else:
                            loop_flag = True
                            record_matrix[i][j] = 1
                            record_matrix[j][i] = 1
                    else:
                        X = list()
                        for index in UC:
                            X.append(data[:, index])                    
                        X = np.array(X).T
                        reg = LinearRegression(
                            copy_X=True
                            , fit_intercept=False
                            ).fit(
                                X
                                , data[:, i]
                                )
                        yi = data[:, i] - reg.predict(X)

                        reg = LinearRegression(
                            copy_X=True
                            , fit_intercept=False
                            ).fit(
                                X
                                , data[:, j]
                                )
                        yj = data[:, j] - reg.predict(X)

                        if is_linear(yi, yj, l_alpha):
                            yi_std = yi
                            yj_std = yj

                            reg = LinearRegression(
                                copy_X=True
                                , fit_intercept=False
                                ).fit(yj.reshape(-1, 1), yi) # x, y
                            ri_j = yi - reg.predict(yj.reshape(-1, 1)) # ri_j = xi - alpha * xj

                            reg = LinearRegression(
                                copy_X=True
                                , fit_intercept=False
                                ).fit(yi.reshape(-1, 1), yj) # x, y
                            rj_i = yj - reg.predict(yi.reshape(-1, 1)) # rj_i = xj - alpha * xi

                            data_set = np.array([ri_j, yj_std, rj_i, yi_std])
                            kci_obj = CIT(data_set.T, "kci")
                            
                            pi_j = kci_obj(0, 1, [])
                            pj_i = kci_obj(2, 3, [])

                            if pi_j > i_alpha and pj_i <= i_alpha:
                                ancestor_dict[i].append(j)
                                loop_flag = True
                                record_matrix[i][j] = 1
                                record_matrix[j][i] = 1                             
                            if pi_j <= i_alpha and pj_i > i_alpha:
                                ancestor_dict[j].append(i)                                
                                loop_flag = True
                                record_matrix[i][j] = 1
                                record_matrix[j][i] = 1     
                            if pi_j > i_alpha and pj_i > i_alpha:
                                if pi_j > pi_j:
                                    ancestor_dict[i].append(j)
                                    loop_flag = True
                                    record_matrix[i][j] = 1
                                    record_matrix[j][i] = 1
                                else:
                                    ancestor_dict[j].append(i)
                                    loop_flag = True
                                    record_matrix[i][j] = 1
                                    record_matrix[j][i] = 1
                        else:
                            loop_flag = True
                            record_matrix[i][j] = 1
                            record_matrix[j][i] = 1
    print(record_matrix)


def get_ancestor_loop_KCI_3(data, ancestor_dict, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
    num = len(data[0])
    record_matrix = np.zeros((num, num))
    rcd_model = lingam_local.RCD(max_explanatory_num=1, shapiro_alpha=1, cor_alpha=l_alpha, ind_alpha=i_alpha)
    rcd_model.fit(data)
    a_dict = rcd_model.ancestors_list_
    for i in ancestor_dict.keys():
        ancestor_dict[i] = list(a_dict[i])
    # print(record_matrix)


def final_ancestor(unfinished_list, data, ancestor_dict, i_alpha, i_alpha_U):
    for item in unfinished_list:
        i = item[0]
        j = item[1]

        UC = list(set(ancestor_dict[i]).intersection(set(ancestor_dict[j])))
        # print(UC)
        if len(UC) == 0:
            xi_std = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
            xj_std = (data[:, j] - np.mean(data[:, j])) / np.std(data[:, j])
            ri_j = get_residual(xi_std, xj_std) # ri_j = xi - alpha * xj
            rj_i = get_residual(xj_std, xi_std) # rj_i = xj - alpha * xi

            _, pi_j = hsic_test_gamma(ri_j, xj_std, bw_method="mdbs") # xj -> xi
            _, pj_i = hsic_test_gamma(rj_i, xi_std, bw_method="mdbs") # xi -> xj

            # print(item, " and ", pi_j, " vs. ", pj_i)
            if pi_j <= i_alpha and pj_i > i_alpha:
                if pi_j > pj_i:
                    ancestor_dict[i].append(j)
                if pi_j <= pj_i:
                    ancestor_dict[j].append(i)
        else:
            X = list()
            for index in UC:
                X.append(data[:, index])
            
            X = np.array(X).T
            
            reg = LinearRegression(fit_intercept=False)
            res = reg.fit(X, data[:, i])
            coef = res.coef_
            # coef = np.linalg.lstsq(X, data[:, i], rcond=None) # X, y
            yi = data[:, i] - np.dot(coef, X.T)
            # yi = data[:, i] - reg.predict(X)
            res = reg.fit(X, data[:, j])
            coef = res.coef_
            # coef = np.linalg.lstsq(X, data[:, j], rcond=None) # X, y
            yj = data[:, j] - np.dot(coef, X.T)
            yi_std = (yi - np.mean(yi)) / np.std(yi)
            yj_std = (yj - np.mean(yj)) / np.std(yj)
            ri_j = get_residual(yi_std, yj_std) # ri_j = yi - alpha * yj
            rj_i = get_residual(yj_std, yi_std) # rj_i = yj - alpha * yi
                
            _, pi_j = hsic_test_gamma(ri_j, yj_std, bw_method="mdbs") # yj -> yi
            _, pj_i = hsic_test_gamma(rj_i, yi_std, bw_method="mdbs") # yi -> yj

            # print(item, " and ", pi_j, " vs. ", pj_i)
            if pi_j > i_alpha_U and pj_i <= i_alpha_U:
                if pi_j > pj_i:
                    ancestor_dict[i].append(j)
                        # unfinished_list.remove(c)
                if pi_j <= pj_i:
                    ancestor_dict[j].append(i)


def get_anc_dict(Ancestor_list):
    ancestor_dict_right = dict()

    for i in range(len(Ancestor_list)):
        temp = Ancestor_list[i]
        temp.append(i)
        ancestor_dict_right[i] = temp

    return ancestor_dict_right


# b is the subset of a
# def is_subset(a, b):
#     for i in b:
#         if i in a:
#             continue
#         else:
#             return False
#     return True 

def is_subset(list1, list2):
    """
    Check if list2 is a subset of list1.
    
    Args:
        list1 (list): The potential superset
        list2 (list): The potential subset
        
    Returns:
        bool: True if list2 is a subset of list1, False otherwise
    """
    set1 = set(list1)
    set2 = set(list2)
    return set2.issubset(set1)


# def get_group(ancestor_dict, sample_size):
#     flag = True
#     if sample_size <= 100:
#         flag = False


#     print(flag)
#     temp = list()
#     for i in ancestor_dict.values():
#         temp.append(sorted(i))
#     # print(temp)
#     temp = sorted(temp, key = lambda i:len(i), reverse=True)
#     # print(temp)

#     # print(is_subset(temp[0], temp[1]))

#     record = [0 for i in range(len(temp))]
    
#     graphs = list()
#     # print(record)
#     for i in range(len(temp)):
#         for j in range(i + 1, len(temp)):
#             if(is_subset(temp[i], temp[j])): # 判断temp[j]是否为temp[i]的子集
#                 record[j] += 1
#         # print(record)

#     if flag == True:    
#         for i in range(len(temp)):
#             if record[i] == 0:
#                 graphs.append(temp[i])
#     else:
# #         print(temp)
#         for i in range(len(temp)):
#             if record[i] == 0: # 不是任何人的子集
#                 if len(temp[i]) >= 2:
#                     graphs.append(temp[i])
#                 else:
#                     for j in range(len(graphs)):
#                         graphs[j] += temp[i]
#                         graphs[j] = list(set(graphs[j]))

#             # if record[i] == 0 and len(temp[i]) >= 2: 
#             #     graphs.append(temp[i])
#             # else:
#             #     if record[i] == 0:
#             #         for j in range(len(graphs)):
#             #             graphs[j] += temp[i]
#             #             graphs[j] = list(set(graphs[j]))

#     print(graphs)
#     if len(graphs) == 0 or len(graphs) == 1:
#     # if len(graphs) == 0:
        
#         if sample_size / len(ancestor_dict) < 2:
#             graphs = []
#             for i in range(len(ancestor_dict)):
#                 for j in range(i+1, len(ancestor_dict)):
#                     graphs.append([i, j])
#         else:
#             graphs = [[i for i in range(len(ancestor_dict))]]      
#     print(graphs)

#     print("******************")
#     return graphs


# def get_group(ancestor_dict, sample_size):
#     temp = list()
#     for i in ancestor_dict.values():
#         temp.append(sorted(i))
#     # print(temp)
#     temp = sorted(temp, key = lambda i:len(i), reverse=True)
#     # print(temp)

#     # print(is_subset(temp[0], temp[1]))

#     record = [0 for i in range(len(temp))]
    
#     graphs_0 = list()
#     # print(record)
#     for i in range(len(temp)):
#         for j in range(i + 1, len(temp)):
#             if(is_subset(temp[i], temp[j])): # 判断temp[j]是否为temp[i]的子集
#                 record[j] += 1
#         # print(record)

#     for i in range(len(temp)):
#         if record[i] == 0:
#             graphs_0.append(temp[i])
    
#     groups = list()
#     for i in range(len(graphs_0)):
#         if len(graphs_0[i]) == 1:
#             for j in range(i + 1, len(graphs_0)):
#                 groups.append(list(set(graphs_0[i]).union(set(graphs_0[j]))))
#         else:
#             groups.append(graphs_0[i])
# #     if flag == True:    
# #         for i in range(len(temp)):
# #             if record[i] == 0:
# #                 graphs.append(temp[i])
# #     else:
# # #         print(temp)
# #         for i in range(len(temp)):
# #             if record[i] == 0: # 不是任何人的子集
# #                 if len(temp[i]) >= 2:
# #                     graphs.append(temp[i])
# #                 else:
# #                     for j in range(len(graphs)):
# #                         graphs[j] += temp[i]
# #                         graphs[j] = list(set(graphs[j]))

# #             # if record[i] == 0 and len(temp[i]) >= 2: 
# #             #     graphs.append(temp[i])
# #             # else:
# #             #     if record[i] == 0:
# #             #         for j in range(len(graphs)):
# #             #             graphs[j] += temp[i]
# #             #             graphs[j] = list(set(graphs[j]))

# #     print(graphs)
#     # if len(graphs) == 0 or len(graphs) == 1:
#     # # if len(graphs) == 0:
        
#     #     if sample_size / len(ancestor_dict) < 2:
#     #         graphs = []
#     #         for i in range(len(ancestor_dict)):
#     #             for j in range(i+1, len(ancestor_dict)):
#     #                 graphs.append([i, j])
#     #     else:
#     #         graphs = [[i for i in range(len(ancestor_dict))]]      
#     print(groups)

#     print("******************")
#     return groups


def get_group(ancestor_dict, sample_size):
    """
    Process groups from an ancestor dictionary and generate combined groups based on subset relationships.
    
    Args:
        ancestor_dict (dict): Dictionary containing ancestor relationships
        sample_size (int): Size of the sample to process
        
    Returns:
        list: List of processed groups
    """
    # Convert dictionary values to sorted lists
    temp = [sorted(group) for group in ancestor_dict.values()]
    
    # Sort by length in descending order
    temp = sorted(temp, key=len, reverse=True)
    
    # Initialize record to track subset relationships
    record = [0] * len(temp)
    
    # Find subset relationships
    for i in range(len(temp)):
        for j in range(i + 1, len(temp)):
            if record[i] != 0 or record[j] != 0:
                continue
            else:
                if is_subset(temp[i], temp[j]):
                    record[j] += 1
    
    # Collect groups with no supersets (record[i] == 0)
    # graphs_0 = [temp[i] for i in range(len(temp)) if record[i] == 0]
    # graphs_0_1 = [temp[i] for i in range(len(temp)) if record[i] == 0]
    groups_0 = []
    groups_0_1 = []
    for i in range(len(temp)):
        if record[i] == 0:
            groups_0.append(temp[i])
        if record[i] == 0 and len(temp[i]) == 1:
            groups_0_1.append(temp[i])

    groups = []
    groups_set = []
    for pair in itertools.product(groups_0, groups_0_1):
        if pair[0] != pair[1] and set(pair[0] + pair[1]) not in groups_set:
            groups_set.append(set(pair[0] + pair[1]))
            groups.append(pair[0] + pair[1])
            # print(i[0] + i[1])


    # # Generate final groups
    # groups = []
    # for i in range(len(graphs_0)):
    #     if len(graphs_0[i]) == 1:
    #         # For single-element groups, combine with all subsequent groups
    #         for j in range(i + 1, len(graphs_0)):
    #             combined = list(set(graphs_0[i]).union(set(graphs_0[j])))
    #             groups.append(combined)
    #     else:
    #         # Add multi-element groups as is
    #         groups.append(graphs_0[i])
    
    print(groups)
    
    return groups


def get_res(data, groups, B, ancestor_list):
    M = np.zeros_like(B)
    visit = np.zeros_like(B)
    
    N_num = len(M)
    
    for g in groups:
        print("g =", g )
        if len(g) == 1 or len(g) == 0:
            continue
        else:
            n_num = len(g)
            
            X = list()
            for index in g:
                X.append(data[:, index])            
            X = np.array(X).T
                
            model = lingam.DirectLiNGAM()
            model.fit(X)
            m = model.adjacency_matrix_

            # print(m)
            
            for i in range(len(m)):
                for j in range(len(m)):
                    visit[g[i]][g[j]] += 1
                    if(m[i][j] != 0):
                        # print(m[i][j])
                        M[g[i]][g[j]] += m[i][j]
                    # print(M)
    # for i in range(len(M)):
    #     for j in range(len(M)):
    #         if(M[i][j]):
    #             M[i][j] /= visit[i][j]

    # print("final", M)
    # print("******************")
    ###### based on wald test
    time_0 = time.process_time()
    M_temp_1 = copy.deepcopy(M)
    temp_loop = find_loop(M_temp_1)
    # count = 0
    while(len(temp_loop)):
        M_temp_1 = wald_eliminate_loop(M_temp_1, temp_loop, data)
        temp_loop = find_loop(M_temp_1)
    
    
    ###### based on ancestors
    time_1 = time.process_time()
    M_temp_2 = copy.deepcopy(M)
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
        # print("a")

    ###### based on value
    time_2 = time.process_time()
    M_temp_3 = copy.deepcopy(M)
    temp_loop = find_loop(M_temp_3)
    # count = 0
    while(len(temp_loop)):
        M_temp_3 = value_eliminate_loop(M_temp_3, temp_loop)
        temp_loop = find_loop(M_temp_3)
    time_3 = time.process_time()

    return [M_temp_1, M_temp_2, M_temp_3, (time_1 - time_0), (time_2 - time_1), (time_3 - time_2)]


def get_res_no_ancestor(data, groups, B):
    M = np.zeros_like(B)
    visit = np.zeros_like(B)
    
    N_num = len(M)
    
    for group in groups:
        # print("g =", g )
        if len(group) == 1 or len(group) == 0:
            continue
        else:
            g = list(group)
            n_num = len(g)
            
            X = list()
            for index in g:
                X.append(data[:, index])            
            X = np.array(X).T
                
            model = lingam.DirectLiNGAM()
            model.fit(X)
            m = model.adjacency_matrix_
            
            for i in range(len(m)):
                for j in range(len(m)):
                    visit[g[i]][g[j]] += 1
                    if(m[i][j] != 0):
                        M[g[i]][g[j]] += m[i][j]
    for i in range(len(M)):
        for j in range(len(M)):
            if(M[i][j]):
                M[i][j] /= visit[i][j]

    # print("******************")
    ###### based on wald test
    time_0 = time.process_time()
    M_temp_1 = copy.deepcopy(M)
    temp_loop = find_loop(M)
    # count = 0
    while(len(temp_loop)):
        M_temp_1 = wald_eliminate_loop(M_temp_1, temp_loop, data)
        temp_loop = find_loop(M_temp_1)
    
    
    ###### based on ancestors
    # time_1 = time.time()
    # M_temp_2 = copy.deepcopy(M)
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
        # print("a")

    ###### based on value
    time_2 = time.process_time()
    M_temp_3 = copy.deepcopy(M)
    temp_loop = find_loop(M_temp_3)
    # count = 0
    while(len(temp_loop)):
        M_temp_3 = value_eliminate_loop(M_temp_3, temp_loop)
        temp_loop = find_loop(M_temp_3)
    time_3 = time.process_time()

    return [M_temp_1, M_temp_3, (time_2 - time_0), (time_3 - time_2)]


def get_res_simple(data, groups, B):
    M = np.zeros_like(B)
    visit = np.zeros_like(B)
    
    N_num = len(M)
    
    for g in groups:
        # print("g =", g )
        n_num = len(g)
        
        X = list()
        for index in g:
            X.append(data[:, index])            
        X = np.array(X).T
            
        model = lingam.DirectLiNGAM()
        model.fit(X)
        m = model.adjacency_matrix_
        
        for i in range(len(m)):
            for j in range(len(m)):
                visit[g[i]][g[j]] += 1
                if(m[i][j] != 0):
                    M[g[i]][g[j]] += m[i][j]
    for i in range(len(M)):
        for j in range(len(M)):
            if(M[i][j]):
                M[i][j] /= visit[i][j]

    # print("******************")
    ###### based on wald test
    # time_0 = time.time()
    M_temp_1 = copy.deepcopy(M)
    temp_loop = find_loop(M)
    # count = 0
    while(len(temp_loop)):
        M_temp_1 = wald_eliminate_loop(M_temp_1, temp_loop, data)
        temp_loop = find_loop(M_temp_1)
    
    # time_1 = time.time()

    return M_temp_1


# def get_res_anc(data, groups, B, ancestor_list):
#     M = np.zeros_like(B)
#     visit = np.zeros_like(B)
    
#     N_num = len(M)
    
#     for g in groups:
#         # print("g =", g )
#         n_num = len(g)
        
#         X = list()
#         for index in g:
#             X.append(data[:, index])            
#         X = np.array(X).T
            
#         model = lingam.DirectLiNGAM()
#         model.fit(X)
#         m = model.adjacency_matrix_
        
#         for i in range(len(m)):
#             for j in range(len(m)):
#                 visit[g[i]][g[j]] += 1
#                 if(m[i][j] != 0):
#                     M[g[i]][g[j]] += m[i][j]
#     for i in range(len(M)):
#         for j in range(len(M)):
#             if(M[i][j]):
#                 M[i][j] /= visit[i][j]

#     ###### based on ancestors
#     # M = copy.deepcopy(M)
#     temp_loop = find_loop(M)
#     count = 0
#     while(len(temp_loop)):
#         # print("loop!")
#         # M = value_eliminate_loop(M, temp_loop)
#         if count == 0:
#             M = ancestor_eliminate_loop(M, temp_loop, ancestor_list)
#         else:
#             M = value_eliminate_loop(M, temp_loop)
#         # M = wald_eliminate_loop(M, temp_loop, data)
#         temp_loop = find_loop(M)
#         count += 1

#     return M


# def get_res_value(data, groups, B, ancestor_list):
#     M = np.zeros_like(B)
#     visit = np.zeros_like(B)
    
#     N_num = len(M)
    
#     for g in groups:
#         # print("g =", g )
#         n_num = len(g)
        
#         X = list()
#         for index in g:
#             X.append(data[:, index])            
#         X = np.array(X).T
            
#         model = lingam.DirectLiNGAM()
#         model.fit(X)
#         m = model.adjacency_matrix_
        
#         for i in range(len(m)):
#             for j in range(len(m)):
#                 visit[g[i]][g[j]] += 1
#                 if(m[i][j] != 0):
#                     M[g[i]][g[j]] += m[i][j]
#     for i in range(len(M)):
#         for j in range(len(M)):
#             if(M[i][j]):
#                 M[i][j] /= visit[i][j]

#     ###### based on value
#     # M_temp_3 = copy.deepcopy(M)
#     temp_loop = find_loop(M)
#     # count = 0
#     while(len(temp_loop)):
#         M = wald_eliminate_loop(M, temp_loop)
#         temp_loop = find_loop(M)

#     return M


def list_diff(A_list, ancestor_dict):
    count = 0
    true_count_sum = 0
    for i in range(len(A_list)):
        l_1 = sorted(A_list[i])
        l_2 = sorted(ancestor_dict[i])

        true_count_sum += len(l_1)
        for item in l_1:
            if item in l_2:
                count += 1
    return count, true_count_sum


def get_true_ancestor(matrix):
  node_num = len(matrix)
  length = 8
  pos_list = list()
  # pos_list = [(10, 10), (5, 7), (10, 7), (2, 4), (4, 4), (6, 4), (8, 4), (1, 1), (2, 1), (3, 1)]
  y_gap = (length - 1) / np.log2(node_num)

  # count_layer = 0
  count_num = 0
  for i in range(int(np.log2(node_num)) + 1):
    if 2**i > node_num - count_num:
      x_gap = (length - 1) / (node_num - count_num + 1)
      for j in range(node_num - count_num):
        pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
        count_num += 1
    else:
      x_gap = (length - 1) / (2**i + 1)
      for j in range(2**i):
        if count_num == node_num:
          break
        pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
        count_num += 1
  
  G = nx.DiGraph()
#   plt.figure(figsize=(length, length))
  for i in range(len(matrix)):
    G.add_node(i)
    for j in range(len(matrix)):
      if(matrix[i][j]):
        G.add_edge(j, i)

  return [list(nx.ancestors(G, i)) for i in range(node_num)]


# def wald_test_pvalue(X, x, b_hat, M):
    
#     A = np.linalg.inv(np.dot(X, X.T))
#     var_b = 

    
#     return 0


def get_ancestor_loop_HSIC_to_RCD(data, ancestor_dict, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
#     l_alpha = 0.01
#     i_alpha = 0.01
#     i_alpha_U = 0.01
#     p_alpha = 0.001
    num = len(data[0])
    unfinished_list = make_unfinished(data)

    flag_anc = True
    flag_unf = True

    loop_count = 0
    
    while flag_anc or flag_unf:
        flag_anc = False
        flag_unf = False

        unfinished_list_temp = copy.deepcopy(unfinished_list)

        
        for c in unfinished_list_temp:
            
            i = c[0]
            j = c[1]
            # print(unfinished_list)
            # print(c)
            
            # print(unfinished_list_temp)
            flag_temp = False

            flag_temp = quick_ancestor(i, j, ancestor_dict)
            if flag_temp:
                flag_anc = True
                flag_unf = True

                unfinished_list.remove(c)
            elif (i == 1 and j == 2) or (i == 2 and j == 1):
                ancestor_dict[1].append(2)
                unfinished_list.remove(c)
            
            else:            
                UC = list(set(ancestor_dict[i]).intersection(set(ancestor_dict[j]))) # 交集
                if len(UC) == 0:
                    
                    if is_linear(data[:, i], data[:, j], l_alpha):
                        
                        xi_std = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
                        xj_std = (data[:, j] - np.mean(data[:, j])) / np.std(data[:, j])
                        ri_j = get_residual(xi_std, xj_std) # ri_j = xi - alpha * xj
                        rj_i = get_residual(xj_std, xi_std) # rj_i = xj - alpha * xi

                        _, pi_j = hsic_test_gamma(ri_j, xj_std, bw_method="mdbs") # xj -> xi
                        _, pj_i = hsic_test_gamma(rj_i, xi_std, bw_method="mdbs") # xi -> xj

                        if pi_j > i_alpha and pj_i <= i_alpha:
                            
                            if True:
                                ancestor_dict[i].append(j)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True
                                
                            else:
                                continue
                                # unfinished_list.append([i, j]) 
                        if pi_j <= i_alpha and pj_i > i_alpha:
                            if True:
                            # if (pi_j / pj_i < p_alpha) or (pj_i / pi_j < p_alpha):
                                ancestor_dict[j].append(i)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True
                                
                            else:
                                continue
                                # unfinished_list.append([i, j])
                        if pi_j > i_alpha and pj_i >= i_alpha:
                            # print("ccccc", pi_j, " and ", pj_i)
                            unfinished_list.remove(c)
                            flag_anc = True
                            flag_unf = True
                            # continue
                        if pi_j <= i_alpha and pj_i < i_alpha:
                            continue
                        
                    else:
                        if loop_count >= 2:
                            flag_anc = True
                            flag_unf = True
                            unfinished_list.remove(c)
                        else:
                            continue
            
                else:
                    X = list()
                    for index in UC:
                        X.append(data[:, index])
                    
                    X = np.array(X).T
                    
                    reg = LinearRegression(fit_intercept=False)
                    res = reg.fit(X, data[:, i])
                    coef = res.coef_
                    # coef = np.linalg.lstsq(X, data[:, i], rcond=None) # X, y
                    yi = data[:, i] - np.dot(coef, X.T)
                    # yi = data[:, i] - reg.predict(X)

                    res = reg.fit(X, data[:, j])
                    coef = res.coef_
                    # coef = np.linalg.lstsq(X, data[:, j], rcond=None) # X, y
                    yj = data[:, j] - np.dot(coef, X.T)
                    # yj = data[:, j] - reg.predict(X)
                    
                    # if not is_independent(yi, yj, 0.01):

                    if is_linear(yi, yj, l_alpha):
                        yi_std = (yi - np.mean(yi)) / np.std(yi)
                        yj_std = (yj - np.mean(yj)) / np.std(yj)
                        ri_j = get_residual(yi_std, yj_std) # ri_j = yi - alpha * yj
                        rj_i = get_residual(yj_std, yi_std) # rj_i = yj - alpha * yi

                        _, pi_j = hsic_test_gamma(ri_j, yj_std, bw_method="mdbs") # yj -> yi
                        _, pj_i = hsic_test_gamma(rj_i, yi_std, bw_method="mdbs") # yi -> yj
#                         _, pi_j = Hsic().test(ri_j, xj_std) # xj -> xi
#                         _, pj_i = Hsic().test(rj_i, xi_std) # xi -> xj
                        

                        if pi_j > i_alpha_U and pj_i <= i_alpha_U:
                            if True:
                            # if (pi_j / pj_i < p_alpha) or (pj_i / pi_j < p_alpha):
                                ancestor_dict[i].append(j)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True                                
                                
                            else:
                                continue
                        if pi_j <= i_alpha_U and pj_i > i_alpha_U:
                            if True:    
                            # if (pi_j / pj_i < p_alpha) or (pj_i / pi_j < p_alpha):
                                ancestor_dict[j].append(i)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True                                
                                
                            else:
                                continue
                        if pi_j > i_alpha_U and pj_i >= i_alpha_U:
                            unfinished_list.remove(c)
                            flag_anc = True
                            flag_unf = True
                            # continue
                        if pi_j <= i_alpha_U and pj_i < i_alpha_U:
                            continue
                    else:
                        unfinished_list.remove(c) 
        loop_count += 1
    

# 拓扑排序
def find_loop(G):
    graph = dict() # outdegree
    for i in range(len(G)):
        out_node = list()
        for j in range(len(G)):
            if(G[i][j]):
                out_node.append(j)
            graph[i] = out_node
     
    in_degrees = dict((u,0) for u in graph)   #初始化所有顶点入度为0  

    num = len(in_degrees)     
    for u in graph:         
        for v in graph[u]:             
            in_degrees[v] += 1    #计算每个顶点的入度     
    Q = [u for u in in_degrees if in_degrees[u] == 0]   # 筛选入度为0的顶点     
    
    Seq = list()     
    while Q:  
        # print("Q: ", Q)       
        u = Q.pop()       #默认从最后一个删除         
        Seq.append(u)         
        for v in graph[u]:             
            in_degrees[v] -= 1    #移除其所有出边
            if in_degrees[v] == 0:        
                Q.append(v)          #再次筛选入度为0的顶点
    
    Universe = [i for i in range(num)]

    # return list(set(Universe).difference(set(Seq)))
    return [i for i in Universe if i not in Seq]
    # if len(Seq) == num:       #输出的顶点数是否与图中的顶点数相等
    #     return True    
    # else:         
    #     return False


# def topoSort(graph):     
#     in_degrees = dict((u,0) for u in graph)   #初始化所有顶点入度为0  

#     num = len(in_degrees)     
#     for u in graph:         
#         for v in graph[u]:             
#             in_degrees[v] += 1    #计算每个顶点的入度     
#     Q = [u for u in in_degrees if in_degrees[u] == 0]   # 筛选入度为0的顶点     
    
#     Seq = []     
#     while Q:  
#         # print("Q: ", Q)       
#         u = Q.pop()       #默认从最后一个删除         
#         Seq.append(u)         
#         for v in graph[u]:             
#             in_degrees[v] -= 1    #移除其所有出边
#             if in_degrees[v] == 0:        
#                 Q.append(v)          #再次筛选入度为0的顶点
#     if len(Seq) == num:       #输出的顶点数是否与图中的顶点数相等
#         return True    
#     else:         
#         return False
    

def value_eliminate_loop(G, temp_loop):
    min_edge = [0, 0]
    min_value = 1000
    for i in temp_loop:
        for j in temp_loop:
            if i != j:
                if G[i][j] != 0:
                    if np.fabs(G[i][j]) < min_value:
                        min_edge = [i, j]
                        min_value = np.fabs(G[i][j])
    G[min_edge[0]][min_edge[1]] = 0
    return G


def ancestor_eliminate_loop(G, temp_loop, ancestor_list):
    # min_edge = [0, 0]
    # min_value = 1000
    for i in temp_loop:
        for j in temp_loop:
            # if i != j:
            if G[i][j] != 0:
                if j not in ancestor_list[i]:
                    G[i][j] = 0
    # G[min_edge[0]][min_edge[1]] = 0
    return G


def wald_pvalue(X, y):
    cols = list()
    if len(X.shape) == 1:
        cols = ["X0"]
    else:
        for i in range(len(X[0])):
            cols.append("X" + str(i))
    X = pd.DataFrame(X, columns=cols)

    # print(X.head())
    # 拟合线性回归模型
    model = sm.OLS(y, X).fit()

    # 打印回归模型的摘要
    # print(model.summary())

    # 执行Wald检验
    hypotheses = '(X0 = 0)'  # 在这里定义你要检验的假设，这里假设X1和X2的系数都等于零
    wald_test = model.wald_test(hypotheses)

    # p值越小越拒绝，即不为零。取p值最大的
    return wald_test.pvalue


def wald_eliminate_loop(G, temp_loop, data):
    node_num = len(G)
    # data shape = n * p
    max_edge = [0, 0]
    max_pvalue = -1000
    for i in temp_loop:
        for j in temp_loop:
            if i != j:
                # j -> i
                if G[i][j] != 0:
                    y = data[:, i]
                    X = [data[:, j]]
                    for k in range(node_num):
                        if k != j and G[i][k] != 0:
                            X.append(data[:, k])
                    X = np.array(X).T

                    pvalue = wald_pvalue(X, y)

                    if pvalue > max_pvalue:
                        max_pvalue = pvalue
                        max_edge = [i, j]
    G[max_edge[0]][max_edge[1]] = 0

    return G



def get_ancestor_loop_KCI_with_map(data, ancestor_dict, group, l_alpha, i_alpha, i_alpha_U, p_alpha=1):
    num = len(data[0])
    unfinished_list = make_unfinished(data)

    flag_anc = True
    flag_unf = True

    loop_count = 0

    inverse_map = dict()
    index_count = 0
    for key in ancestor_dict:
        inverse_map[key] = index_count
        index_count += 1
    
    while flag_anc or flag_unf:
        flag_anc = False
        flag_unf = False

        unfinished_list_temp = copy.deepcopy(unfinished_list)

        for c in unfinished_list_temp:
            
            i = c[0]
            j = c[1]
            i_map = group[i]
            j_map = group[j]
            
            flag_temp = False

            flag_temp = quick_ancestor(i_map, j_map, ancestor_dict)
            if flag_temp:
                flag_anc = True
                flag_unf = True

                unfinished_list.remove(c)
            else:            
                UC = list(set(ancestor_dict[i_map]).intersection(set(ancestor_dict[j_map]))) # 交集
                if len(UC) == 0:                    
                    if is_linear(data[:, i], data[:, j], l_alpha):
                        # xi_std = data[:, i]
                        # xj_std = data[:, j]
                        # xi_std = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
                        # xj_std = (data[:, j] - np.mean(data[:, j])) / np.std(data[:, j])
                        # ri_j = get_residual(xi_std, xj_std) # ri_j = xi - alpha * xj
                        # rj_i = get_residual(xj_std, xi_std) # rj_i = xj - alpha * xi

                        xi_std = data[:, i]
                        xj_std = data[:, j]

                        reg = LinearRegression(fit_intercept=False)

                        res = reg.fit(data[:, j].reshape(-1, 1), data[:, i]) # x, y
                        coef = res.coef_
                        ri_j = data[:, i] - coef * data[:, j] # ri_j = xi - alpha * xj

                        res = reg.fit(data[:, i].reshape(-1, 1), data[:, j]) # x, y
                        coef = res.coef_
                        rj_i = data[:, j] - coef * data[:, i]# rj_i = xj - alpha * xi
                        
                        data_set = np.array([ri_j, xj_std, rj_i, xi_std])
                        
                        kci_obj = CIT(data_set.T, "kci")
                        # pi_j = kci_obj(ri_j, xj_std, [])
                        # pj_i = kci_obj(rj_i, xi_std, [])
                        pi_j = kci_obj(0, 1, [])
                        pj_i = kci_obj(2, 3, [])

                        # print("(", i, ", ", j, ")", "pi_j, pj_i:", pi_j, "----", pj_i)
                        # _, pi_j = hsic_test_gamma(ri_j, xj_std, bw_method="mdbs") # xj -> xi
                        # _, pj_i = hsic_test_gamma(rj_i, xi_std, bw_method="mdbs") # xi -> xj

                        if pi_j > i_alpha and pj_i <= i_alpha:
                            if True:
                                ancestor_dict[i_map].append(j_map)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True
                            else:
                                continue
                        if pi_j <= i_alpha and pj_i > i_alpha:
                            if True:
                                ancestor_dict[j_map].append(i_map)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True                                
                            else:
                                continue
                        if pi_j > i_alpha and pj_i > i_alpha:
                            unfinished_list.remove(c)
                            flag_anc = True
                            flag_unf = True
                            # continue
                        if pi_j <= i_alpha and pj_i <= i_alpha:
                            continue        
                    else:
                        if loop_count >= 0: ##########################
                            flag_anc = True
                            flag_unf = True
                            unfinished_list.remove(c)
                        else:
                            continue
            
                else:
                    X = list()
                    for index in UC:
                        X.append(data[:, inverse_map[index]])
                    
                    X = np.array(X).T
                    
                    reg = LinearRegression(fit_intercept=False)
                    res = reg.fit(X, data[:, i])
                    coef = res.coef_
                    # coef = np.linalg.lstsq(X, data[:, i], rcond=None) # X, y
                    yi = data[:, i] - np.dot(coef, X.T)
                    # yi = data[:, i] - reg.predict(X)

                    res = reg.fit(X, data[:, j])
                    coef = res.coef_
                    # coef = np.linalg.lstsq(X, data[:, j], rcond=None) # X, y
                    yj = data[:, j] - np.dot(coef, X.T)
                    # yj = data[:, j] - reg.predict(X)
                    
                    # if not is_independent(yi, yj, 0.01):

                    if is_linear(yi, yj, l_alpha):
                        # yi_std = yi
                        # yj_std = yj
                        # yi_std = (yi - np.mean(yi)) / np.std(yi)
                        # yj_std = (yj - np.mean(yj)) / np.std(yj)
                        # ri_j = get_residual(yi_std, yj_std) # ri_j = yi - alpha * yj
                        # rj_i = get_residual(yj_std, yi_std) # rj_i = yj - alpha * yi

                        yi_std = yi
                        yj_std = yj

                        reg = LinearRegression(fit_intercept=False)

                        res = reg.fit(yj.reshape(-1, 1), yi) # x, y
                        coef = res.coef_
                        ri_j = yi - coef * yj # ri_j = xi - alpha * xj

                        res = reg.fit(yi.reshape(-1, 1), yj) # x, y
                        coef = res.coef_
                        rj_i = yj - coef * yi# rj_i = xj - alpha * xi
                        # _, pi_j = hsic_test_gamma(ri_j, yj_std, bw_method="mdbs") # yj -> yi
                        # _, pj_i = hsic_test_gamma(rj_i, yi_std, bw_method="mdbs") # yi -> yj

                        
                        data_set = np.array([ri_j, yj_std, rj_i, yi_std])
                        kci_obj = CIT(data_set.T, "kci")
                        pi_j = kci_obj(0, 1, [])
                        pj_i = kci_obj(2, 3, [])
                        # pi_j = kci_obj(ri_j, yj_std, [])
                        # pj_i = kci_obj(rj_i, yi_std, [])
                        # print("(", i, ", ", j, ")", "pi_j, pj_i:", pi_j, "----", pj_i)
#                         _, pi_j = Hsic().test(ri_j, xj_std) # xj -> xi
#                         _, pj_i = Hsic().test(rj_i, xi_std) # xi -> xj
                        

                        if pi_j > i_alpha_U and pj_i <= i_alpha_U:
                            if True:
                            # if (pi_j / pj_i < p_alpha) or (pj_i / pi_j < p_alpha):
                                ancestor_dict[i_map].append(j_map)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True                                
                                
                            else:
                                continue
                        if pi_j <= i_alpha_U and pj_i > i_alpha_U:
                            if True:    
                            # if (pi_j / pj_i < p_alpha) or (pj_i / pi_j < p_alpha):
                                ancestor_dict[j_map].append(i_map)
                                unfinished_list.remove(c)
                                flag_anc = True
                                flag_unf = True                                
                                
                            else:
                                continue
                        if pi_j > i_alpha_U and pj_i > i_alpha_U:
                            # unfinished_list.remove(c)
                            # flag_anc = True
                            # flag_unf = True
                            continue
                        if pi_j <= i_alpha_U and pj_i <= i_alpha_U:
                            continue
                    else:
                        flag_anc = True
                        flag_unf = True
                        unfinished_list.remove(c) 
        loop_count += 1