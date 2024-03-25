import copy
from random import randint
import random

import data_generator as ge
from causallearn.utils.cit import CIT
import time
import draw
import numpy as np

from collections.abc import Iterable
import itertools

from sklearn.linear_model import LinearRegression

# pVal = 0.001

def CAPA(x_list, sigma, pVal):
    # return findCausalSet(x_list, sigma, pVal)
    # return findCausalSet_test(x_list, sigma, pVal)
    if len(x_list[0]) == 1: # 分组中只有一个元素
        res = set()
        res.add(0)
        return [list(res)]
    res = findCausalSet(x_list, sigma, pVal)
    V_1 = list(res[0])
    V_2 = list(res[1])
    print(res)
    # return findCausalSet(x_list, sigma, pVal)
    if len(V_1) == len(x_list[0]) or len(V_2) == len(x_list[0]):
        return [list(V_1), list(V_2)]
    else:
        res_V = list()

        x_list_1 = list()
        x_list_2 = list()

        for i in V_1:
            x_list_1.append(x_list[:, i])
        x_list_1 = np.array(x_list_1).T

        for i in V_2:
            x_list_2.append(x_list[:, i])
        x_list_2 = np.array(x_list_2).T

        V_1_sub = list()
        if len(V_1) == 1: # V_1只含有一个元素的时候
            V_1_sub.append(0)
            V_1_sub = [list(V_1_sub)]
            V_1_sub[0][0] = V_1[V_1_sub[0][0]] # map出V_1中那个元素的实际值
        else: 
            V_1_sub = CAPA(x_list_1, sigma, pVal)
            V_1_sub = list(V_1_sub)
            # print(V_1_sub)
            for i in range(len(V_1_sub)):
                V_1_sub[i] = list(V_1_sub[i])
                for j in range(len(V_1_sub[i])):
                    V_1_sub[i][j] = V_1[V_1_sub[i][j]]

        V_2_sub = list()
        if len(V_2) == 1:
            V_2_sub.append(0)
            V_2_sub = [list(V_2_sub)]
            V_2_sub[0][0] = V_2[V_2_sub[0][0]]
        else:
            V_2_sub = CAPA(x_list_2, sigma, pVal)
            V_2_sub = list(V_2_sub)
            # print(V_2_sub)
            for i in range(len(V_2_sub)):
                V_2_sub[i] = list(V_2_sub[i])
                for j in range(len(V_2_sub[i])):
                    V_2_sub[i][j] = V_2[V_2_sub[i][j]]

        print("vvv1 ", V_1_sub)
        print("vvv2 ", V_2_sub)
        
        for c in V_1_sub:
            # print(c)
            res_V.append(c)
        for c in V_2_sub:
            # print(c)
            res_V.append(c)
        return res_V


# def coding_map():


def generate_all_subset(Z, sigma):
    res = list()
    # for a in range(len(Z)):
    for i in itertools.combinations(Z, sigma):
        res.append(set(i))
    # random.shuffle(res)
    return res


def make_M_table(features, kci_obj, pVal, sigma):
    M = np.zeros((features, features))

    if sigma == 0:
        for i in range(0, features):
            for j in range(i+1, features):
                if kci_obj(i, j) > pVal:
                    M[i][j] = 1
                    M[j][i] = 1
    else:
        for i in range(0, features):
            for j in range(i+1, features):
                Z = set(range(0, features)) - {i, j}
                C = generate_all_subset(Z, sigma)
                for c in C:
                    # print((i, j), " given ", c)
                    if kci_obj(i, j, c) > pVal:
                        M[i][j] = 1
                        M[j][i] = 1
                        break
    return M


def make_M_table_ReCIT(data, features, kci_obj, pVal, sigma):
    M = np.zeros((features, features))

    if sigma == 0:
        for i in range(0, features):
            for j in range(i+1, features):
                if kci_obj(i, j) > pVal:
                    M[i][j] = 1
                    M[j][i] = 1
    else:
        for i in range(0, features):
            for j in range(i+1, features):
                Z = set(range(0, features)) - {i, j}
                C = generate_all_subset(Z, sigma)
                for c in C:
                    X = list()
                    for index in c:
                        X.append(data[:, index])
                    X = np.array(X).T
                    reg = LinearRegression(fit_intercept=False)
                    res = reg.fit(X, data[:, i])
                    coef = res.coef_
                    
                    yi = data[:, i] - np.dot(coef, X.T)
                    res = reg.fit(X, data[:, j])
                    coef = res.coef_
                    yj = data[:, j] - np.dot(coef, X.T)
                    data_set = np.array([yj, yi])
                    kci_obj_C = CIT(data_set.T, "kci")
                    # if kci_obj(i, j, c) > pVal:
                    #     M[i][j] = 1
                    #     M[j][i] = 1
                    #     break
                    if kci_obj_C(0, 1) > pVal:
                        M[i][j] = 1
                        M[j][i] = 1
                        break
    return M


def sort_by_M(M):
    features = len(M)
    
    res = [i for i in range(features)]
    sum_list = list()
    
    for i in range(0, features):
        temp = 0
        for j in range(0, features):
            if i != j:
                temp += M[i][j]
        sum_list.append(int(temp))

    print("sum_list: ", sum_list)
    res = sorted(res, key=lambda x: sum_list[x], reverse=True)
            
    return res


def initiate_group(M, sorted_node_list):
    features = len(sorted_node_list)
    V1 = {sorted_node_list[0]}
    V2 = set()
    for i in range(1, features):
        if M[sorted_node_list[0]][sorted_node_list[i]] == 1:
            V2 = {sorted_node_list[i]}
            break
    if len(V2) == 0:
        temp_node = randint(0, features - 1)
        while temp_node == sorted_node_list[0]:
            temp_node = randint(0, features - 1)
    
    return V1, V2, set(sorted_node_list) - V1 - V2


def for_all_test_one_C(w, V, M):
    addFlag = True
    # print(V)
    for v in V:
        if(M[w][v]): 
            continue
        else:
            addFlag = False
            break
    return addFlag


def find_D(V1, V2, C, M):
    D = set()
    features = len(M)
    node_list = [i for i in range(features)]
    for i in node_list:
        if i in V1 or i in V2:
            for j in C:
                if M[i][j] == 0:
                    # print(i, " and ", j)
                    D.add(i)
                    break
    return D


def findCausalSet(V, sigma, pVal):
    kci_obj = CIT(V, "kci")
    samples = len(V)
    features = len(V[0])
    # pVal = 0.01
    # V1 = set()
    # V2 = set()
    print(features)
    # sigma_now = 0
    # while sigma_now < sigma and (len(V1) == features or len(V2) == features):
    for sigma_now in range(sigma + 1):
        print("---------------------")    
        print("sigma now", sigma_now)
        V1 = set()
        V2 = set()
        # print("V1: ", V1)
        # print("V2: ", V2)
        # M = make_M_table(features, kci_obj, pVal, sigma_now)

        M = make_M_table_ReCIT(V, features, kci_obj, pVal, sigma_now)

        res = sort_by_M(M)
        print(res)
        
        A, B, res_set = initiate_group(M, res)
        C = set()
        
        res_list = list(res_set)
        random.shuffle(res_list)

        print("after: ", res_list)

        for i in res_list:
            addFlag = for_all_test_one_C(i, A, M)
            if addFlag:
                B.add(i)
                
            addFlag = for_all_test_one_C(i, B, M)
            if addFlag:
                A.add(i)
                
        # print(res)
        C = set(res) - A - B
        
        print("A: ", A)
        print("B: ", B)
        print("C: ", C)
        # for s in range(sigma):
        D = find_D(A, B, C, M)
        print("D: ", D)
        
        V1 = A | C | D
        V2 = B | C | D

        sigma_now += 1

        V1 = list(V1)
        V2 = list(V2)

        print("V1: ", V1)
        print("V2: ", V2)
        
        if len(V1) == features or len(V2) == features:
            # if sigma_now < sigma:
            #     print("continue: ", sigma_now, " is smaller than ", sigma + 1)
            continue
        else:
            # print("break: ", " the size of V1 is ", len(V1), " and the size of V2 is ", len(V2))
            break


    return [V1, V2]


def findCausalSet_test(V, sigma, pVal):
    kci_obj = CIT(V, "kci")
    samples = len(V)
    features = len(V[0])
    pVal = 0.01
    # V1 = set()
    # V2 = set()
    print(features)
    # sigma_now = 0
    # while sigma_now < sigma and (len(V1) == features or len(V2) == features):
    for sigma_now in range(sigma + 1):
        print("---------------------")    
        print("sigma now", sigma_now)
        V1 = set()
        V2 = set()
        # print("V1: ", V1)
        # print("V2: ", V2)
        M = make_M_table(features, kci_obj, pVal, sigma_now)

        res = sort_by_M(M)
        print("M Table: ")
        print(M)
        A, B, res_set = initiate_group(M, res)
        C = set()

        A = {2}
        B = set()
        C = set()
        
        print("init: A = ", A, " and B = ", B)
        res_list = [4, 5, 8, 1, 7, 0, 6, 3]
        # random.shuffle(res_list)

        for i in res_list:
            addFlag = for_all_test_one_C(i, A, M)
            if addFlag:
                B.add(i)
                
            addFlag = for_all_test_one_C(i, B, M)
            if addFlag:
                A.add(i)
                
        # print(res)
        C = set(res) - A - B
        
        print(A)
        print(B)
        print(C)
        # for s in range(sigma):
        D = find_D(A, B, C, M)
        print(D)
        
        V1 = A | C | D
        V2 = B | C | D

        sigma_now += 1

        print("V1: ", V1)
        print("V2: ", V2)
        
        if len(V1) == features or len(V2) == features:
            # if sigma_now < sigma:
            #     print("continue: ", sigma_now, " is smaller than ", sigma + 1)
            continue
        else:
            # print("break: ", " the size of V1 is ", len(V1), " and the size of V2 is ", len(V2))
            break


    return (V1, V2)