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

# pVal = 0.001

def SADA(x_list, pVal):
    return findCausalSet(x_list, pVal)

# error alogrithm
def findSmallIndependentSet2(x, y, Z, kci_obj, pVal):
    result = copy.deepcopy(Z)
    for f in Z:
        pValue = kci_obj(x, y, result - {f})
        if(pValue > pVal):
            result = result - {f}
    return result


def generate_all_subset(Z):
    res = list()
    for a in range(len(Z)):
        for i in itertools.combinations(Z, a+1):
            res.append(set(i))
    random.shuffle(res)
    return res


def find_C(x, y, Z, kci_obj, pVal):
    all_subset = generate_all_subset(Z)
    # print(all_subset)
    # print("subsets: ", all_subset)
    result = copy.deepcopy(Z)
    max_size = len(Z)
    for C_hat in all_subset:
        if (kci_obj(x, y, C_hat) > pVal):
            temp_size = len(C_hat)
            if temp_size < max_size:
                max_size = temp_size
                result = C_hat
    if (kci_obj(x, y) > pVal):
        result = set()    
    return result
#

def findSmallIndependentSet(x, y, Z, kci_obj, pVal):
    minSubSet = Z
    minSubSetNum = len(Z)
    sub_sets = []
    sub_sets.append(set())
    for z in Z:
        for subSet in sub_sets[:]:
            sub_sets.append(copy.deepcopy(subSet))
            subSet.add(z)

    for subSet in sub_sets:
        if(len(subSet) < minSubSetNum and kci_obj(x, y, subSet) > pVal):
            minSubSetNum = len(subSet)
            minSubSet = subSet
    print(len(sub_sets))

    print("init u = ", x, " v = ", y)
    print("init V: ")
    print(minSubSet)
    return minSubSet


def for_all_test_one_C(w, V, C, kci_obj, pVal):
    addFlag = True
    for v in V:
        if(kci_obj(w, v, C) > pVal): 
            continue
        else:
            addFlag = False
            break
    return addFlag


def for_all_test_one_C_empty(w, V, kci_obj, pVal):
    addFlag = True
    for v in V:
        if(kci_obj(w, v) > pVal): 
            continue
        else:
            addFlag = False
            break
    return addFlag


def for_all_test(w, V, C_subsets, kci_obj, pVal):
    addFlag = False
    for C_hat in C_subsets:
        if for_all_test_one_C(w, V, C_hat, kci_obj, pVal):
            addFlag = True
        else:
            continue
    if addFlag is False and for_all_test_one_C_empty(w, V, kci_obj, pVal):
        addFlag = True

    return addFlag


def findCausalSet(V, pVal):
    kci_obj = CIT(V, "kci")
    samples = len(V)
    features = len(V[0])

    u = 0
    v = 0
    CMax = {}
    while(1):
        u = randint(0, features - 1)
        v = randint(0, features - 1)
        X = u
        Y = v

        if(kci_obj(X, Y, set(range(0, features)) - {X, Y}) > pVal):
            CMax = set(range(0, features)) - {X, Y}
            break

    C = find_C(u, v, CMax, kci_obj, pVal)
    print("C = ", C)

    V1 = {u}
    V2 = {v}
    print("V1 = ", V1, " V2 = ", V2)

    leftV = set(range(0, features)) - V1 - V2 - C
    
    for w in leftV:
        C_subsets = generate_all_subset(C)
        addFlag = for_all_test(w, V1, C_subsets, kci_obj, pVal)
        if addFlag:
            print("add ", w, " into V2")
            V2.add(w)
            continue
        else:    
            addFlag = for_all_test(w, V2, C_subsets, kci_obj, pVal)
            if addFlag:
                print("add ", w, " into V1")
                V1.add(w)
                continue
        
        
        if(not addFlag):
            print("add ", w, " into C")
            C.add(w)

    copyC = copy.deepcopy(C)
    for s in copyC:
        # addFlag = True
        C_subsets = generate_all_subset(C - {s})
        print(C_subsets)
        addFlag = for_all_test(s, V1, C_subsets, kci_obj, pVal)
        
        if addFlag:
            print("add ", s, " into V2 from C")
            V2.add(s)
            C.remove(s)
            continue
        
        else:
            addFlag = for_all_test(s, V2, C_subsets, kci_obj, pVal)    
            if addFlag:
                print("add ", s, " into V1 from C")
                V1.add(s)
                C.remove(s)
                continue
    return (C | V1, C| V2)