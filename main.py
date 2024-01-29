import numpy as np
import sys
from phe import paillier
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from query import Query
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
import pandas as pd
from pymongo import MongoClient


def measure_index(knn, points, num):
    i = 0
    for M in num:
        pk, sk = paillier.generate_paillier_keypair(n_length=kappa)

        #points = np.random.rand(n, d)

        start = time.time()
        digest = knn.fit(n_cluster=clu, dataSpace=points,n_node=M)
        tree = knn.bplustree
        #tree.encrypt(tree.root, pk)
        tree.encrypt(tree.root, pk)
        used = time.time() - start
        size = tree.cal_memory(tree.root)

        print("{}".format(M))
        print("memory:{}".format(size / (1024 ** 2)))
        print("time:{}".format(used))
        i += 1


n = 2000
d = 256
kappa = 512
M = 6
k = 1
clu = 20

n_num = [2000,4000,8000,16000,32000]
d_num = [256,512,1024,2048]
kappa_num = [512,1024,2048]
M_num = [3,4,5,6]
k_num = [1,3,5,7,9,11]
clu_num = [5,10,15,20]

cost_dict = {
    "key1": {"kappa": 512, "d": 512, "MB": 72, "time": 0.98},
    "key2": {"kappa": 512, "d": 1024, "MB": 728, "time": 1.99},
    "key3": {"kappa": 512, "d": 2048, "MB": 72, "time": 3.93},
    "key4": {"kappa": 512, "d": 4096, "MB": 1056, "time": 7.97},
    "key5": {"kappa": 512, "d": 512, "MB": 72, "time": 0.98},
    "key6": {"kappa": 1024, "d": 512, "MB": 72, "time": 6.27},
    "key7": {"kappa": 2048, "d": 512, "MB": 72, "time": 42.74},
    "key8": {"kappa": 4096, "d": 512, "MB": 72, "time": 343}
}

paillier_dict = {512: {"enc":0.002, "dec":0.0007},
                 1024: {"enc":0.012, "dec":0.0038},
                 2048: {"enc":0.089, "dec":0.026}
                 }

enc_time_dict = {
    "add": {"communication": 0.0001464, "compute": 0.0000217},
    "SM": {"communication": 0.0001960, "compute": 0.0037935},
    "compare": {"communication": 0.0001528, "compute": 0.0020651},
    "SSED": {"communication": 0.0001860, "compute": 1.024},
    "OSSED": {"communication": 0.0001860, "compute": 0.0031624}
}


for clu in clu_num:
    np.random.seed(1)
    points = np.random.rand(n, d)
    pointTest = np.random.rand(d)

    knn = Query(n_neighbors=k)
    digest = knn.fit(n_cluster=clu, dataSpace=points,n_node=M)
    pk, sk = paillier.generate_paillier_keypair(n_length=kappa)


    start = time.time()
    nNeighbors, add, SSED, SM, compare = knn.kNeighbors(pointTest)
    used = time.time() - start

    print(add)
    print(SSED)
    print(SM)
    print(compare)

    communication = add*enc_time_dict["add"]["communication"] + SSED*enc_time_dict["SSED"]["communication"]*(d/256) + SM*enc_time_dict["SM"]["communication"] + compare*enc_time_dict["compare"]["communication"]
    computation = used + add*enc_time_dict["add"]["compute"] + SSED*enc_time_dict["SSED"]["compute"]*(d/256) + SM*enc_time_dict["SM"]["compute"] + compare*enc_time_dict["compare"]["compute"]

    communication_opt = add * enc_time_dict["add"]["communication"] + SSED * enc_time_dict["OSSED"]["communication"] + SM * enc_time_dict["SM"]["communication"] + compare * enc_time_dict["compare"]["communication"]
    computation_opt = used + add * enc_time_dict["add"]["compute"] + SSED * enc_time_dict["OSSED"]["compute"] + SM * enc_time_dict["SM"]["compute"] + compare * enc_time_dict["compare"]["compute"]

    print("{}".format(clu))
    print("BasCMQ:{}".format(used))

    print("SVCMQ-communication:{}".format(communication))
    print("SVCMQ-computation:{}".format(computation))

    print("opt-communication:{}".format(communication_opt))
    print("opt-computation:{}".format(computation_opt))

