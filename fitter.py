import numpy as np
from Oi import O
from sklearn.cluster import KMeans

def hyperCubeStats(dim, dataSpace):
    
    l = 0
    

    for i in range(dim):
        minValue = float("+inf")
        maxValue = float("-inf")
        for j in range(len(dataSpace)):
            if (dataSpace[j][i] > maxValue):
                maxValue = dataSpace[j][i]
            if (dataSpace[j][i] < minValue):
                minValue = dataSpace[j][i]
        
        curr_l = abs(maxValue - minValue)
        
        if(curr_l > l):
            l = curr_l
    

    orig = np.zeros(dim)
    edge = np.zeros(dim)
    for i in range(len(edge)):
        edge[i] = l
    
    diag = np.linalg.norm(edge - orig)
    
    c = 2 * diag
    dRadius = l/100 #1%
    

    o = []
    i = 0
    j = 0 
    while(i < dim):
        
        O1 = O(np.zeros(dim))
        O2 = O(np.zeros(dim))
        
        O1.vect[i] = l/2
        O2.vect[i] = -l/2
        
        o.append(O1)
        o.append(O2)
        i += 1

    kmeans = KMeans(n_clusters = 1, random_state = 0).fit(dataSpace)
    
    s = kmeans.cluster_centers_[0]
    

    for i in range(len(o)):
        o[i].vect += s
        

    mapO = np.zeros(len(dataSpace), dtype = int)
    
    for i in range(len(dataSpace)):
        
        minValue = float('+inf')

        for j in range(len(o)):
            curr_dist = np.linalg.norm(dataSpace[i] - o[j].vect)
            if (curr_dist < minValue):
                minValue = curr_dist
                mapO[i] = j
            

    for j in range(len(o)):
        maxValue = float("-inf")

        for i in range(len(dataSpace)):
            if(mapO[i] == j):
                curr_dist = np.linalg.norm(dataSpace[i] - o[j].vect)
                if curr_dist > o[j].max_r :
                    maxValue = curr_dist
                    o[j].max_r = curr_dist
    
    
    return o, c, dRadius, mapO

def hyperCubeFurthestFit(dim, dataSpace):
    
    l = 0
    

    for i in range(dim):
        minValue = float("+inf")
        maxValue = float("-inf")
        for j in range(len(dataSpace)):
            if (dataSpace[j][i] > maxValue):
                maxValue = dataSpace[j][i]
            if (dataSpace[j][i] < minValue):
                minValue = dataSpace[j][i]
        
        curr_l = abs(maxValue - minValue)
        
        if(curr_l > l):
            l = curr_l
    

    orig = np.zeros(dim)
    edge = np.zeros(dim)
    for i in range(len(edge)):
        edge[i] = l
    
    diag = np.linalg.norm(edge - orig)
    
    c = 2 * diag
    dRadius = l/100 #1%
    

    o = []
    i = 0
    j = 0 
    while(i < dim):
        
        O1 = O(np.zeros(dim))
        O2 = O(np.zeros(dim))
        
        O1.vect[i] = l/2
        O2.vect[i] = -l/2
        
        o.append(O1)
        o.append(O2)
        i += 1


    kmeans = KMeans(n_clusters = 1, random_state = 0).fit(dataSpace)
    
    s = kmeans.cluster_centers_[0]
    

    for i in range(len(o)):
        o[i].vect += s
        

    mapO = np.zeros(len(dataSpace), dtype = int)
    
    for i in range(len(dataSpace)):
        
        maxValue = float('-inf')

        for j in range(len(o)):
            curr_dist = np.linalg.norm(dataSpace[i] - o[j].vect)
            if (curr_dist > maxValue):
                maxValue = curr_dist
                mapO[i] = j
            

    for j in range(len(o)):
        maxValue = float("-inf")

        for i in range(len(dataSpace)):
            if(mapO[i] == j):
                curr_dist = np.linalg.norm(dataSpace[i] - o[j].vect)
                if curr_dist > o[j].max_r :
                    maxValue = curr_dist
                    o[j].max_r = curr_dist
    
    
    return o, c, dRadius, mapO
    
def fit(dataSpace):
    
    dim = len(dataSpace[0])
    
    o, c, dRadius, mapO = hyperCubeStats(dim, dataSpace)
    
    return o, c, dRadius, mapO

def clusterFit(num_cluster, dataSpace):
    
    dim = len(dataSpace[0])
    
    l = 0
    

    for i in range(dim):
        minValue = float("+inf")
        maxValue = float("-inf")
        for j in range(len(dataSpace)):
            if (dataSpace[j][i] > maxValue):
                maxValue = dataSpace[j][i]
            if (dataSpace[j][i] < minValue):
                minValue = dataSpace[j][i]
        
        curr_l = abs(maxValue - minValue)
        
        if(curr_l > l):
            l = curr_l
    

    orig = np.zeros(dim)
    edge = np.zeros(dim)
    for i in range(len(edge)):
        edge[i] = l
    
    diag = np.linalg.norm(edge - orig)
    
    c = 2 * diag
    dRadius = l/100 #1%
    

    kmeans = KMeans(n_clusters = num_cluster, random_state = 0).fit(dataSpace)
    
    clusterCenters = kmeans.cluster_centers_
    
    o = []
    
    for i in range(len(clusterCenters)):
        Oi = O(clusterCenters[i])
        o.append(Oi)
        

    mapO = np.zeros(len(dataSpace), dtype = int)
    
    for i in range(len(dataSpace)):
        
        minValue = float('+inf')

        for j in range(len(o)):
            curr_dist = np.linalg.norm(dataSpace[i] - o[j].vect)
            if (curr_dist < minValue):
                minValue = curr_dist
                mapO[i] = j
            

    for j in range(len(o)):
        maxValue = float("-inf")

        for i in range(len(dataSpace)):
            if(mapO[i] == j):
                curr_dist = np.linalg.norm(dataSpace[i] - o[j].vect)
                if curr_dist > o[j].max_r :
                    maxValue = curr_dist
                    o[j].max_r = curr_dist
    
    return o, c, dRadius, mapO