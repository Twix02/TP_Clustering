#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  10 12:14:32 2022

@author: maher
"""

import matplotlib.pyplot as plt
from scipy.io import arff
import numpy as np
import pandas as pd
import time
import hdbscan
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors


def calcul_silhouette_score(data, labels):
    if len(set(labels)) <= 1:
        return -1
    return metrics.silhouette_score(data, labels)

def get_distance_max (data):
    #Distances k plus proches voisins
    #Donnees dans X
    k = 5
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(data)
    distances, indices = neigh.kneighbors(data)
    
    #retirer le point "origine"
    newDistances = np.array([np.average(distances[i][1:]) for i in range(0, distances.shape[0])])
    trie = np.sort(newDistances)
    return max(trie)
    


def hdbscan_method(data) :
    tps1=time.time()
    meilleur_score = 0
    meilleurs_labels = (0, [0 for _ in data])
    meilleurs_k_clusters = None
    
    meilleur_min_samples = 0
    ms_list = [2**i for i in range(2,5)]
    
    for m in ms_list:
        model = hdbscan.HDBSCAN(min_samples=m)
        model = model.fit(data)
        labels = model.labels_
        k_clusters = len(np.unique(labels))
        print(labels)
        score = calcul_silhouette_score(data, labels)
        print(score)
        if not meilleur_score or score > meilleur_score :
            meilleur_score = score
            meilleurs_labels = labels
            meilleurs_k_clusters = k_clusters
            meilleur_min_samples=m
        
    tps2=time.time()
    return meilleurs_k_clusters, meilleur_score, meilleurs_labels, meilleur_min_samples, round((tps2-tps1)*1000,2)
          
        
              
#path = './clustering-benchmark-master/src/main/resources/datasets/artificial/'
#databrut = arff.loadarff(open(path+"R15.arff", 'r'))
#databrut = arff.loadarff(open(path+"aggregation.arff", 'r'))
#databrut = arff.loadarff(open(path+"3-spiral.arff", 'r'))
#databrut = arff.loadarff(open(path+"banana.arff", 'r'))


#data = [[x[0],x[1]] for x in databrut[0]]

#Affichage en 2D 
#Extraire chaque valeur de features pour en faire une liste
#Ex pour f0 = [-0.499261, -1.51369, -1.60321, ...]
#Ex pour f1 = [-0.0612356, 0.265446, 0.362039, ...]
#f0 = [f[0] for f in data]
#f1 = [f[1] for f in data]

data = pd.read_fwf("/home/maher/Bureau/TP_Clustering/dataset-rapport/x4.txt", header=None)

f0 = data[0]
f1 = data[1]

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

linkages=['ward', 'complete', 'average', 'single']
k, score, labels, meilleur_min_samples, execution_time = hdbscan_method(data)
print(labels)
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Donnees après clustering")
plt.show()
