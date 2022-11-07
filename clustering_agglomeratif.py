#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:35:23 2022

@author: maher
"""

import matplotlib.pyplot as plt
from scipy.io import arff

import time
from sklearn import cluster
from sklearn import metrics

import pandas as pd


import scipy . cluster . hierarchy as shc

def calcul_silhouette_score(data, labels):
    if len(set(labels)) <= 1:
        return -1
    return metrics.silhouette_score(data, labels)


def create_dendrogramme(data):
    # Donnees dans datanp
    print("Dendrogramme ’single’ donnees initiales")
    linked_mat = shc.linkage(data, 'ward')
    plt.figure(figsize = (12, 12))
    shc.dendrogram(linked_mat, orientation = 'top', distance_sort = 'descending', show_leaf_counts = False )
    plt.show()
    
def clustering_agglo_method(data, link):
    tps1=time.time()
    meilleur_score = 0
    meilleurs_labels = (0, [0 for _ in data])
    meilleurs_k_clusters = None
    
    meilleur_linkage = None
    

    for l in link :
        model = cluster.AgglomerativeClustering(distance_threshold=100, linkage = l, n_clusters = None)
        model = model.fit(data)
        labels = model.labels_
        k_clusters = model.n_clusters_
        score = calcul_silhouette_score(data, labels)
        if score > meilleur_score  :
            meilleur_score = score
            meilleurs_labels = labels
            meilleurs_k_clusters = k_clusters
            meilleur_linkage = l
    tps2=time.time()
    return meilleurs_k_clusters, meilleur_score, meilleurs_labels, meilleur_linkage, (round((tps2-tps1)*1000,2))
                
path = './clustering-benchmark-master/src/main/resources/datasets/artificial/'
#databrut = arff.loadarff(open(path+"R15.arff", 'r'))
databrut = arff.loadarff(open(path+"aggregation.arff", 'r'))
#databrut = arff.loadarff(open(path+"3-spiral.arff", 'r'))
#databrut = arff.loadarff(open(path+"banana.arff", 'r'))


data = [[x[0],x[1]] for x in databrut[0]]


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
k, score, labels, linkage, execution_time = clustering_agglo_method(data, linkages)
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Donnees après clustering")
plt.show()

