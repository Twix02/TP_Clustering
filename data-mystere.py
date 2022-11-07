#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:49:28 2022

@author: maher
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import kmedoids
import scipy . cluster . hierarchy as shc
import hdbscan


from sklearn import cluster
from sklearn import metrics
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.neighbors import NearestNeighbors


from scipy.io import arff

def calcul_silhouette_score(data, labels):
    if len(set(labels)) <= 1:
        return -1
    return metrics.silhouette_score(data, labels)

def kmeans_method(data):
    tps1=time.time()
    meilleur_score = 0
    meilleurs_labels = (0, [0 for _ in data])
    meilleurs_k_clusters = None
    
    borne_k = round(len(data)**0.5)
    for i in range(5,borne_k):
        model = cluster.KMeans(n_clusters=i, init='k-means++')
        model.fit(data)
        labels = model.labels_
        score = calcul_silhouette_score(data, labels)
        if score > meilleur_score  :
            meilleur_score = score
            meilleurs_labels = labels
            meilleurs_k_clusters = i
        
        tps2=time.time()
        return meilleurs_k_clusters, meilleur_score, meilleurs_labels, (round((tps2-tps1)*1000,2))
    
def kmedoids_method(data):
    tps1=time.time()
    meilleur_score = 0
    meilleurs_labels = (0, [0 for _ in data])
    meilleurs_k_clusters = None
    
    borne_k = round(len(data)**0.5)
    for i in range(5,borne_k):
        distmatrix = manhattan_distances(data)
        model = kmedoids.fasterpam(distmatrix, i)
        labels = model.labels
        score = calcul_silhouette_score(data, labels)
        if score > meilleur_score  :
            meilleur_score = score
            meilleurs_labels = labels
            meilleurs_k_clusters = i
        
        tps2=time.time()
        return meilleurs_k_clusters, meilleur_score, meilleurs_labels, (round((tps2-tps1)*1000,2))
    
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

def dbscan_method(data):
    tps1=time.time()
    meilleur_score = 0
    meilleurs_labels = (0, [0 for _ in data])
    meilleurs_k_clusters = None
    
    meilleur_eps = 0
    meilleur_min_samples = 0
    
    max_dist = get_distance_max(data)
    e_samples = 10
    e_list = [max_dist*i/e_samples for i in range(1,2*e_samples)]
    
    ms_list = [2**i for i in range(2,5)]
    for e in e_list:
        for m in ms_list :
            model = cluster.DBSCAN(eps=e, min_samples=m) 
            model = model.fit(data)
            labels = model.labels_
            k_clusters = len(np.unique(model.labels_))
            score = calcul_silhouette_score(data, labels)
            if score > meilleur_score  :
                meilleur_score = score
                meilleurs_labels = labels
                meilleurs_k_clusters = k_clusters
                meilleur_eps = e
                meilleur_min_samples = m
    tps2=time.time()
    return meilleurs_k_clusters, meilleur_score, meilleurs_labels, meilleur_eps, meilleur_min_samples, (round((tps2-tps1)*1000,2))
           
def hdbscan_method(data):
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
         score = calcul_silhouette_score(data, labels)
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

data = pd.read_fwf("/home/maher/Bureau/TP_Clustering/dataset-rapport/zz1.txt", header=None)

f0 = data[0]
f1 = data[1]

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

# KMeans method
k, score, labels, execution_time = kmeans_method(data)
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Donnees après clustering KMeans")
plt.show()

# KMedoids method 
k_medoids, score_medoids, labels_medoids, execution_time_medoids = kmedoids_method(data)
plt.scatter(f0, f1, c=labels_medoids, s=8)
plt.title("Donnees après clustering KMedoids")
plt.show()

# Clustering agglomeratif
linkages=['ward', 'complete', 'average', 'single']
k_agglo, score_agglo, labels_agglo, linkage, execution_time_agglo = clustering_agglo_method(data, linkages)
plt.scatter(f0, f1, c=labels_agglo, s=8)
plt.title("Donnees après Clustering Agglomeratif")
plt.show()

# Clustering DBSCAN
k_dbscan, score_dbscan, labels_dbscan, eps, min_samples, execution_time_dbscan = dbscan_method(data)
plt.scatter(f0, f1, c=labels_dbscan, s=8)
plt.title("Donnees après clustering DBSCAN")
plt.show()

# Clustering HDBSCAN
k_hdbscan, score_hdbscan, labels_hdbscan, meilleur_min_samples, execution_time_hdbscan = hdbscan_method(data)
plt.scatter(f0, f1, c=labels_hdbscan, s=8)
plt.title("Donnees après clustering HDBSCAN")
plt.show()

print("Kmeans : nombre de clusters =", k, ", score = ", score, ", temps d'execution = ", execution_time)
print("Kmedoids : nombre de clusters =", k_medoids, ", score = ", score_medoids, ", temps d'execution = ", execution_time_medoids)
print("Clustering Agglomeratif : nombre de clusters =", k_agglo, ", score = ", score_agglo,", temps d'execution = ", execution_time_agglo, ", linkage = ", linkage)
print("DBSCAN : nombre de clusters =", k_dbscan, ", score = ", score_dbscan, ", temps d'execution = ", execution_time_dbscan, ", meilleur eps = ", eps, ", meilleur min_samples = ", min_samples)
print("HDBSCAN : nombre de clusters =", k_hdbscan, ", score = ", score_hdbscan, ", temps d'execution = ", execution_time_hdbscan)



