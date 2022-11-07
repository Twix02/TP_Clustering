#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:08:37 2022

@author: maher
"""

import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn import cluster
from sklearn import metrics

from scipy.io import arff

#Parser un fichier de donnees au format arff
#data est un tableau d'exemples avec pour chacun
#la liste des valeurs des features
#
#Dans les jeux de donnees consideres :
#il y a 2 features (dimension 2)
#Ex : [[-0.499261, -0.0612356],
#       [-1.51369, 0.265446],
#       [-1.60321, 0.362039], ...
#       ]
#
#Note : chaque exemple du jeu de donnees contient aussi un 
#numero de cluster. On retire cette information

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
    print (borne_k)
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

data = pd.read_fwf("/home/maher/Bureau/TP_Clustering/dataset-rapport/zz1.txt", header=None)

f0 = data[0]
f1 = data[1]


plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

k, score, labels, execution_time = kmeans_method(data)
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Donnees après clustering")
plt.show()


