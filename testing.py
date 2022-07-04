import search
import time
import os
import pandas as pd
import pickle
import numpy as np
from rtree import index
#Testing Data
path_1 = "./DataSet/40/"


image_path = path_1 + "S2_10.jpg"

#Traning Data
rtree_path = "./bin_test/rtree_index_99_60"
dataset_path = "./data/dataset_PCA_99_60.csv"
scaler_path = "./bin/scaler_PCA_99_60.dat"
pca_path = "./bin/pca_PCA_99_60.dat"
ncomponents_path = "./bin/ncomponents_PCA_99_60.dat"

size = 90 ##cambio

def timeKnrearest(size): #knn_rtree
    start = time.perf_counter()
    result = search.knearest(image_path, 1, rtree_path)
    end = time.perf_counter()
    print(f"Knearest with size: {size} elements finished in {end - start:0.4f} seconds")
    print(result)

def timeSequentialKnn(size): #knn_sequential
    start = time.perf_counter()
    result = search.searchKNN_sequential(image_path, 1, dataset_path)
    end = time.perf_counter()
    print(f"Sequential Knn with size: {size} elements finished in {end - start:0.4f} seconds")
    print(result)

search.initialize_models(scaler_path, pca_path, ncomponents_path)

timeKnrearest(size)
timeSequentialKnn(size)
