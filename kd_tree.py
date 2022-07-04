import numpy as np
from scipy.spatial import KDTree
import face_recognition
import pandas as pd
import pickle
import time
size = 90
#Traning_Data
dataset_path = "./data/dataset_PCA_90_90.csv"
scaler_path = "./bin/scaler_PCA_90_90.dat"
pca_path = "./bin/pca_PCA_90_90.dat"
ncomponents_path = "./bin/ncomponents_PCA_90_90.dat"


#Testing Data
path_1 = "./DataSet/10/"
image_path = path_1 + "S2_6.jpg"



df = pd.read_csv(dataset_path)
scaler = pickle.load(open(scaler_path, "rb"))
pca = pickle.load(open(pca_path, "rb"))
ncomponents = pickle.load(open(ncomponents_path, "rb"))

#print("Completed")
#df.tail(5)


features = [str(i) for i in range(1, ncomponents+1)]
x = df.loc[:, features]
y = df.loc[:, ["path"]]
#y.tail(5)

tree = KDTree(x)

def generate_df(x):
    return pd.DataFrame(data=x, columns = [str(i) for i in range(1, 129)])

def generate_point(v):
    return tuple(np.concatenate([v, v], axis = None))

def parser_image(image_path):
    picture = face_recognition.load_image_file(image_path)    
    all_face_encodings = face_recognition.face_encodings(picture)
    x = generate_df(all_face_encodings)
    x_scaled = scaler.transform(x)
    x_pca = pca.transform(x_scaled)
    return x_pca[0]


image = parser_image(image_path)

start = time.perf_counter()
dd, ii = tree.query([image],k=1)
end = time.perf_counter()
#print(dd, ii, sep='\n')
#print(df.iloc[ii[0]])
print(f"Knearest with size: {size} elements finished in {end - start:0.7f} seconds")
