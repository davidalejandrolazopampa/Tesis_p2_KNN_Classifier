import os
import face_recognition
import csv
import numpy as np
import threading
import math

images_path = "./DataSet/90/"
images_folder = os.listdir(images_path)
#print(images_folder)
csvfile = open("dataset_FC_90.csv", 'w', encoding='UTF8', newline='') 

writer = csv.writer(csvfile)
header = [i for i in range(1, 129)]
header.append("path")
writer.writerow(header)

lock = threading.Lock()

def processing(images_folders = []):
    global writer    
    for image in images_folder:
        #print(image)   
        image_path = images_path + image
        picture = face_recognition.load_image_file(image_path)    
        all_face_encodings = face_recognition.face_encodings(picture) #Aqui
        if len(all_face_encodings) == 0:
            print(image)
        for face_encoding in all_face_encodings:
            path = images_path + image
            row = np.append(face_encoding, path)
            lock.acquire()
            writer.writerow(row)
            lock.release()

def execute():
    nthreads = 1
    size = len(images_folder)
    step = math.ceil(size / nthreads)
    threads = []
    for i in range(nthreads):
        start = i * step
        end = min((i+1)*step, size)
        v = images_folder[start:end]
        thread = threading.Thread(target=processing, args=(v,))
        threads.append(thread)
    
    for i in range(nthreads):
        threads[i].start()
    
    for i in range(nthreads):
        threads[i].join()

execute()