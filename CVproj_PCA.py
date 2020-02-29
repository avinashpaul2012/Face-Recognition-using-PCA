# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:21:36 2019

@author: hp
"""
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt

def build_dataset():
    org_dataset1 = []
    flag=0
#The zfill() method returns a copy of the string with '0' characters padded to the left,  
    for i in range(1, 16):
        #p=r'C:\Users\hp\Downloads\Yale\Yale\subject'+str(i).zfill(2)+"*"
        data = glob.glob(r'C:\Users\hp\Downloads\Yale\Yale\subject'+str(i).zfill(2)+"*")
        for fname in data:
            img = np.array(Image.open(fname))
            #print(img.shape[0],img.shape[1])
            a=img.shape[0]
            b=img.shape[1]
            if flag==0:
                print(a,b)
                flag=1
            img = img.reshape(img.shape[0]*img.shape[1])  #to convert the image from 2d to 1d vector
            org_dataset1.append(img)  # adding new row

    org_dataset1 = np.array(org_dataset1)
    print(org_dataset1.shape)   
    return org_dataset1
org_dataset1 = build_dataset()
training_data=[]
org_dataset=[]
for i in range(165):
    if i%11==2:
        training_data.append(org_dataset1[i,:])
    else:
        org_dataset.append(org_dataset1[i,:])
org_dataset = np.array(org_dataset) 
training_data = np.array(training_data)
print(training_data.shape)
print(org_dataset.shape)
#print(a)
def normalize(org_dataset):
    mean_vector = np.mean(org_dataset, axis=0)
    dataset = org_dataset - mean_vector
    return dataset, mean_vector

dataset, mean_vector = normalize(org_dataset)
training_dataset, mean_vector1 = normalize(training_data)
from numpy import linalg as la

def calc_eig_val_vec(dataset):
    cov_mat = np.dot(dataset, dataset.T)
    eig_values, eig_vectors = la.eig(cov_mat)
    #eig_vectors = np.dot(dataset.T, eigen_vectors)
    #print(eig_vectors.shape)
    #for i in range(eig_vectors.shape[1]):
        #eig_vectors[:, i] = eig_vectors[:, i]/la.norm(eig_vectors[:, i])
    idx = eig_values.argsort()[::-1]   
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:,idx]
    return eig_values.astype(float), eig_vectors.astype(float)

eig_values, eig_vectors = calc_eig_val_vec(dataset)
S=np.sum(eig_values)
sum_ev=0
for i in range(150):
    sum_ev=sum_ev+eig_values[i]
    norm_ev=sum_ev/S
    if norm_ev>=.90:
        break
print(i)    # no of vectors to get 90% energy
eig_faces = np.matmul(dataset.T, eig_vectors[:,:i])
eig_faces=eig_faces.T
print(eig_faces.shape)
eigen_faces=np.zeros((231,195))
for k in range(5):   # displaying only 5 out of 36 faces
    eigen_faces=np.reshape(eig_faces[k,:],(231,195))
    plt.figure()
    plt.imshow(eigen_faces)
weight=np.matmul(eig_faces,training_dataset.T) 
weight=weight.T
print(weight.shape)
for k in range(15):
    training_datanew=np.reshape(training_data[k,:],(231,195))
    plt.figure()
    plt.imshow(training_datanew)

    sum_image=np.matmul(weight[k,:],eig_faces)
    reconstructed_image=np.reshape(sum_image,(231,195))
    plt.figure()
    plt.imshow(reconstructed_image)