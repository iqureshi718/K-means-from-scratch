import pandas as pd
import math
import numpy as np
import statistics
import random
from collections import Counter
import matplotlib.pyplot as plt
data=pd.read_csv("/Users/imanqureshi/Downloads/breast-cancer-wisconsin.csv",header=None)
df=pd.DataFrame(data)
df.replace('?',np.nan,inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df[6]=df[6].astype(str).astype('int64')
def normalize_data(dataset):
    normalized_data = ((dataset-dataset.min())/(dataset.max()-dataset.min()))
    normalized_data[0]=dataset[0]
    normalized_data[10]=dataset[10]
    return normalized_data
normalized_data=normalize_data(df)
arraydata=np.array(normalized_data)
normalized_data=normalized_data.sample(frac=1)
normalized_list_uncleaned=normalized_data.values.tolist()
random.seed(3)
random.shuffle(normalized_list_uncleaned)
normalized_list=[]
for item in normalized_list_uncleaned:
    if(len(item)==11):
        normalized_list.append(item)
def stratify(normalized_list):
    malignent_df=[]
    benign_df=[]
    for row in normalized_list:
        if(row[10]==2):
            benign_df.append(row)
        else:
            malignent_df.append(row)
    size_bengin=len(benign_df)
    size_malig=len(malignent_df)
    size_training_benign=round(.7*size_bengin)
    size_training_malig=round(.7*size_malig)
    size_val_benign=round(.2*size_bengin)
    size_val_malig=round(.2*size_malig)
    size_test_benign=size_bengin-size_training_benign-size_val_benign
    size_test_malig=size_malig-size_training_malig-size_val_malig
    train=[]
    val=[]
    test=[]
    for i in range(size_training_benign):
        train.append(benign_df[i])
    for i in range(size_training_malig):
        train.append(malignent_df[i])
    for i in range(size_training_benign,size_training_benign+size_val_benign):
        val.append(benign_df[i])
    for i in range(size_training_malig,size_training_malig+size_val_malig):
        val.append(malignent_df[i])
    for i in range(size_val_benign+size_training_benign,size_bengin):
        test.append(benign_df[i])
    for i in range(size_val_malig+size_training_malig,size_malig):
        test.append(malignent_df[i])
    return train,val,test
train,val,test=stratify(normalized_list)
def euclidean_distance(num1, num2):
    distance=0
        # return np.sqrt(np.sum((num1-num2)**2))
    for i in range(len(num1)):
        try:
            distance+=(num1[i]-num2[i])**2
        except print(num1,num2):
            pass
    # print(distance)
    return np.sqrt(distance)
def empty_clusters(k):
    clusters = {}
    for i in range(k):
        clusters[i] = []
    return clusters
def k_means(k, train):
    centroids = random.sample(train, k)
    c1=[]
    for cen in centroids:
        c1.append(cen[1:9])
    new_centroids = []
    clusters = empty_clusters(k)
    counter = 0
    # print("This is c1: ",c1,"This is centroids: "  ,centroids)
    while(counter < 10000):
        new_centroids=[]
        # print(c1[0])
        clusters = empty_clusters(k)
        for row in train:
            distances = []
            for centroid in c1:
                distances.append(euclidean_distance(row[1:9], centroid))
            if(len(row)==11):
                clusters[distances.index(min(distances))].append(row)
            else:
                print(row)
        for i in range(k):
            if(len(clusters[i]) != 0):
                temp=(np.mean(clusters[i], axis=0)).tolist()[1:9]
                # print(temp)
                new_centroids.append(temp)
            else:
                new_centroids.append(c1[i])
        counter += 1
        if(c1 == new_centroids):
            # print(c1[0],new_centroids[0])
            # print("New Centroids: ", new_centroids)
            break
        c1 = new_centroids
    # print(counter)
    return clusters, new_centroids
def optimal_k(train):
    accuracy_list=[]
    for k in range(1,len(train)):
        clust, cent = (k_means(k, train))
        # majority_cluster={}
        majority_class=[]
        for i in range(len(cent)):
            majority_list=[]
            for row in clust[i]:
                majority_list.append(row[10])
            if(len(clust[i])!=0):
                majority=statistics.mode(majority_list)
            else:
                majority=4
            majority_class.append(majority)
        predicted=[]
        actual=[]
        for row in val:
            distances=[]
            for c in cent:
                distances.append(euclidean_distance(row[1:9],c))
            predicted.append(majority_class[distances.index(min(distances))])
            actual.append(row[10])
        accuracy=0
        for i in range(len(predicted)):
            if(predicted[i]==actual[i]):
                accuracy+=1
        accuracy=accuracy/len(predicted)
        print('Accurracy',accuracy*100,'For K value: ',k)
        accuracy_list.append(accuracy)
        bestK=accuracy_list.index(max(accuracy_list))+1
    # return max(accuracy_list),accuracy_list.index(max(accuracy_list))
    print('Best accuracy: ',max(accuracy_list)*100,bestK)
    return bestK
#optimal_k(test)

def optimal_kTest(test):
    p=optimal_k(train)
#
    accuracy_list=[]
#   for k in range(0,1):
    clust, cent = (k_means(p, test))
    # majority_cluster={}
    majority_class=[]
    for i in range(len(cent)):
        majority_list=[]
        for row in clust[i]:
            majority_list.append(row[10])
        if(len(clust[i])!=0):
            majority=statistics.mode(majority_list)
        else:
            majority=4
        majority_class.append(majority)
    predicted=[]
    actual=[]
    for row in val:
        distances=[]
        for c in cent:
            distances.append(euclidean_distance(row[1:9],c))
        predicted.append(majority_class[distances.index(min(distances))])
        actual.append(row[10])
    accuracy=0
    for i in range(len(predicted)):
        if(predicted[i]==actual[i]):
            accuracy+=1
    accuracy=accuracy/len(predicted)
    print('ACCURACY FOR TEST SET: ',accuracy*100,'WITH BEST K VALUE:',p)
    accuracy_list.append(accuracy)
# return max(accuracy_list),accuracy_list.index(max(accuracy_list))
optimal_kTest(test)
