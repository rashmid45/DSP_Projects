# Loading all the required Library and Packages

import os
os.chdir('C:\\Users\\shardul\\Desktop\\Rashmi\\Clustering\\Hierarchical Clustering')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering



# importing google review ratings data set
gd1 = pd.read_csv('google_review_ratings.csv')

gd1.dtypes
gd1.shape # (5456, 25)
gd=gd1
gd= pd.DataFrame(gd)
type(gd)


gd  = gd.iloc[:,1:24]
gd.info
gd.shape # (5426, 23)
gd.head(3)


#checking data types of all variables 
gd.dtypes
'''
Out[102]: 
Category_1     float64
Category_2     float64
Category_3     float64
Category_4     float64
Category_5     float64
Category_6     float64
Category_7     float64
Category_8     float64
Category_9     float64
Category_10    float64
Category_12    float64
Category_13    float64
Category_14    float64
Category_15    float64
Category_16    float64
Category_17    float64
Category_18    float64
Category_19    float64
Category_20    float64
Category_21    float64
Category_22    float64
Category_23    float64
Category_24    float64
dtype: object
'''
#  Checking for missing values 
gd.isna().sum() #  missing values in Category 12 and Category 24
'''
Out[116]: 
Category_1        0
Category_2        0
Category_3        0
Category_4        0
Category_5        0
Category_6        0
Category_7        0
Category_8        0
Category_9        0
Category_10       0
Category_12       1
Category_13       0
Category_14       0
Category_15       0
Category_16       0
Category_17       0
Category_18       0
Category_19       0
Category_20       0
Category_21       0
Category_22       0
Category_23       0
Category_24       1
Unnamed: 24    5454
dtype: int64
'''


# Impute missing Values with mean in Category_12  and Category_24

gd['Category_12'] = gd['Category_12'].fillna(gd['Category_12'].mean())
gd['Category_12'].describe() # count = 5456

gd['Category_24'] = gd['Category_24'].fillna(gd['Category_24'].mean())
gd['Category_24'].describe() # count = 5456


gd.isnull().sum() 

gd.to_csv('gd.csv')

# Scaling the data
from sklearn import preprocessing
scaled_data_gd =preprocessing.normalize(gd)
scaled_data_gd = pd.DataFrame(scaled_data_gd,columns =  gd.columns)

gd.head(scaled_data_gd)



'''
Lets start making clusters by applying different clustering methods 
namely Ward, Single,Complete and Average 
'''


### USING WARD LINKAGE

cluster =  AgglomerativeClustering(n_clusters =2 ,affinity ='euclidean', linkage='ward')
cluster.fit_predict(scaled_data_gd)

## plotting the scatterplot
plt.figure(figsize = (10,7))
plt.scatter(scaled_data_gd['Category_5'],scaled_data_gd['Category_6'],c=cluster.labels_)

## build dendograms
plt.figure(figsize = (10,7))
plt.title('Dendograms')
dend = shc.dendrogram(shc.linkage(scaled_data_gd,method ='ward'))

## Cophenetic correlation Coefficient
Z_ward=shc.linkage(scaled_data_gd,method ='ward')
c, coph_dists = cophenet(Z_ward, pdist(scaled_data_gd))
print ( c,coph_dists) # 0.52 



### USING SINGLE LINKAGE

cluster =  AgglomerativeClustering(n_clusters =2 ,affinity ='euclidean', linkage='single')
cluster.fit_predict(scaled_data_gd)

## plotting the scatterplot
plt.figure(figsize = (10,7))
plt.scatter(scaled_data_gd['Category_5'],scaled_data_gd['Category_6'],c=cluster.labels_)


## build dendogram
plt.figure(figsize = (10,7))
plt.title('Dendograms')
dend = shc.dendrogram(shc.linkage(scaled_data_gd,method ='single'))


# Cophenetic correlation coefficient
Z_single =shc.linkage(scaled_data_gd,method ='single')
c, coph_dists_s = cophenet(Z_single, pdist(scaled_data_gd))
print (c,coph_dists_s)
#0.4572855805563199 [0.02550139 0.02550139 0.04654327 ... 0.31218498 0.11967217 0.31218498]


### USING COMPLETE LINKAGE

cluster =  AgglomerativeClustering(n_clusters =2 ,affinity ='euclidean', linkage='complete')
cluster.fit_predict(scaled_data_gd)

## plotting the scatterplot
plt.figure(figsize = (10,7))
plt.scatter(scaled_data_gd['Category_5'],scaled_data_gd['Category_6'],c=cluster.labels_)

## build dendogram
plt.figure(figsize = (10,7))
plt.title('Dendograms')
dend = shc.dendrogram(shc.linkage(scaled_data_gd,method ='complete'))

# Cophenetic correlation coefficient
Z_complete =shc.linkage(scaled_data_gd,method ='complete')
c, coph_dists_c = cophenet(Z_complete, pdist(scaled_data_gd))
print(c,c, coph_dists_c)
#0.5518323109371003 0.5518323109371003 [0.02638123 0.02638123 0.05336658 ... 0.47285664 0.11967217 0.47285664]

### USING AVERAGE LINKAGE
from sklearn.cluster import AgglomerativeClustering
cluster =  AgglomerativeClustering(n_clusters =2 ,affinity ='euclidean', linkage='average')
cluster.fit_predict(scaled_data_gd)

## plotting the scatterplot
plt.figure(figsize = (10,7))
plt.scatter(scaled_data_gd['Category_5'],scaled_data_gd['Category_6'],c=cluster.labels_)

## build dendogram
plt.figure(figsize = (10,7))
plt.title('Dendograms')
dend = shc.dendrogram(shc.linkage(scaled_data_gd,method ='average'))

# Cophenetic correlation coefficient
Z_avg =shc.linkage(scaled_data_gd,method ='average')
c, coph_dists_a = cophenet(Z_avg, pdist(scaled_data_gd))
print (c, coph_dists_a)
#0.6946179499953213 [0.02609734 0.02609734 0.05190216 ... 0.43274971 0.11967217 0.43274971]


