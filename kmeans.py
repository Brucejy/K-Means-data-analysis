
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from pandas import Series
from sklearn import preprocessing
from sklearn import cluster


# In[2]:


f=pd.read_csv("C:\Users\ASUS\Documents\Python\data\item_feature2.csv")


# In[3]:


dataframe=pd.DataFrame({'CommodityID':f.CommodityID,'BigItemID':f.BigItemID,'BrowseOrder':f.BrowseOrder,'AuctionedAmount':f.AuctionedAmount},columns=['CommodityID','BigItemID','BrowseOrder','AuctionedAmount'])


# In[4]:


dataframe.to_csv("C:\Users\ASUS\Documents\Python\data\item_feature3.csv")


# In[5]:


dataframe=pd.read_csv("C:\Users\ASUS\Documents\Python\data\item_feature3.csv")


# In[6]:


grouped=dataframe.groupby(['CommodityID', 'BigItemID'])


# In[7]:


dy=grouped.sum()


# In[8]:


db=preprocessing.scale(dy.BrowseOrder)
dB=Series(db)


# In[9]:


da=preprocessing.scale(dy.AuctionedAmount)
dA=Series(da)
dy.to_csv("C:\Users\ASUS\Documents\Python\data\item_feature5.csv")


# In[10]:


dh=pd.read_csv("C:\Users\ASUS\Documents\Python\data\item_feature5.csv")


# In[11]:


dz=pd.DataFrame({'CommodityID':dh.CommodityID,'BigItemID':dh.BigItemID,'BrowseOrder':dB,'AuctionedAmount':dA},columns=['CommodityID','BigItemID','BrowseOrder','AuctionedAmount'])


# In[12]:


dz


# In[13]:


data=[dz.BrowseOrder, dz.AuctionedAmount]


# In[14]:


datat=np.transpose(data)


# In[15]:


SSE=[]
for k in range(2,10):
    clf=KMeans(n_clusters=k)
    s=clf.fit(datat)
    numSamples=len(datat)
    centroids=clf.labels_
    print centroids,type(centroids)
    print clf.inertia_
    SSE.append(clf.inertia_)
    mark=['or','ob','og','ok','^r','+r','sr','dr','<r','pr']
    for i in xrange(numSamples):
        plt.plot(datat[i][0],datat[i][1],mark[clf.labels_[i]])
    mark=['Dr','Db','Dg','Dk','^b','+b','sb','db','<b','pb']
    centroids=clf.cluster_centers_
    for i in range(k):
        plt.plot(centroids[i][0],centroids[i][1],mark[i],markersize=12)
    plt.show()


# In[16]:


X=range(2, 10)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE,'o-')
plt.show()


# In[17]:


print('k=4 is the best')

