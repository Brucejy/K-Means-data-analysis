
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from sklearn.cluster import KMeans


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


from sklearn.externals import joblib


# In[6]:


from pandas import Series


# In[7]:


from sklearn import preprocessing


# In[8]:


from sklearn import cluster


# In[9]:


f=pd.read_csv("/home/bruce/JupyterData/item_feature2.csv")


# In[10]:


dataframe=pd.DataFrame({'item_id':f.item_id,'cate_level_id':f.cate_level_id,'pv_ipv':f.pv_ipv,'amt_gmv':f.amt_gmv},columns=['item_id','cate_level_id','pv_ipv','amt_gmv'])


# In[11]:


dataframe.to_csv("/home/bruce/JupyterData/item_feature3.csv")


# In[12]:


grouped=dataframe.groupby(['item_id','cate_level_id'])


# In[13]:


dy=grouped.sum()


# In[14]:


db=preprocessing.scale(dy.pv_ipv)


# In[15]:


dB=Series(db)


# In[16]:


da=preprocessing.scale(dy.amt_gmv)


# In[17]:


dA=Series(da)


# In[18]:


dy.to_csv("/home/bruce/JupyterData/item_feature5.csv")


# In[19]:


dh=pd.read_csv("/home/bruce/JupyterData/item_feature5.csv")


# In[20]:


dz=pd.DataFrame({'item_id':dh.item_id,'cate_level_id':dh.cate_level_id,'pv_ipv':dB,'amt_gmv':dA},columns=['item_id','cate_level_id','pv_ipv','amt_gmv'])


# In[21]:


dz


# In[22]:


data=[dz.pv_ipv,dz.amt_gmv]


# In[23]:


datat=np.transpose(data)


# In[24]:


SSE=[]
for k in range(2,10):
    clf=KMeans(n_clusters=k)
    s=clf.fit(datat)
    numSamples=len(datat)
    label=clf.labels_
    print(label)
    type(label)
    print(clf.inertia_)
    SSE.append(clf.inertia_)
    mark=['or','ob','og','ok','dc','dy','^r','+k','<g','pr']
    for i in range(numSamples):
        plt.plot(datat[i][0],datat[i][1],mark[clf.labels_[i]])
    mark=['or','ob','og','ok','dc','dy','^r','+k','<g','pr']
    centroids=clf.cluster_centers_
    for i in range(k):
        plt.plot(centroids[i][0],centroids[i][1],mark[i],markersize=12)
    plt.show()


# In[25]:


X=range(2,10)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')
plt.show()


# In[26]:


print('k=4 is the best')

