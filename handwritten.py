#!/usr/bin/env python
# coding: utf-8

# In[117]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[118]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[119]:


len(X_train)


# In[120]:


X_train[0].shape


# In[121]:


X_train[0]


# In[122]:


plt.matshow(X_train[0])


# In[123]:


flatrain = X_train.reshape(len(X_train),28*28)
flatest = X_test.reshape(len(X_test),28*28)


# In[124]:


flatrain = flatrain/255
flatest=flatest/255


# In[125]:


flatest.shape


# In[126]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,),activation='sigmoid')
])
tb = tf.keras.callbacks.TensorBoard(log_dir="logs/",histogram_freq=1)
model.compile(
optimizer='adam',
 loss='sparse_categorical_crossentropy',
metrics = ['accuracy']
)
model.fit(flatrain,y_train,epochs=5,callbacks=[tb])


# In[127]:


model.evaluate(flatest,y_test)


# In[128]:


plt.matshow(X_test[1])


# In[129]:


y = model.predict(flatest)


# In[130]:


y[1]


# In[131]:


np.argmax(y[1])


# In[132]:


y_labels = [np.argmax(i) for i in y]
y_labels[:5]


# In[133]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_labels)
cm


# In[134]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm,annot=True , fmt='d')
plt.xlabel('predicted')
plt.ylabel('truth')


# In[135]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,),activation='relu'),
    keras.layers.Dense(10, input_shape=(784,),activation='sigmoid')
])
model.compile(
optimizer='adam',
 loss='sparse_categorical_crossentropy',
metrics = ['accuracy']
)
model.fit(flatrain,y_train,epochs=5)


# In[136]:


model.evaluate(flatest,y_test)


# In[137]:


yu = model.predict(flatest)


# In[138]:


yu_labels = [np.argmax(p) for p in yu]


# In[139]:


cm1 = tf.math.confusion_matrix(labels=y_test, predictions = yu_labels)
cm1


# In[140]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm1,annot=True,fmt='d')
plt.xlabel('predictions')
plt.ylabel('truth')



# In[ ]:





# In[ ]:





# In[ ]:




