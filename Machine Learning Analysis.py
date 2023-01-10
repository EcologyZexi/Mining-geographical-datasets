#!/usr/bin/env python
# coding: utf-8

# ## GEOG0051 Mining Social and Geographic Datasets
# 
# ## Coursework Part Two Machine Learning Analysis with Venue Review Data in Calgary, Canada
# 
# For this second task, we would like you to analyse a dataset that contains review data of different venues in the city of Calgary, Canada. With the help of several machine learning techniques that we have learnt in the course, you will be tasked to distill insights from this social media dataset. Two of its notable features are the geocoding of every reviewed venues and the availability of a considerable amount of text data in it, which lend to its ability to be processed using spatial and text analysis techniques respectively. 
# 
# 
# As a prelude to the analysis prompts below, have a brief think about some of these questions: 
# 
# What can we discover about the venue review data? 
# 
# Are there any spatial patterns that can be extracted from the data?
# 
# Can we build a machine learning model that predicts review rating for unseen data points using the text of the reviews?
# 

# In[1]:


# suppress warning 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
myfont = matplotlib.font_manager.FontProperties(fname='heiti.ttf',size=40)
plt.rcParams['axes.unicode_minus']=False

#One-hot
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#Word2Vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors

#MLP
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

import nltk
from nltk.stem import WordNetLemmatizer

from nltk.stem import porter as pt
from nltk.corpus import stopwords

import joblib
import glob
import os
import re
import string

import geopandas as gpd
import contextily as ctx


# ## 1	Loading and cleaning the textual dataset
# 
# In a realistic context, most text datasets are messy in their raw forms. They require considerable data cleaning before any analysis can be conducted and, not unlike data cleaning for non-textual datasets, this would include the removal of invalid data, missing values, and outliers. In this first prompt you will be required to complete the tasks stated below to prepare the dataset for subsequent analysis.
# 
# 1.Load and understand the dataset.
# 
# 2.Think about which attributes you will use / focus on (in subsequent prompts) and check its data distribution.
# 
# 3.Pre-process the text review data and create a new column in the data frame which will hold the cleaned review data.
# 
# 4.Some of the steps to consider are: removal of numbers, punctuation, short words, stopwords, lemmatise words, etc.
# 
# Example Pipeline
# for each review in geoDataFrame:
#     # removes all numbers (hint: re)
#     # removes all punctuations (hint: re)
#     # removes short words (hint: re
#     # tokenize words (hint:nltk)
#     # removes stopwords (hint: nltk)
#     # lemmatize (hint: nltk)
#     # rejoin as sent
#     # cleantxt = sent

# **1.1 Load and understand the dataset.**

# In[2]:


# load dataset
df = pd.read_csv('df_Calgary_pre.csv')


# In[3]:


len(df['categories'].unique())


# In[4]:


len(df['name'].unique()) 


# In[5]:


df.head()


# **1.2 Think about which attributes will be used and check its data distribution**

# In[6]:


# Return DataFrame with duplicate rows removed.
df.drop_duplicates()
df.describe()


# In[9]:


fig, ax = plt.subplots(2, 2,figsize=(12,6))  
#ax[0][0].set_title('stars_y')
ax[0][0].set(xlabel='stars',ylabel='count')
ax[0][0].hist(df['stars_y'])

#ax[0][1].set_title('useful')
ax[0][1].set(xlabel='useful',ylabel='count')
ax[0][1].hist(df['useful'])

#ax[1][0].set_title('funny')
ax[1][0].set(xlabel='funny',ylabel='count')
ax[1][0].hist(df['funny'])

ax[1][1].set(xlabel='cool',ylabel='count')
#ax[1][1].set_title('cool')
ax[1][1].hist(df['cool'])

plt.show()


# In[10]:


fig, ax = plt.subplots(figsize=(8,6))  
#ax.set_title('review count')
ax.set(xlabel='review',ylabel='count')
ax.hist(df['review_count'])


# **1.3 Pre-process the text review data and create a new column in the data frame which will hold the cleaned review data.**

# In[11]:


df2 = df.copy()


# In[12]:


df2.head(n=2)


# In[13]:


# removal of numbers, punctuation, short words, stopwords, lemmatise words, etc.
def clean_text(text):
    stop_words = stopwords.words("english")

    text = text.lower()
    text = text.encode('ascii', 'ignore').decode()
    text = ' '.join([word for word in text.split(' ') if word not in stop_words])# Remove stop words
    text = re.sub(r'https*\S+', ' ', text) # Remove URL
    text = re.sub(r'@\S+', ' ', text) # Remove mentions
    text = re.sub(r'#\S+', ' ', text) # Remove Hashtags
    text = re.sub(r'\'\w+', '', text) # Remove ticks and the next character
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)# Remove punctuations
    text = re.sub(r'\w*\d+\w*', '', text) # Remove numbers
    text = re.sub(r'\s{2,}', ' ', text) # Replace the over spaces

    return text


# In[14]:


import nltk 
nltk.download('wordnet') 
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')

df2['cleaned_text']=np.NaN
df2['cleaned_categories']=np.NaN

for i in range(len(df2)):
    text = df2.iloc[i]['text']
    text = clean_text(text)
    wnl = WordNetLemmatizer()
    word_list1 = nltk.word_tokenize(text)
    text = ' '.join([wnl.lemmatize(words) for words in word_list1])
    df2.loc[i,'cleaned_text'] = text
    
    category = df2.iloc[i]['categories']
    category = clean_text(category)
    word_list2 = nltk.word_tokenize(category)
    category = ' '.join([wnl.lemmatize(words) for words in word_list2])
    df2.loc[i,'cleaned_categories'] = category    
    
    
df2.head()


# *Example for cleaned text

# In[15]:


a1 = df2.loc[0,'text']
print(a1)


# In[16]:


a1_clean = df2.loc[0,'cleaned_text']
print(a1_clean)


# In[17]:


print("previous categories：{}，cleaned categories:{}".format(len(df2['categories'].drop_duplicates()), len(df2['cleaned_categories'].drop_duplicates())))


# ## 2	 Build a supervised learning model for text analysis 
# 
# The objective of this sub-task is to build a supervised learning model that predicts the polarity (positive or negative) of the venue from the data, based on the different features of each review included in the dataset. Positive polarity here is defined as a venue rating of 4 or more stars and negative polarity here is defined as a venue rating of 3 or less stars. You can choose a subset of venues to review for example based on a general category(use) the venue falls under. You can use a combination of text and non-text features, and below are some guidelines that you could follow:
# 
# 
# 
# • Firstly, tokenize the pre-processed review text data to give a bag-of-words feature that can be used in your model. 
# 
# • Create polarity score from the stars rating. 
# 
# • Split dataset (eg. train and test-set). 
# 
# • Train and compare the efficacy of not fewer than two machine learning models predicting its polarity. 
# 
# • Report the model results (on out-of-sample testset).
# 
# • Discuss and interpret the results you obtained.

# **2.1 tokenize the pre-processed review text data to give a bag-of-words feature that can be used in your model**

# ** subset - extract the most category

# In[18]:


ls_group = []
count=[]
for name, group in df2.groupby('cleaned_categories'):
    num = len(group)
    if num > 300:
        ls_group.append(name)
        count.append(num)
        print(name, num)


# In[19]:


df21 = pd.DataFrame()
df21['count'] = pd.DataFrame(count,columns=['count'])
df21['categories']=pd.DataFrame(ls_group,columns=['categories'])
df21.sort_values("count",inplace=True)
df21


# In[20]:


df21.set_index('categories').plot.barh(figsize=(8,8),color='black')


# In[23]:


df_sub =df2[df2['cleaned_categories'].str.contains('restaurant')==True]


# In[24]:


df_sub


# In[25]:


df_sub = df_sub.reset_index()


# In[26]:


len(df_sub)


# ** countvectorizer

# In[27]:


vectorizer = CountVectorizer(min_df=0.01,max_df=0.95)
X_vc = vectorizer.fit_transform(df_sub['cleaned_text'])
print(vectorizer.get_feature_names())
print(X_vc.toarray())


# In[28]:


print(X_vc[0].toarray)


# In[29]:


pd.DataFrame(X_vc.toarray(),columns=[vectorizer.get_feature_names()])


# ** tf-idf

# In[35]:


# convert cleaned_text to tf-idf vector
tf_vectorizer = TfidfVectorizer(min_df=0.01,max_df=0.95)
X_tf_idf = tf_vectorizer.fit_transform(df_sub['cleaned_text'])  


# In[36]:


print(X_tf_idf.toarray)


# In[37]:


pd.DataFrame(X_tf_idf.toarray(),columns=[tf_vectorizer.get_feature_names()])


# **2.2 Create polarity score from the stars rating**

# In[38]:


# Identify Pos and Neg from stars
df_sub['polarity'] = None
df_sub['polarity'][df_sub['stars_y']>=4] = 'pos'
df_sub['polarity'][df_sub['stars_y']<=3] = 'neg'

polarity = []
for name, group in df_sub.groupby('polarity'):
    polarity.append(name)
    print(name, len(group))


# **2.3 Split dataset (eg. train and test-set)**

# ** countvectorizer

# In[39]:


# data used in model
data1 = X_vc.toarray()  # train data (independent)
label1 = df_sub['polarity']  #  label (dependent)

# split train and test data based on index, to find the corresponding location
train_idx1, test_idx1 = train_test_split(range(len(df_sub)), test_size=0.2, random_state=0)

# model data according to index
X_train1, X_test1 = data1[train_idx1], data1[test_idx1]
y_train1, y_test1 = label1[train_idx1], label1[test_idx1]


# In[40]:


len(X_test1)


# In[41]:


len(X_train1)


# ** tf-idf

# In[43]:


# data used in model
data2 = X_tf_idf.toarray()  # train data (independent)
label2 = df_sub['polarity']  #  label (dependent)

# split train and test data based on index, to find the corresponding location
train_idx2, test_idx2 = train_test_split(range(len(df_sub)), test_size=0.2, random_state=0)

# model data according to index
X_train2, X_test2 = data2[train_idx2], data2[test_idx2]
y_train2, y_test2 = label2[train_idx2], label2[test_idx2]


# **2.4 Train and compare the efficacy of not fewer than two machine learning models predicting its polarity**

# 1) MLP

# **countvectorizer

# In[44]:


#MLP-Multi-layer Perceptron classifier.
mlp1 = MLPClassifier(solver='adam', activation = 'tanh', max_iter = 200, 
                    random_state = 1, verbose = False, alpha = 1e-4)
# model training
history = mlp1.fit(X_train1, y_train1)

# save the model
joblib.dump(mlp1, "MLP")

y_pred11 = mlp1.predict(X_test1)


# In[45]:


# accuracy
from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix,mean_squared_error

print("Accuracy: %.2f"% accuracy_score(y_test1, y_pred11))
print("F-score: %.2f"% f1_score(y_test1, y_pred11, average='macro'))
#print('REC: %.2f'% recall_score(y_test1, y_pred11, average='macro'))      
#print('MSE: %.2f'% mean_squared_error(y_test, y_pred1))
#print('RMSE: %.2f'% np.sqrt(mean_squared_error(y_test, y_pred1)))


# In[76]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
#creates a grid to plot on
f, ax = plt.subplots(figsize=(7, 5))
#plots confusion matrix
sns.heatmap(confusion_matrix(y_test1, y_pred11),
            annot=True, 
            fmt="d", 
            linewidths=.5, 
            cmap="YlGnBu",
            xticklabels=['neg','pos'], 
            yticklabels=['neg','pos']
           )

plt.title('Confusion Matrix for MLP Prediction (BOW)' +'\n')

plt.show()


# **tf-idf

# In[47]:


#MLP-Multi-layer Perceptron classifier.
mlp2 = MLPClassifier(solver='adam', activation = 'tanh', max_iter = 200, 
                    random_state = 1, verbose = False, alpha = 1e-4)
# model training
history = mlp2.fit(X_train2, y_train2)

# save the model
joblib.dump(mlp2, "MLP")

y_pred21 = mlp2.predict(X_test2)


# In[48]:


# accuracy
from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix,mean_squared_error

print("ACC: %.2f"% accuracy_score(y_test2, y_pred21))
print("F-score: %.2f"% f1_score(y_test2, y_pred21, average='macro'))
#print('REC: %.2f'% recall_score(y_test2, y_pred21, average='macro'))      
#print('MSE: %.2f'% mean_squared_error(y_test, y_pred1))
#print('RMSE: %.2f'% np.sqrt(mean_squared_error(y_test, y_pred1)))


# In[77]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
#creates a grid to plot on
f, ax = plt.subplots(figsize=(7, 5))
#plots confusion matrix
sns.heatmap(confusion_matrix(y_test2, y_pred21),
            annot=True, 
            fmt="d", 
            linewidths=.5, 
            cmap="YlGnBu",
            xticklabels=['neg','pos'], 
            yticklabels=['neg','pos']
           )

plt.title('Confusion Matrix for MLP Prediction (TF-IDF)' +'\n')

plt.show()


# **2) RF**

# **countvectorizer

# In[50]:


rf1 = RandomForestClassifier(n_estimators=200, n_jobs = -1)

rf1.fit(X_train1, y_train1)

joblib.dump(rf1, "rf")

y_pred12 = rf1.predict(X_test1)


# In[51]:


from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix,mean_squared_error

print("Accuracy: %.2f"% accuracy_score(y_test1, y_pred12))
print("F-score: %.2f"% f1_score(y_test1, y_pred12, average='macro'))
#print('REC: %.2f'% recall_score(y_test1, y_pred12, average='macro'))      


# In[78]:


#creates a grid to plot on
f, ax = plt.subplots(figsize=(7, 5))

#plots confusion matrix
sns.heatmap(confusion_matrix(y_test1, y_pred12),
            annot=True, 
            fmt="d", 
            linewidths=.5, 
            cmap="YlGnBu",
            xticklabels=['neg','pos'], 
            yticklabels=['neg','pos']
           )

plt.title('Confusion Matrix for RF Prediction (BOW)' +'\n')

plt.show()


# **tf-idf

# In[53]:


rf2 = RandomForestClassifier(n_estimators=200, n_jobs = -1)

rf2.fit(X_train2, y_train2)

joblib.dump(rf2, "rf")

y_pred22 = rf2.predict(X_test2)


# In[54]:


from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix,mean_squared_error

print("Accuracy: %.2f"% accuracy_score(y_test2, y_pred22))
print("F-score: %.2f"% f1_score(y_test2, y_pred22, average='macro'))
#print('REC: %.2f'% recall_score(y_test2, y_pred22, average='macro'))      


# In[79]:


#creates a grid to plot on
f, ax = plt.subplots(figsize=(7, 5))

#plots confusion matrix
sns.heatmap(confusion_matrix(y_test2, y_pred22),
            annot=True, 
            fmt="d", 
            linewidths=.5, 
            cmap="YlGnBu",
            xticklabels=['neg','pos'], 
            yticklabels=['neg','pos']
           )

plt.title('Confusion Matrix for RF Prediction (TF-IDF)' +'\n')

plt.show()


# (3)NB

# **countvectorizer

# In[58]:


from sklearn.naive_bayes import MultinomialNB
# this fit the Naive Bayes Classifier
clf1 = MultinomialNB()
clf1.fit(X_train1, y_train1)

# this uses the Naive Bayes Classifier to predict
y_pred13 = clf1.predict(X_test1)


# In[59]:


from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix,mean_squared_error

print("ACC: %.2f"% accuracy_score(y_test1, y_pred13))
print("F-score: %.2f"% f1_score(y_test1, y_pred13, average='macro'))
#print('REC: %.2f'% recall_score(y_test1, y_pred31, average='macro'))      


# In[80]:


#creates a grid to plot on
f, ax = plt.subplots(figsize=(7, 5))

#plots confusion matrix
sns.heatmap(confusion_matrix(y_test1, y_pred13),
            annot=True, 
            fmt="d", 
            linewidths=.5, 
            cmap="YlGnBu",
            xticklabels=['neg','pos'], 
            yticklabels=['neg','pos']
           )

plt.title('Confusion Matrix for MultiNB Prediction (BOW)' +'\n')

plt.show()


# **tf-idf

# In[61]:


from sklearn.naive_bayes import MultinomialNB
# this fit the Naive Bayes Classifier
clf2 = MultinomialNB()
clf2.fit(X_train2, y_train2)

# this uses the Naive Bayes Classifier to predict
y_pred23 = clf2.predict(X_test2)


# In[62]:


from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix,mean_squared_error

print("Accuracy: %.2f"% accuracy_score(y_test2, y_pred23))
print("F-score: %.2f"% f1_score(y_test2, y_pred23, average='macro'))
#print('REC: %.2f'% recall_score(y_test2, y_pred32, average='macro'))      


# In[81]:


#creates a grid to plot on
f, ax = plt.subplots(figsize=(7, 5))

#plots confusion matrix
sns.heatmap(confusion_matrix(y_test2, y_pred23),
            annot=True, 
            fmt="d", 
            linewidths=.5, 
            cmap="YlGnBu",
            xticklabels=['neg','pos'], 
            yticklabels=['neg','pos']
           )

plt.title('Confusion Matrix for MultiNB Prediction (TF-IDF)' +'\n')

plt.show()


# ** unseen data

# In[70]:


#creates two dummy strings for the model to predict on
docs_new = ['food is delicious', 'good seat','serving is litte slow']
X_new_counts = tf_vectorizer.transform(docs_new)

predictions=clf1.predict(X_new_counts)

#prints the predictions of the model
predictions


# In[71]:


#creates two dummy strings for the model to predict on

predictions=clf2.predict(X_new_counts)

#prints the predictions of the model
predictions


# In[72]:


predictions=mlp1.predict(X_new_counts)

#prints the predictions of the model
predictions


# In[73]:


predictions=mlp2.predict(X_new_counts)

#prints the predictions of the model
predictions


# In[74]:


predictions=rf1.predict(X_new_counts)

#prints the predictions of the model
predictions


# In[75]:


predictions=rf2.predict(X_new_counts)

#prints the predictions of the model
predictions


# ## 3 Geospatial analysis and visualisation of review data 
# Having explored the dataset, its constituent variables and coverage above, the objective of this sub-task is for you to visualise any of the spatial patterns that emerge from the data that you find interesting. This task is intentionally open-ended and leaves you with some choice. To achieve this, you should: 
# 
# 
# • Choose 1 or 2 variables (including any variables you generated from 3.2.2) that you wish to explore and from the list of variables available in the dataset 
# 
# • Use either or both geopandas and folium libraries in Python to produce up to 3 visualisations 
# 
# • Comment on the spatial distributions of the 1-2 variables you chose, any trends or outliers that emerge and if they have any notable implications. 
# 
# • Note: You may use any subset of the dataset instead of the entire dataset, but comment on why you chose this subset. 
# 

# **3.1 Stars**

# **1) actual star value spatial distribution**

# In[82]:


df1 = df.copy()
df1.drop_duplicates()
df1.describe()


# In[83]:


df1 = df1[['name','latitude','longitude','stars_y','useful','funny','cool','text','date']]
df1.head()


# In[84]:


# Identify Pos and Neg from stars
df1['polarity'] = None

df1['polarity'][df1['stars_y']>=4] = 'pos'
df1['polarity'][df1['stars_y']<=3] = 'neg'

polarity = []
for name, group in df1.groupby('polarity'):
    polarity.append(name)
    print(name, len(group))


# In[85]:


get_ipython().system('pip install folium')
import folium
from folium import plugins
from folium.plugins import HeatMap


# In[90]:


#1 star of entire dataset inc predictions

middle_lat = df1["latitude"].median()
middle_lon = df1["longitude"].median()


map_star = folium.Map([middle_lat, middle_lon], zoom_start=15, tiles="cartodbpositron", control_scale=True)
map_star_pred = folium.Map([middle_lat, middle_lon], zoom_start=15, tiles="cartodbpositron")


# In[91]:



rate_5 = df1[(df1["stars_y"] == 5.0)]

rate_5_heat = rate_5[['latitude', 'longitude']]
rate_5_heat.describe()


heat_data = [[row['latitude'],row['longitude']] for index, row in rate_5_heat.iterrows()]

HeatMap(heat_data).add_to(map_star)


map_star


# In[92]:


map_star.save('heatmap.html')


# In[89]:


# plot star level figure
gdf = gpd.GeoDataFrame(df1, geometry=gpd.points_from_xy(df1.longitude, df1.latitude))
gdf =gdf.set_crs(epsg=4326)
gdf =gdf.to_crs(epsg=3857)
fig,ax=plt.subplots(figsize=(15, 10))

venue_plot=gdf.plot(figsize=(8, 8), alpha=0.8, linewidth=0.1,
                edgecolor='g',ax=ax,
                column='polarity', colormap = 'Reds',legend=True,
                k=2, categories=['pos','neg'])

# add the basemap
ctx.add_basemap(ax, zoom=12, source=ctx.providers.Stamen.TonerLite,alpha=0.8)

ax.set_title("The Stars of Venue Review in the City of Calgary, Canada",fontsize= 22)

# removes the axis
ax.set_axis_off()


# **2) predicted star value spatial distribution via MLP**

# In[99]:


# plot star level figure train_idx, test_idx
gdf = gpd.GeoDataFrame(df_sub.iloc[test_idx2, :], geometry=gpd.points_from_xy(df_sub.iloc[test_idx2, :]['longitude'], df_sub.iloc[test_idx2, :]['latitude']))
gdf =gdf.set_crs(epsg=4326)
gdf =gdf.to_crs(epsg=3857)
fig,ax=plt.subplots(figsize=(15, 10))

venue_plot=gdf.plot(figsize=(8, 8), alpha=1.0, linewidth=0.1,
                edgecolor='g',ax=ax,
                column='polarity', colormap = 'bwr',legend=True,
                k=2, categories=['pos','neg'])

# add the basemap
ctx.add_basemap(ax, zoom=12, source=ctx.providers.Stamen.TonerLite,alpha=0.8)

ax.set_title("The Actual Stars of Venue Review in the City of Calgary, Canada",fontsize= 22)

# removes the axis
ax.set_axis_off()


# In[98]:


import copy
df_sub_for_pred_test = copy.deepcopy(df_sub.iloc[test_idx2, :])  
df_sub_for_pred_test["pred_12"] = y_pred12

gdf = gpd.GeoDataFrame(df_sub_for_pred_test, geometry=gpd.points_from_xy(df_sub_for_pred_test['longitude'], df_sub_for_pred_test['latitude']))
gdf =gdf.set_crs(epsg=4326)
gdf =gdf.to_crs(epsg=3857)
fig,ax=plt.subplots(figsize=(20, 10))
venueplot=gdf.plot(figsize=(8, 8), alpha=0.8,linewidth=0.1,
                edgecolor='w',ax=ax,
                column='pred_12',colormap = 'bwr',legend=True,
                k=2, categories=['pos','neg'])

# add the basemap
ctx.add_basemap(ax, zoom = 12, source=ctx.providers.Stamen.TonerLite,alpha=0.8)
#ctx.add_basemap(ax, source=url,alpha=0.8)

ax.set_title("Prediction via MLP for Venue Stars Level in the City of Calgary, Canada",fontsize= 20)

# this removes the axis
ax.set_axis_off()

# this tightens the layout
fig.tight_layout()


# **3) predicted star value spatial distribution via RF**

# In[100]:


df_sub_for_pred_test1 = copy.deepcopy(df_sub.iloc[test_idx2, :])  # 深拷贝
df_sub_for_pred_test1["pred_22"] = y_pred22

gdf = gpd.GeoDataFrame(df_sub_for_pred_test1, geometry=gpd.points_from_xy(df_sub_for_pred_test1['longitude'], df_sub_for_pred_test1['latitude']))
gdf =gdf.set_crs(epsg=4326)
gdf =gdf.to_crs(epsg=3857)
fig,ax=plt.subplots(figsize=(15, 10))
houseplot=gdf.plot(figsize=(8, 8), alpha=0.8, linewidth=0.1,
                edgecolor='w',ax=ax,
                column='pred_22',colormap = 'bwr',legend=True,
                k=2, categories=['pos','neg'])

# add the basemap
ctx.add_basemap(ax, zoom=12, source=ctx.providers.Stamen.TonerLite,alpha=0.8)

ax.set_title("Prediction via RF for Venue Stars Level in the City of Calgary, Canada",fontsize= 20)

# this removes the axis
ax.set_axis_off()

# this tightens the layout
fig.tight_layout()


# **4) predicted star value spatial distribution via MultiNB**

# In[105]:


df_sub_for_pred_test1 = copy.deepcopy(df_sub.iloc[test_idx2, :])  # 深拷贝
df_sub_for_pred_test1["pred_23"] = y_pred23

gdf = gpd.GeoDataFrame(df_sub_for_pred_test1, geometry=gpd.points_from_xy(df_sub_for_pred_test1['longitude'], df_sub_for_pred_test1['latitude']))
gdf =gdf.set_crs(epsg=4326)
gdf =gdf.to_crs(epsg=3857)
fig,ax=plt.subplots(figsize=(15, 10))
houseplot=gdf.plot(figsize=(8, 8), alpha=0.8, linewidth=0.1,
                edgecolor='w',ax=ax,
                column='pred_23',colormap = 'bwr',legend=True,
                k=2, categories=['pos','neg'])

# add the basemap
ctx.add_basemap(ax, zoom=12, source=ctx.providers.Stamen.TonerLite,alpha=0.8)

ax.set_title("Prediction via MultiNB for Venue Stars Level in the City of Calgary, Canada",fontsize= 20)

# this removes the axis
ax.set_axis_off()

# this tightens the layout
fig.tight_layout()


# **3.2 tags(useful,funny,cool)**

# In[106]:


df_tag1 =  df2.iloc[:][['name','latitude','longitude',
                     'review_count','stars_y']]
df_tag1 = df_tag1.groupby(['name','latitude','longitude']).mean()
df_tag1 = df_tag1.reset_index()


# In[107]:


df_tag1.head()


# In[108]:


df_tag2 =  df2.iloc[:][['name','latitude','longitude','useful','funny','cool']]
df_tag2 = df_tag2.groupby(['name','latitude','longitude']).sum()
df_tag2 = df_tag2.reset_index()


# In[109]:


df_tag=df_tag1.merge(df_tag2, on = 'name')
df_tag.head()


# In[110]:


len(df_tag)


# In[111]:


fig, ax = plt.subplots(2, 2,figsize=(6,6))  
#ax[0][0].set_title('review count')
ax[0][0].set_xlabel('review count')
ax[0][0].set_ylabel('star rating')
ax[0][0].scatter(df_tag['review_count'],df_tag['stars_y'],c='skyblue',marker='o')

ax[0][1].set_xlabel('useful')
ax[0][1].set_ylabel('star rating')
#ax[0][1].set_title('useful')
ax[0][1].scatter(df_tag['useful'],df_tag['stars_y'],c='skyblue',marker='o')

ax[1][0].set_xlabel('funny')
ax[1][0].set_ylabel('star rating')
#ax[1][0].set_title('funny')
ax[1][0].scatter(df_tag['funny'],df_tag['stars_y'],c='skyblue',marker='o')

ax[1][1].set_xlabel('cool')
ax[1][1].set_ylabel('star rating')
#ax[1][1].set_title('cool')
ax[1][1].scatter(df_tag['cool'],df_tag['stars_y'],c='skyblue',marker='o')

fig.tight_layout(pad=2.0)


# In[112]:


df_tags1 =  df_tag.iloc[:][['name','stars_y','useful','funny','cool']]

clist = []
for i in range(len(df_tags1)):
    if float(df_tags1.iloc[i][['stars_y']]) > 4:
        c = '4-5'
    elif float(df_tags1.iloc[i][['stars_y']]) > 3:
        c = '3-4'
    elif float(df_tags1.iloc[i][['stars_y']]) > 2:
        c = '2-3'
    else:
        c = '1-2'
    clist.append(c)
df_tags1['class'] = clist
df_tags1.head(n=2)


# In[113]:


df_tags11 = df_tags1.groupby(['class']).sum()
df_tags11= df_tags11.reset_index()
df_tags11


# In[114]:


df_tags11=df_tags11.drop(columns=['stars_y'])


# In[115]:


df_tags11


# In[116]:


color_list = {'useful':'red','funny':'orange','cool':'pink'}
df_tags11.plot(x='class',kind='barh',
                   figsize=(6,6),color=color_list) 
plt.tight_layout()  
#plt.savefig("figure/tags.png",dpi=300)
plt.show()


# In[122]:


# plot tag level figure
gdf = gpd.GeoDataFrame(df_tag, geometry=gpd.points_from_xy(df_tag.longitude_x, df_tag.latitude_x))
gdf =gdf.set_crs(epsg=4326)
gdf =gdf.to_crs(epsg=3857)
fig,ax=plt.subplots(figsize=(15, 10))

tag_plot=gdf.plot(figsize=(8, 8), linewidth=0.1,
                edgecolor='g',ax=ax,
                column='useful', colormap = 'PuOr',legend=True,scheme='NaturalBreaks',
                legend_kwds={'title': "useful count"})

# add the basemap
ctx.add_basemap(ax, zoom=12, source=ctx.providers.Stamen.TonerLite,alpha=0.8)

ax.set_title("The count of useful of Venue Review in the City of Calgary, Canada",fontsize= 22)

# removes the axis
ax.set_axis_off()


# In[123]:


# plot tag level figure
gdf = gpd.GeoDataFrame(df_tag, geometry=gpd.points_from_xy(df_tag.longitude_x, df_tag.latitude_x))
gdf =gdf.set_crs(epsg=4326)
gdf =gdf.to_crs(epsg=3857)
fig,ax=plt.subplots(figsize=(15, 10))

tag_plot=gdf.plot(figsize=(8, 8),linewidth=0.1,
                edgecolor='g',ax=ax,
                column='funny', colormap =  'PuOr',legend=True,scheme='NaturalBreaks',
                legend_kwds={'title': "funny count"})

# add the basemap
ctx.add_basemap(ax, zoom=12, source=ctx.providers.Stamen.TonerLite,alpha=0.8)

ax.set_title("The count of funny of Venue Review in the City of Calgary, Canada",fontsize= 22)

# removes the axis
ax.set_axis_off()


# In[124]:


# plot tag level figure
gdf = gpd.GeoDataFrame(df_tag, geometry=gpd.points_from_xy(df_tag.longitude_x, df_tag.latitude_x))
gdf =gdf.set_crs(epsg=4326)
gdf =gdf.to_crs(epsg=3857)
fig,ax=plt.subplots(figsize=(15, 10))

tag_plot=gdf.plot(figsize=(8, 8), linewidth=0.1,
                edgecolor='g',ax=ax,
                column='cool', colormap =  'PuOr',legend=True,
                scheme='NaturalBreaks',
                legend_kwds={'title': "cool count"})

# add the basemap
ctx.add_basemap(ax, zoom=12, source=ctx.providers.Stamen.TonerLite,alpha=0.8)

ax.set_title("The count of cool of Venue Review in the City of Calgary, Canada",fontsize= 22)

# removes the axis
ax.set_axis_off()


# ## 4 Use a pretrained neural word embedding method (eg. word2vec) for the supervised learning task and compare the results with the bag of words features, 

# In[125]:


pip install tensorflow


# In[126]:


import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import copy
import matplotlib.pyplot as plt


# In[127]:


# load data 
sentences = [one_sentence.split() for one_sentence in df_sub['cleaned_text']]


# In[128]:


print(sentences)


# In[129]:



tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[130]:


print(word_index)


# In[131]:


training_samples = 500  
validation_samples = 300 
test_samples = 200 

maxlen=100  

data = pad_sequences(sequences, maxlen=maxlen) 
labels_df_cp = copy.deepcopy(df_sub['polarity'])
labels_df_cp[labels_df_cp=="pos"] = 1
labels_df_cp[labels_df_cp=="neg"] = 0
labels = np.asarray(labels_df_cp)

indices = np.arange(data.shape[0])
np.random.shuffle(indices) 
data = data[indices]  
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val =   data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
x_test = data[training_samples + validation_samples: training_samples + validation_samples + test_samples]
y_test = labels[training_samples + validation_samples: training_samples + validation_samples + test_samples]


# In[132]:



def load_wv(sentences=sentences):
    model = Word2Vec(sentences, window=5, min_count=5, workers=4)
    model.save('my_word2vec.model')
    loaded_model = KeyedVectors.load("my_word2vec.model", mmap='r')
    word_vectors = loaded_model.wv
    word_vectors.save("my_word2vec.wordvectors")
    # Load back with memory-mapping = read-only, shared across processes.
    wv = KeyedVectors.load("my_word2vec.wordvectors", mmap='r')

    return wv
wv = load_wv(sentences)

embedding_dim = 100  
max_words = 10000  
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    try:
        embedding_vector = wv[word]
    except:
        continue
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


# In[133]:



print("'restaurant' wv result:")
print(wv['the'])


# In[135]:




model = Word2Vec(sentences, window=5, min_count=5, workers=4)
model.save('my_word2vec.model')
loaded_model = KeyedVectors.load("my_word2vec.model", mmap='r')
word_vectors = loaded_model.wv
word_vectors.save("my_word2vec.wordvectors")
# Load back with memory-mapping = read-only, shared across processes.
wv = KeyedVectors.load("my_word2vec.wordvectors", mmap='r')



embedding_dim = 100 
max_words = 10000  
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    try:
        embedding_vector = wv[word]
    except:
        continue
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    
model.wv.most_similar('restaurant')
    


# In[136]:


len(wv.key_to_index)


# In[137]:


vectors      = np.array(model.wv.vectors.tolist())
vocabulary   = np.array(model.wv.index_to_key)


# In[138]:



model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.layers[0].set_weights([embedding_matrix])  
model.layers[0].trainable = False


# In[139]:


x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)


# In[148]:


x_test.shape


# In[140]:


print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)


# In[141]:



model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')


# In[142]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[143]:



y_pred = model.predict(x_test).flatten()
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0


# In[144]:


print("Accuracy: %.2f"% accuracy_score(y_test, y_pred))
print("F-score: %.2f"% f1_score(y_test, y_pred, average='macro'))


# In[145]:


import seaborn as sns
#creates a grid to plot on
f, ax = plt.subplots(figsize=(7, 5))
#plots confusion matrix
#plots confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, 
            fmt="d", 
            linewidths=.5, 
            cmap="YlGnBu",
            xticklabels=['neg','pos'], 
            yticklabels=['neg','pos']
           )

plt.title('Confusion Matrix for Word2vec Prediction' +'\n')

plt.show()


# In[ ]:





# In[ ]:




