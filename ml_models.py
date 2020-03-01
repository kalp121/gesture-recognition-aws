# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:38:46 2019

@author: kalp
"""

import os
import numpy as np
import pandas as pd
import pickle
# Reading Book Data-----------------------------------------------------------------------------------
book_data=[]
directory = os.path.join("CSV","book")
for root,dirs,files in os.walk(directory):
	for file in files:
		if file.endswith(".csv"):
			df = pd.read_csv(directory+'/'+file)
			length=len(df)
			if length>150:
				x=df.iloc[0:150,1:].values
			elif length<150:
				x=df.iloc[:,1:].values
				y=df.iloc[-1:,1:].values
				rp=150-length
				y=np.tile(y,(rp,1))
				x=np.concatenate((x, y), axis=0)
				
			else:
				x=df.iloc[:,1:].values
			sd_x=np.mean(x,axis=0)
			#sd_x=np.std(x, axis=0)
			#sd_x=skew(x,axis=0)
			#sd_x=kurtosis(x,axis=0)
			
			sd_x= np.reshape(sd_x, (1, 52))
			label=np.reshape([0],(1,1))
			sd_x=np.append(sd_x,label,axis=1)
			book_data.append(sd_x)
			
# Reading Car Data-----------------------------------------------------------------------------------
car_data=[]
directory = os.path.join("CSV","car")
for root,dirs,files in os.walk(directory):
	for file in files:
		if file.endswith(".csv"):
			df = pd.read_csv(directory+'/'+file)
			length=len(df)
			if length>150:
				x=df.iloc[0:150,1:].values
			elif length<150:
				x=df.iloc[:,1:].values
				y=df.iloc[-1:,1:].values
				rp=150-length
				y=np.tile(y,(rp,1))
				x=np.concatenate((x, y), axis=0)
				
			else:
				x=df.iloc[:,1:].values
			sd_x=np.mean(x,axis=0)
			#sd_x=np.std(x, axis=0)
			#sd_x=skew(x,axis=0)
			#sd_x=kurtosis(x,axis=0)	
			sd_x= np.reshape(sd_x, (1, 52))
			label=np.reshape([1],(1,1))
			sd_x=np.append(sd_x,label,axis=1)
			car_data.append(sd_x)
# Reading Gift Data-----------------------------------------------------------------------------------
gift_data=[]
directory = os.path.join("CSV","gift")
for root,dirs,files in os.walk(directory):
	for file in files:
		if file.endswith(".csv"):
			df = pd.read_csv(directory+'/'+file)
			length=len(df)
			if length>150:
				x=df.iloc[0:150,1:].values
			elif length<150:
				x=df.iloc[:,1:].values
				y=df.iloc[-1:,1:].values
				rp=150-length
				y=np.tile(y,(rp,1))
				x=np.concatenate((x, y), axis=0)
				
			else:
				x=df.iloc[:,1:].values
			sd_x=np.mean(x,axis=0)
			#sd_x=np.std(x, axis=0)
			#sd_x=skew(x,axis=0)
			#sd_x=kurtosis(x,axis=0)	
			sd_x= np.reshape(sd_x, (1, 52))
			label=np.reshape([2],(1,1))
			sd_x=np.append(sd_x,label,axis=1)
			gift_data.append(sd_x)

# Reading Movie Data-----------------------------------------------------------------------------------
			
movie_data=[]
directory = os.path.join("CSV","movie")
for root,dirs,files in os.walk(directory):
	for file in files:
		if file.endswith(".csv"):
			df = pd.read_csv(directory+'/'+file)
			length=len(df)
			if length>150:
				x=df.iloc[0:150,1:].values
			elif length<150:
				x=df.iloc[:,1:].values
				y=df.iloc[-1:,1:].values
				rp=150-length
				y=np.tile(y,(rp,1))
				x=np.concatenate((x, y), axis=0)
				
			else:
				x=df.iloc[:,1:].values
			sd_x=np.mean(x,axis=0)
			#sd_x=np.std(x, axis=0)
			#sd_x=skew(x,axis=0)
			#sd_x=kurtosis(x,axis=0)	
			sd_x= np.reshape(sd_x, (1, 52))
			label=np.reshape([3],(1,1))
			sd_x=np.append(sd_x,label,axis=1)
			movie_data.append(sd_x)
			
			
# Reading Sell Data-----------------------------------------------------------------------------------

sell_data=[]
directory = os.path.join("CSV","sell")
for root,dirs,files in os.walk(directory):
	for file in files:
		if file.endswith(".csv"):
			df = pd.read_csv(directory+'/'+file)
			length=len(df)
			if length>150:
				x=df.iloc[0:150,1:].values
			elif length<150:
				x=df.iloc[:,1:].values
				y=df.iloc[-1:,1:].values
				rp=150-length
				y=np.tile(y,(rp,1))
				x=np.concatenate((x, y), axis=0)
				
			else:
				x=df.iloc[:,1:].values
			sd_x=np.mean(x,axis=0)
			#sd_x=np.std(x, axis=0)
			#sd_x=skew(x,axis=0)
			#sd_x=kurtosis(x,axis=0)	
			sd_x= np.reshape(sd_x, (1, 52))
			label=np.reshape([4],(1,1))
			sd_x=np.append(sd_x,label,axis=1)
			sell_data.append(sd_x)
	
# Reading Total Data-----------------------------------------------------------------------------------
		
total_data=[]
directory = os.path.join("CSV","total")
for root,dirs,files in os.walk(directory):
	for file in files:
		if file.endswith(".csv"):
			df = pd.read_csv(directory+'/'+file)
			length=len(df)
			if length>150:
				x=df.iloc[0:150,1:].values
			elif length<150:
				x=df.iloc[:,1:].values
				y=df.iloc[-1:,1:].values
				rp=150-length
				y=np.tile(y,(rp,1))
				x=np.concatenate((x, y), axis=0)
				
			else:
				x=df.iloc[:,1:].values
			sd_x=np.mean(x,axis=0)
			#sd_x=np.std(x, axis=0)
			#sd_x=skew(x,axis=0)
			#sd_x=kurtosis(x,axis=0)	
			sd_x= np.reshape(sd_x, (1, 52))
			label=np.reshape([5],(1,1))
			sd_x=np.append(sd_x,label,axis=1)
			total_data.append(sd_x)
			
arr1 = np.vstack(book_data)
arr2=np.vstack(car_data)
arr3=np.vstack(gift_data)
arr4=np.vstack(movie_data)
arr5=np.vstack(sell_data)
arr6=np.vstack(total_data)

train_data=np.concatenate((arr1,arr2,arr3,arr4,arr5,arr6),axis=0)
np.random.shuffle(train_data)
X=train_data[:,0:52]
y=train_data[:,52]


 #Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.00, random_state = 0)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)



#model-1 Random Forest---------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 30)   
	
classifier1.fit(X_train, y_train)
#y_pred = classifier1.predict(X_train) 
   
#cm = metrics.confusion_matrix(y_test, y_pred)
#accuracy=metrics.accuracy_score(y_train,y_pred)
filename='random_forest.sav'
pickle.dump(classifier1, open(filename, 'wb'))


# Model 2 --------------------------------------------------------------------------------------------

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
#y_pred = classifier.predict(X_test)

#cm = metrics.confusion_matrix(y_test, y_pred)
#accuracy=metrics.accuracy_score(y_test,y_pred)
filename='support_vector.sav'
pickle.dump(classifier1, open(filename, 'wb'))

# Model 3 --------------------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(random_state = 0,multi_class='auto',solver='liblinear')
classifier2.fit(X_train, y_train)

# Predicting the Test set results
#y_pred = classifier.predict(X_test)

#cm = metrics.confusion_matrix(y_test, y_pred)
#accuracy=metrics.accuracy_score(y_test,y_pred)

filename='logistic_regression.sav'
pickle.dump(classifier2, open(filename, 'wb'))

# Model 4 --------------------------------------------------------------------------------------------
	
from sklearn.neural_network import MLPClassifier
classifier3 = MLPClassifier(activation='relu', alpha=0.1,hidden_layer_sizes=(128, 1024,128),solver='lbfgs',max_iter=500)
classifier3.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)

#cm = metrics.confusion_matrix(y_test, y_pred)
#accuracy=metrics.accuracy_score(y_test,y_pred)

filename = 'MLP_model.sav'
pickle.dump(classifier3, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk


