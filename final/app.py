import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,train_test_split
from keras.models import Sequential,load_model
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

df = pd.read_csv("finalmodel.csv")
df = df.dropna()
le = preprocessing.LabelEncoder()
columns = ["District_Name","Season","Area","Production"]
column2 = ["Crop"]
# print(df[["District_Name","Season","Crop"]])


dictionary={0: "Tur",1:"Bajra",2:"Coconut",3:"Cotton",4:"Groundnut",5:"Jowar",6:"Maize",7:"Paddy",8:"Moong",9:"Niger Seed",10:"Ragi",11:"Rice",12:"Sesamun",13:"Soyabean",14:"Sugarcane",15:"Sunflower",16:"Urad",17:"Wheat"}

series = pd.Series(dictionary)
var = le.fit_transform(df.District_Name)
df.District_Name =var
#print(df.District_Name.unique)

var = le.fit_transform(df.Season)
df.Season =var
#print(df.Season.unique)

var = le.fit_transform(df.Crop)
df.Crop =var
#print(df.Crop.unique)

var1  = MinMaxScaler()
a0 = np.array(df.Area)
a1 = np.array(df.Production)
a0 = a0.reshape(a0.shape[0],1)
a1 = a1.reshape(a1.shape[0],1)

a11 = var1.fit_transform(a0)
a21 = var1.fit_transform(a1)

# print(a11.max())
# print(a21.min())

X = df[columns].values
#print(X)
y = list(df.Crop)
#print(y)
# print(df.isna().sum())

(X_train, X_test, ytrain, ytest) = train_test_split(X,y, test_size=0.25, random_state=42,shuffle = True)

kmeans = KMeans(n_clusters=18,init='k-means++')
y_kmeans = kmeans.fit(X_train)
print(list(ytest))
print("------------------------------------------------------------------------------------------------------------------------")
y_test = list(kmeans.predict(X_test))
print("We predict ",y_test)

'''
print(Xtrain.shape,ytrain.shape)

model = Sequential()
model.add(Dense(1000, activation='', input_shape=[Xtrain.shape[1]]))
model.add(Dense(200, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='rmsprop', metrics=['mean_squared_error'])
model.fit(Xtrain, ytrain, epochs=10, batch_size=10, verbose=1,validation_split = 0.2)

xnew = np.array([[51,2,30,6040]])
ynew=model.predict_classes(xnew)
print(ynew)
'''# model.save('re.h5')
# model = load_model('re.h5')
# model.evaluate(Xtest,ytest)
#pred = model.predict(Xtest)
#print(int(pred))
# lol=int(np.argmax(pred))
# print(lol)
# print(ytest[lol][0])
# print(dictionary[ytest[lol][0]])