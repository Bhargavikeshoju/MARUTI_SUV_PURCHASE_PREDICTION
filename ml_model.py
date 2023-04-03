#step 1 reading the datsset
import pandas as pd
df=pd.read_csv("SUV_Purchase.csv")

#step 2 feature Engineering - drop unecessay or unimportant features - simplifying the dataset
df=df.drop(['User ID','Gender'],axis =1)#axis=1 i.e columns  ....axis =0 i.e rows

#step 3- loading the data
#setting the data into input and output values
X=df.iloc[:,:-1].values #iloc==>index location 2D array
Y=df.iloc[:,-1:].values #2D array

#step 4 - Split dataset into training in test
#Training and Testing the dataset
#more data-Trainig; Less data-Testing datai.e Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Normalizing the data-Standard Scalar
from sklearn.preprocessing  import StandardScaler
sst=StandardScaler()
X_train=sst.fit_transform(X_train)
X_test=sst.transform(X_test)

#Build model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

#Training
rf.fit(X_train,Y_train)

#Testing
ypred=rf.predict(X_test)
print(ypred)

print("Training Accuracy",rf.score(X_train,Y_train))
print("Testing accuracy",rf.score(X_test,Y_test))
print("Overall Accuracy",rf.score(sst.transform(X),Y))

#Pickling - DePickling
import pickle
pickle.dump(rf,open('model.pkl','wb')) #we are serializing our model by creating model.pkl file where we are dumping rf - mode (trained)
print("Model is dumped")
