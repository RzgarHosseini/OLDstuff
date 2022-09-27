
from sklearn.preprocessing import label_binarize
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import random
import sys
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input, Activation,Flatten
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import models, layers
from sklearn.metrics import r2_score,mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def Prepare_Features(DIR,num):

    drug_features=[]
    drug_features_name=[]

    data=pd.read_csv(str(DIR)+"/Data/Sanger.csv")
    exp=pd.read_csv(str(DIR)+"/Data/Cell"+str(num)+".csv")
    responses=np.zeros((np.array(data).shape[0], 1))
    celllines=np.unique(data.Cell_Line)
 
    #Reading drug features from the 25 matrices
    for filename in os.listdir(str(DIR)+"/Data/Metrics/Sanger"):
            M=np.array(pd.read_csv(str(DIR)+"/Data/Metrics/Sanger/"+filename))
            drug_features.append(M)
            drug_features_name.append(filename)

    features=np.zeros((np.array(data).shape[0],356,25))
    Synergy=data['Synergy']

    d_i=0
   
    data.Cell_Line=data.Cell_Line.astype(str) 
  
    for d1,d2,cell in zip(data.Index1, data.Index2, data.Cell_Line):
            m_i=0
            for i in range(len(drug_features)):
                M=drug_features[i][:,1:]
                cell=str(cell).strip()
                features[d_i,:,m_i]=np.concatenate((M[d1-1,:],M[d2-1,:],np.array(exp[cell])),axis=0)
                m_i=m_i+1
            d_i=d_i+1

    return features



def Synergy_Scores(DIR):

    data=pd.read_csv(str(DIR)+"/Data/Sanger.csv")
    responses=np.zeros((np.array(data).shape[0], 1))
    Synergy=data['Synergy']

    d_i=0   
    data.Cell_Line=data.Cell_Line.astype(str) 
  
    for d1,d2,cell in zip(data.Index1, data.Index2, data.Cell_Line):
            responses[d_i,0]=Synergy[d_i]
            d_i=d_i+1

    return responses


def DNN(inputShape,n1=2000,n2=1000,n3=500,lr=0.0001):
	model = models.Sequential()
	model.add(layers.Dense(n1,kernel_initializer="he_normal", input_shape=[inputShape]))
	model.add(layers.Dropout(0.5))# % of features dropped)
	model.add(layers.Dense(n2, activation='relu',kernel_initializer="he_normal"))
	model.add(layers.Dropout(0.3))# % of features dropped)
	model.add(layers.Dense(n3, activation='tanh',kernel_initializer="he_normal"))
	# output layer
	model.add(layers.Dense(1,activation = "sigmoid")) 
	model.compile( optimizer=keras.optimizers.Adam(learning_rate=float(lr),beta_1=0.9, beta_2=0.999, amsgrad=False), loss = "binary_crossentropy", metrics=["accuracy"])
	return model

FeatureName=['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5']

def concatRefine(x_train,y_train,num):

    x_new=np.zeros((x_train.shape[0]*2,x_train.shape[1]))
    y_new=[]
    i=-1
    for x,y in zip(x_train,y_train):
        i=i+1

        x_new[i,:]=np.concatenate((x[128:256],x[0:128],x[256:num]),axis=0)
        y_new.append(y)
        i=i+1
        x_new[i,:]=x
        y_new.append(y)
    return x_new,y_new

def creatCombUNiq(data):
    UniqList=[]
    for d1, d2 in zip(data.Drug1,data.Drug2):
        if((d2+'//'+d1) not in UniqList):
            UniqList.append(d1+'//'+d2)
        
    UniqList=np.unique(np.array(UniqList))
    return UniqList

def concatPandas(data,out):    
    NewTrain=pd.DataFrame()
    for d1, d2 in zip(out[:,0],out[:,1]):
            TrainName=data[(((data.Drug1==d1)  &  (data.Drug2==d2 ))|((data.Drug1==d2)  & (data.Drug2==d1 )))]
            NewTrain = pd.concat([NewTrain,TrainName])
    return NewTrain

def toNUMPY(d):
    data=[]
    for i in d:
        data.append(np.array(i))
    return np.array(data)

def runapp(DIR,j,i,n1,n2,n3,batch,lr,seedd,num,TINDX):
    data=pd.read_csv(str(DIR)+"/Data/Sanger.csv")
    UniqList=creatCombUNiq(data)

    X=Prepare_Features(DIR,j)[:,:,i]
    y=Synergy_Scores(DIR)[:,0]
    kf = KFold(n_splits=5, random_state=seedd, shuffle=True)
    kf.get_n_splits(UniqList)
    prenew=np.array([])
    realnew=np.array([])
    ii=0
    outFinal=pd.DataFrame()
    for train_index, test_index in kf.split(UniqList):
        outPut=pd.DataFrame()

        ii=ii+1
        Comb_train, Comb_test = UniqList[train_index], UniqList[test_index]
        out_train= toNUMPY(np.char.split(Comb_train, sep ='//'))
        out_test =toNUMPY( np.char.split(Comb_test, sep ='//'))
        TrainName=concatPandas(data,out_train)
        TestName=concatPandas(data,out_test)
        TrainName=TrainName[TrainName.Tindex!=TINDX]
        TestName=TestName[TestName.Tindex==TINDX]
        X_test=X[TestName.index.astype(int)]
        y_test=y[TestName.index.astype(int)]
        X_train=X[TrainName.index.astype(int)]
        y_train=y[TrainName.index.astype(int)]
        X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.20,random_state=seedd)
        X_train,y_train=concatRefine(X_train,y_train,num)
        outPut['fold']=np.zeros([len(TestName),])+ii
        outPut['Index']=TestName.index.astype(int)
        
        CNN_model=DNN(num,n1,n2,n3,lr)

        cb_check = ModelCheckpoint((str(DIR)+'/DNN_CV2_Classification_Cell'+str(j)+'_'+str(FeatureName[i])+'_'+str(TINDX)), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

        y_train=np.array(y_train)
        
        CNN_model.fit(x=X_train,y=y_train,batch_size=batch,epochs = 100,shuffle=True,validation_data = (X_val,y_val),callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience = 10),cb_check] )

        CNN_model = tf.keras.models.load_model((str(DIR)+'/DNN_CV2_Classification_Cell'+str(j)+'_'+str(FeatureName[i])+'_'+str(TINDX)))
        pre_test=CNN_model.predict(X_test)
        
        prenew=np.concatenate((pre_test[:,0],prenew), axis=None)
        realnew=np.concatenate((y_test,realnew), axis=None)
        outPut['Real']=y_test
        outPut['Pre']=pre_test[:,0]
        outFinal = pd.concat([outFinal,outPut])

    outFinal.to_csv((str(DIR)+'/DNN_CV2_Classification_Cell'+str(j)+'_'+str(FeatureName[i])+'_'+str(TINDX)+'.csv'))
    
 
def main():

    DIR=""   #Working directory
    j=0      #Index for cell line representation method (1-5)
    i=0      #Index for drug representation method (1-25)
    TINDX=0  #Tissue index
    
    n1=2000 #number of neurons in the first layer
    n2=1000 #number of neurons in the second layer
    n3=500  #number of neurons in the third layer
    lr=0.0001 #learning rate
    batch=128 #batch size
    seedd=94  #seed number 
    num=356   #size of the input vector

    DIR=sys.argv[1]
    j=sys.argv[2]
    i=sys.argv[3]
    TINDX=sys.argv[4]

    runapp(str(DIR),int(j),int(i),int(n1),int(n2),int(n3),int(batch),float(lr),int(seedd),int(num),int(TINDX))


main()
