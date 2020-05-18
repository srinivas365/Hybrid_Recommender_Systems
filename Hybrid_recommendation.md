
# Building a Hybrid Recommendation

This notebook contains following sections

1. Importing necessary Libraries & dataset
2. Building a dataset Module
3. Building performance Module
4. Building Evaluator Module
    1. Evaluated Algorithm submodule
    2. Evaluated Data submodule
    
5. Building Hybrid Module



```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
!pip install surprise
```

    Requirement already satisfied: surprise in /usr/local/lib/python3.6/dist-packages (0.1)
    Requirement already satisfied: scikit-surprise in /usr/local/lib/python3.6/dist-packages (from surprise) (1.1.0)
    Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.6/dist-packages (from scikit-surprise->surprise) (1.4.1)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-surprise->surprise) (0.14.1)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from scikit-surprise->surprise) (1.12.0)
    Requirement already satisfied: numpy>=1.11.2 in /usr/local/lib/python3.6/dist-packages (from scikit-surprise->surprise) (1.18.2)
    


```python
folderpath='drive/My Drive/datasets/'
```

## Importing Necessary Libraries


```python
import os
import csv
import sys
import re

import numpy as np
import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import dump

from collections import defaultdict
```

![alt text](https://drive.google.com/uc?id=1y-_naXebtRfC6L-5og9DQ3JjrPwQhryn)

## DataLoader Module

This module takes the raw dataset and provides the processed the dataset along with other details 

It has following functions

1. loadDataset
2. getUserRating
3. getPopularityRanking
4. getArtistName
5. getArtistID


```python
#user_id	artist_mbid	artist_name	plays	norm_plays	rating

class DataLoader:
    path='drive/My Drive/datasets/user-songs-rating-3000.csv'
    artistID_to_name={}
    name_to_artistID={}
    #user_id	artist_mbid	norm_plays	rating
    
    def loadDataset(self):

        ratingsDataset = 0
        self.artistID_to_name = {}
        self.name_to_artistID = {}

        reader = Reader(rating_scale=(0, 5))
        df_matrix=pd.read_csv(self.path)
        #df_matrix=df_matrix.iloc[:200000,:]
        ratingsDataset= Dataset.load_from_df(df_matrix[['user_id', 'artist_mbid', 'rating']], reader)
    
        with open(self.path, newline='', encoding='ISO-8859-1') as csvfile:
                artistReader = csv.reader(csvfile)
                next(artistReader)  #Skip header line
                for row in artistReader:
                    artistID = row[1]
                    artistName = row[2]
                    self.artistID_to_name[artistID] = artistName
                    self.name_to_artistID[artistName] = artistID

        return ratingsDataset
    
    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        with open(self.path, newline='', encoding='ISO-8859-1') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = row[0]
                if (user == userID):
                    artistID = row[1]
                    rating = float(row[5])
                    userRatings.append((artistID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings
    
    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.path, newline='', encoding='ISO-8859-1') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                artistID = row[1]
                ratings[artistID] += 1
        rank = 1
        for artistID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[artistID] = rank
            rank += 1
        return rankings
    
    def getArtistName(self, artistID):
        if artistID in self.artistID_to_name:
            return self.artistID_to_name[artistID]
        else:
            return ""
        
    def getArtistID(self, artistName):
        if artistName in self.name_to_artistID:
            return self.name_to_artistID[artistName]
        else:
            return 0
    
```

# performance Class Module

This module generated the metrics by taking the predictions of the models.
It outputs two metrics
1. Mean Absolute Error
2. Root mean square Error


```python
from surprise import accuracy
class PerformanceMetrics:
	
	def MAE(predictions):
		return accuracy.mae(predictions)
		
	def RMSE(predictions):
		return accuracy.rmse(predictions)
```

# ModelBuilder Module

This module is to build the algorithms/models to train the dataset
It has following models
1. getName - returns the name of model
2. getModel - returns the model
3. saveModel - save the model
4. Evaluate - train the model and returns the metrics


```python
class ModelBuilder:
    def __init__(self, model, name):
        self.model = model
        self.name = name
    def GetName(self):
        return self.name
    
    def GetModel(self):
        return self.model

    def SaveModel(self,predictions):
        
        dump.dump(folderpath+self.name,predictions,self.model)
        print('Model saved at '+folderpath+self.name)
        
    
    def Evaluate(self, evaluationData,save=False):
        metrics = {}
        # Compute accuracy
    
        print("Evaluating accuracy...")
        predictions = self.model.fit(evaluationData.GetTrainSet()).test(evaluationData.GetTestSet())
        metrics["RMSE"] = PerformanceMetrics.RMSE(predictions)
        metrics["MAE"] = PerformanceMetrics.MAE(predictions)
        
        
        print("Analysis complete.")

        if(save):
            print('saving the model.....')
            self.SaveModel(predictions)
            
    
        return metrics
    
    
    
```

# ModelFactory Module

This module is used to load a set of models into the returns the metrics/performace of each algorithm

It has following functions
1. addmodel
2. Evaluate
3. flushModels


```python
class ModelFactory:
    
    models = []
    
    def __init__(self, dataset):        
        ed = DataGenerator(dataset)
        self.dataset = ed
        self.models=[]
        
    def AddModel(self, model, name):
        alg = ModelBuilder(model, name)
        self.models.append(alg)
        
    def Evaluate(self,save=False):
        results = {}
        for model in self.models:
            print("Evaluating ", model.GetName(), "...")
            results[model.GetName()] = model.Evaluate(self.dataset,save)

        # Print results
        print("\n")
        print(results)
    def flushModels(self):
        self.models=[]
```

# DataGenerator Module

This model takes the dataset splits it into training dataset and testing dataset and return them 


```python
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut

class DataGenerator:
    
    def __init__(self, data):
        #Build a 75/25 train/test split for measuring accuracy
        self.trainSet, self.testSet = train_test_split(data, test_size=.25, random_state=1)
            
    def GetTrainSet(self):
        return self.trainSet
    
    def GetTestSet(self):
        return self.testSet
    

```

# Hybrid Algorithm Module

This module takes multiple models along with their preference weights and returns the results


```python
from surprise import AlgoBase

class HybridModel(AlgoBase):

    def __init__(self, models, weights, sim_options={}):
        AlgoBase.__init__(self)
        self.models = models
        self.weights = weights

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        for model in self.models:
            model.fit(trainset)
                
        return self

    def estimate(self, user_id, item_id):
        
        scores_sum = 0
        weights_sum = 0
        
        for i in range(len(self.models)):
            scores_sum += self.models[i].estimate(user_id, item_id) * self.weights[i] # 3*1/4+4*3/4 laga ra
            weights_sum += self.weights[i] # always becomes one
            
        return scores_sum / weights_sum

```


```python
def LoadData():
    ml = DataLoader()
    print("Loading songs ratings...")
    data = ml.loadDataset()
    return (ml, data)
```


```python
# Load up common data set for the recommender algorithms
(ml, evaluationData) = LoadData()
```

    Loading songs ratings...
    


```python
from surprise import BaselineOnly
#Construct an Evaluator to, you know, evaluate them
modelfactory = ModelFactory(evaluationData)

# BaselineOnly
baseline= BaselineOnly()
modelfactory.AddModel(baseline, "baseline")
modelfactory.Evaluate(True)
```

    Evaluating  baseline ...
    Evaluating accuracy...
    Estimating biases using als...
    RMSE: 0.9934
    MAE:  0.6817
    Analysis complete.
    saving the model.....
    Model saved at drive/My Drive/datasets/baseline
    
    
    {'baseline': {'RMSE': 0.9934226190571713, 'MAE': 0.6816700266309835}}
    


```python
from surprise import SVD
# BaselineOnly
svd= SVD()
modelfactory.AddModel(svd, "svd")
modelfactory.Evaluate()
```

    Evaluating  baseline ...
    Evaluating accuracy...
    Estimating biases using als...
    RMSE: 0.9934
    MAE:  0.6817
    Analysis complete.
    Evaluating  svd ...
    Evaluating accuracy...
    RMSE: 1.0063
    MAE:  0.6831
    Analysis complete.
    
    
    {'baseline': {'RMSE': 0.9934226190571713, 'MAE': 0.6816700266309835}, 'svd': {'RMSE': 1.0063057804737225, 'MAE': 0.6830706904210475}}
    


```python
#Combine them
Hybrid = HybridModel([svd, baseline], [0.5, 0.5])
# Fight!
modelfactory.AddModel(Hybrid, "Hybrid")
modelfactory.Evaluate(True)

```

    Evaluating  baseline ...
    Evaluating accuracy...
    Estimating biases using als...
    RMSE: 0.9934
    MAE:  0.6817
    Analysis complete.
    saving the model.....
    Model saved at drive/My Drive/datasets/baseline
    Evaluating  svd ...
    Evaluating accuracy...
    RMSE: 1.0058
    MAE:  0.6821
    Analysis complete.
    saving the model.....
    Model saved at drive/My Drive/datasets/svd
    Evaluating  Hybrid ...
    Evaluating accuracy...
    Estimating biases using als...
    RMSE: 0.9971
    MAE:  0.6795
    Analysis complete.
    saving the model.....
    Model saved at drive/My Drive/datasets/Hybrid
    
    
    {'baseline': {'RMSE': 0.9934226190571713, 'MAE': 0.6816700266309835}, 'svd': {'RMSE': 1.005848123790534, 'MAE': 0.6821244100215875}, 'Hybrid': {'RMSE': 0.9970686978736479, 'MAE': 0.6794833671716486}}
    

# Training dataset using Deep Learing Technique

RBMs have two layers, input layer which is also known as visible layer and the hidden layer. The neurons in each layer communicate with neurons in the other layer but not with neurons in the same layer. there is no intralayer communication among the neurons.





```python
import sys
sys.path.append('/content/drive/My Drive/datasets/')
import RBM 
import RBMModel

import importlib
importlib.reload(RBM)
importlib.reload(RBMModel)

# Construct an Evaluator to, you know, evaluate them
deep_factory= ModelFactory(evaluationData)

#Simple RBM
SimpleRBM = RBMModel.RBMAlgorithm(epochs=10)
deep_factory.AddModel(SimpleRBM,'rbm')

svd= SVD()
deep_factory.AddModel(svd, "svd")



#Combine them
Hybrid = HybridModel([svd, SimpleRBM], [0.5, 0.5])
# Fight!
deep_factory.AddModel(Hybrid, "Hybrid")

deep_factory.Evaluate()
```

    Evaluating  rbm ...
    Evaluating accuracy...
    Trained epoch  0
    Trained epoch  1
    Trained epoch  2
    Trained epoch  3
    Trained epoch  4
    Trained epoch  5
    Trained epoch  6
    Trained epoch  7
    Trained epoch  8
    Trained epoch  9
    Processing user  0
    Processing user  50
    Processing user  100
    Processing user  150
    Processing user  200
    Processing user  250
    Processing user  300
    Processing user  350
    Processing user  400
    Processing user  450
    Processing user  500
    Processing user  550
    Processing user  600
    Processing user  650
    Processing user  700
    Processing user  750
    Processing user  800
    Processing user  850
    Processing user  900
    Processing user  950
    Processing user  1000
    Processing user  1050
    Processing user  1100
    Processing user  1150
    Processing user  1200
    Processing user  1250
    Processing user  1300
    Processing user  1350
    Processing user  1400
    Processing user  1450
    Processing user  1500
    Processing user  1550
    Processing user  1600
    Processing user  1650
    Processing user  1700
    Processing user  1750
    Processing user  1800
    Processing user  1850
    Processing user  1900
    Processing user  1950
    Processing user  2000
    Processing user  2050
    Processing user  2100
    Processing user  2150
    Processing user  2200
    Processing user  2250
    Processing user  2300
    Processing user  2350
    Processing user  2400
    Processing user  2450
    Processing user  2500
    Processing user  2550
    Processing user  2600
    Processing user  2650
    Processing user  2700
    Processing user  2750
    Processing user  2800
    Processing user  2850
    Processing user  2900
    Processing user  2950
    RMSE: 1.5733
    MAE:  1.4437
    Analysis complete.
    Evaluating  svd ...
    Evaluating accuracy...
    RMSE: 1.0070
    MAE:  0.6827
    Analysis complete.
    Evaluating  Hybrid ...
    Evaluating accuracy...
    Trained epoch  0
    Trained epoch  1
    Trained epoch  2
    Trained epoch  3
    Trained epoch  4
    Trained epoch  5
    Trained epoch  6
    Trained epoch  7
    Trained epoch  8
    Trained epoch  9
    Processing user  0
    Processing user  50
    Processing user  100
    Processing user  150
    Processing user  200
    Processing user  250
    Processing user  300
    Processing user  350
    Processing user  400
    Processing user  450
    Processing user  500
    Processing user  550
    Processing user  600
    Processing user  650
    Processing user  700
    Processing user  750
    Processing user  800
    Processing user  850
    Processing user  900
    Processing user  950
    Processing user  1000
    Processing user  1050
    Processing user  1100
    Processing user  1150
    Processing user  1200
    Processing user  1250
    Processing user  1300
    Processing user  1350
    Processing user  1400
    Processing user  1450
    Processing user  1500
    Processing user  1550
    Processing user  1600
    Processing user  1650
    Processing user  1700
    Processing user  1750
    Processing user  1800
    Processing user  1850
    Processing user  1900
    Processing user  1950
    Processing user  2000
    Processing user  2050
    Processing user  2100
    Processing user  2150
    Processing user  2200
    Processing user  2250
    Processing user  2300
    Processing user  2350
    Processing user  2400
    Processing user  2450
    Processing user  2500
    Processing user  2550
    Processing user  2600
    Processing user  2650
    Processing user  2700
    Processing user  2750
    Processing user  2800
    Processing user  2850
    Processing user  2900
    Processing user  2950
    RMSE: 1.1650
    MAE:  1.0040
    Analysis complete.
    
    
    {'rbm': {'RMSE': 1.5732653327536739, 'MAE': 1.4437298577219584}, 'svd': {'RMSE': 1.0069576973032366, 'MAE': 0.6826501315236461}, 'Hybrid': {'RMSE': 1.1649843535460234, 'MAE': 1.0040280997374962}}
    


```python

```
