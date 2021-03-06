import numpy as np
from .config import Config,LineConfig
import random
from collections import defaultdict
class Relation:
    'data access control'
    def __init__(self,config,trainingSet, testSet):
        self.config = config
        # self.evalSettings = LineConfig(self.config['evaluation.setup'])
        self.ncRNA = {}
        self.drug = {}
        self.id2ncRNA = {}
        self.id2drug = {}
        self.ncRNAMeans = {}
        self.drugMeans = {}
        self.globalMean = 0
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)
        self.testSet_i = defaultdict(dict)
        self.rScale = []
        self.trainingData = trainingSet[:]
        self.testData = testSet[:]
        self.__generateSet()
        self.__computedrugMean()
        self.__computencRNAMean()
        self.__globalAverage()

    def __generateSet(self):
        scale = set()
        # if self.evalSettings.contains('-val'):
        #     random.shuffle(self.trainingData)
        #     separation = int(self.elemCount()*float(self.evalSettings['-val']))
        #     self.testData = self.trainingData[:separation]
        #     self.trainingData = self.trainingData[separation:]
        for i,entry in enumerate(self.trainingData):
            ncRNAName,drugName,rating = entry
            if ncRNAName not in self.ncRNA:
                self.ncRNA[ncRNAName] = len(self.ncRNA)
                self.id2ncRNA[self.ncRNA[ncRNAName]] = ncRNAName
            if drugName not in self.drug:
                self.drug[drugName] = len(self.drug)
                self.id2drug[self.drug[drugName]] = drugName
            self.trainSet_u[ncRNAName][drugName] = rating
            self.trainSet_i[drugName][ncRNAName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()
        for entry in self.testData:
            # if self.evalSettings.contains('-predict'):
            #     self.testSet_u[entry]={}
            # else:
            ncRNAName, drugName, rating = entry
            self.testSet_u[ncRNAName][drugName] = rating
            self.testSet_i[drugName][ncRNAName] = rating

    def __globalAverage(self):
        total = sum(self.ncRNAMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.ncRNAMeans)

    def __computencRNAMean(self):
        for u in self.ncRNA:
            self.ncRNAMeans[u] = sum(self.trainSet_u[u].values())/len(self.trainSet_u[u])

    def __computedrugMean(self):
        for c in self.drug:
            self.drugMeans[c] = sum(self.trainSet_i[c].values())/len(self.trainSet_i[c])

    def getncRNAId(self,u):
        if u in self.ncRNA:
            return self.ncRNA[u]

    def getdrugId(self,i):
        if i in self.drug:
            return self.drug[i]

    def trainingSize(self):
        return (len(self.ncRNA),len(self.drug),len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

    def contains(self,u,i):
        'whether ncRNA u rated drug i'
        if u in self.ncRNA and i in self.trainSet_u[u]:
            return True
        else:
            return False

    def containsncRNA(self,u):
        'whether ncRNA is in training set'
        if u in self.ncRNA:
            return True
        else:
            return False

    def containsdrug(self,i):
        'whether drug is in training set'
        if i in self.drug:
            return True
        else:
            return False

    def ncRNARated(self,u):
        return list(self.trainSet_u[u].keys()),list(self.trainSet_u[u].values())

    def drugRated(self,i):
        return list(self.trainSet_i[i].keys()),list(self.trainSet_i[i].values())

    def row(self,u):
        k,v = self.ncRNARated(u)
        vec = np.zeros(len(self.drug))
        for pair in zip(k,v):
            iid = self.drug[pair[0]]
            vec[iid]=pair[1]
        return vec

    def col(self,i):
        k,v = self.drugRated(i)
        vec = np.zeros(len(self.ncRNA))
        for pair in zip(k,v):
            uid = self.ncRNA[pair[0]]
            vec[uid]=pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.ncRNA),len(self.drug)))
        for u in self.ncRNA:
            k, v = self.ncRNARated(u)
            vec = np.zeros(len(self.drug))
            # print vec
            for pair in zip(k, v):
                iid = self.drug[pair[0]]
                vec[iid] = pair[1]
            m[self.ncRNA[u]]=vec
        return m

    def sRow(self,u):
        return self.trainSet_u[u]

    def sCol(self,c):
        return self.trainSet_i[c]

    def rating(self,u,c):
        if self.contains(u,c):
            return self.trainSet_u[u][c]
        return -1

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return len(self.trainingData)
