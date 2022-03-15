import os.path
from os import makedirs,remove
from re import compile,findall,split
from .config import LineConfig
class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def writeFile(dir,file,content,op = 'w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+file,op) as f:
            f.writelines(content)

    @staticmethod
    def deleteFile(filePath):
        if os.path.exists(filePath):
            remove(filePath)

    @staticmethod
    def loadDataSet(conf, file, bTest=False,binarized = False, threshold = 3.0):
        trainingData = []
        testData = []
        # print(conf['ratings.setup'])
        # ratingConfig = LineConfig(conf['ratings.setup'])
        if not bTest:
            print('loading training data...')
        else:
            print('loading test data...')
        with open(file) as f:
            ratings = f.readlines()
        # ignore the headline
        # if ratingConfig.contains('-header'):
        #     ratings = ratings[1:]
        # order of the columns
        # order = ratingConfig['-columns'].strip().split()
        order = [0,1,2]
        delim = ' |,|\t'
        # if ratingConfig.contains('-delim'):
        #     delim=ratingConfig['-delim']
        for lineNo, line in enumerate(ratings):
            drugs = split(delim,line.strip())
            if not bTest and len(order) < 2:
                print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            try:
                ncRNAId = drugs[int(order[0])]
                drugId = drugs[int(order[1])]
                if len(order)<3:
                    rating = 1 #default value
                else:
                    rating = drugs[int(order[2])]
                if binarized:
                    if float(drugs[int(order[2])])<threshold:
                        continue
                    else:
                        rating = 1
            except ValueError:
                print('Error! Have you added the option -header to the rating.setup?')
                exit(-1)
            if bTest:
                testData.append([ncRNAId, drugId, float(rating)])
            else:
                trainingData.append([ncRNAId, drugId, float(rating)])
        if bTest:
            return testData
        else:
            return trainingData

    @staticmethod
    def loadncRNAList(filepath):
        ncRNAList = []
        print('loading ncRNA List...')
        with open(filepath) as f:
            for line in f:
                ncRNAList.append(line.strip().split()[0])
        return ncRNAList



