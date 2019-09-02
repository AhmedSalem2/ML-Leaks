'''
Created on 5 Dec 2018

@author: Wentao Liu, Ahmed Salem
'''

import sys
sys.dont_write_bytecode = True

import numpy as np

import pickle
from sklearn.model_selection import train_test_split
import random
import lasagne
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import argparse
import deeplearning as dp
import classifier

parser = argparse.ArgumentParser()
parser.add_argument('--adv',  default='1', help='Which adversary 1, 2, or 3')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (CIFAR10 or News)')
parser.add_argument('--classifierType', default='cnn', help='Which classifier cnn or nn')
parser.add_argument('--dataset2', default='News', help='Which second dataset for adversary 2 (CIFAR10 or News)')
parser.add_argument('--classifierType2', default='nn', help='Which classifier cnn or nn')
parser.add_argument('--dataFolderPath', default='./data/', help='Path to store data')
parser.add_argument('--pathToLoadData', default='./data/cifar-10-batches-py-official', help='Path to load dataset from')
parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs to train shadow/target models')
parser.add_argument('--preprocessData', action='store_true', help='Preprocess the data, if false then load preprocessed data')
parser.add_argument('--trainTargetModel', action='store_true', help='Train a target model, if false then load an already trained model')
parser.add_argument('--trainShadowModel', action='store_true', help='Train a shadow model, if false then load an already trained model')

opt = parser.parse_args()
#Picking the top X probabilities 
def clipDataTopX(dataToClip, top=3):
	res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
	return np.array(res)

def readCIFAR10(data_path):
	for i in range(5):
		f = open(data_path + '/data_batch_' + str(i + 1), 'rb')
		train_data_dict = pickle.load(f)
		f.close()
		if i == 0:
			X = train_data_dict["data"]
			y = train_data_dict["labels"]
			continue
		X = np.concatenate((X , train_data_dict["data"]),   axis=0)
		y = np.concatenate((y , train_data_dict["labels"]), axis=0)
	f = open(data_path + '/test_batch', 'rb')
	test_data_dict = pickle.load(f)
	f.close()
	XTest = np.array(test_data_dict["data"])
	yTest = np.array(test_data_dict["labels"])
	return X, y, XTest, yTest

def trainTarget(modelType, X, y,
				X_test=[], y_test =[],
				splitData=True,
				test_size=0.5, 
				inepochs=50, batch_size=300,
				learning_rate=0.001):

	
	if(splitData):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
	else:
		X_train = X
		y_train = y

	dataset = (X_train.astype(np.float32),
			   y_train.astype(np.int32),
			   X_test.astype(np.float32),
			   y_test.astype(np.int32))

	attack_x, attack_y, theModel = dp.train_target_model(dataset=dataset, epochs=inepochs, batch_size=batch_size,learning_rate=learning_rate,
				   n_hidden=128,l2_ratio = 1e-07,model=modelType)

	return attack_x, attack_y, theModel

def load_data(data_name):
	with np.load( data_name) as f:
		train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
	return train_x, train_y


def trainAttackModel(X_train, y_train, X_test, y_test):
	dataset = (X_train.astype(np.float32),
			   y_train.astype(np.int32),
			   X_test.astype(np.float32),
			   y_test.astype(np.int32))

	output = classifier.train_model(dataset=dataset,
									epochs=50,
									batch_size=10,
									learning_rate=0.01,
									n_hidden=64,
									l2_ratio = 1e-6,
									model='softmax')

	return output

# def preprocessesCIFAR(X):
# 	#normalizing the CIFAR data
# 	X = np.dstack((X[:, :1], X[:, 1:2], X[:, 2:]))
# 	X = X.reshape((X.shape[0], 32, 32, 3)).transpose(0,3,1,2)
# 	offset = np.mean(X, 0)
# 	scale = np.std(X, 0).clip(min=1)
# 	X = (X - offset) / scale
# 	return X.astype(np.float32)

def preprocessingCIFAR(toTrainData, toTestData):
	def reshape_for_save(raw_data):
		raw_data = np.dstack((raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
		raw_data = raw_data.reshape((raw_data.shape[0], 32, 32, 3)).transpose(0,3,1,2)
		return raw_data.astype(np.float32)

	offset = np.mean(reshape_for_save(toTrainData), 0)
	scale  = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

	def rescale(raw_data):
		return (reshape_for_save(raw_data) - offset) / scale

	return rescale(toTrainData), rescale(toTestData)

def preprocessingNews(toTrainData, toTestData):
	def normalizeData(X):
		offset = np.mean(X, 0)
		scale = np.std(X, 0).clip(min=1)
		X = (X - offset) / scale
		X = X.astype(np.float32)	
		return X 
	return normalizeData(toTrainData),normalizeData(toTestData)


def shuffleAndSplitData(dataX, dataY,cluster):
	c = zip(dataX, dataY)
	random.shuffle(c)
	dataX, dataY = zip(*c)
	toTrainData  = np.array(dataX[:cluster])
	toTrainLabel = np.array(dataY[:cluster])
	
	shadowData  = np.array(dataX[cluster:cluster*2])
	shadowLabel = np.array(dataY[cluster:cluster*2])
	
	toTestData  = np.array(dataX[cluster*2:cluster*3])
	toTestLabel = np.array(dataY[cluster*2:cluster*3])
	
	shadowTestData  = np.array(dataX[cluster*3:cluster*4])
	shadowTestLabel = np.array(dataY[cluster*3:cluster*4])

	return toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel


def initializeData(dataset,orginialDatasetPath,dataFolderPath = './data/'):
	if(dataset == 'CIFAR10'):
		print("Loading data")
		dataX, dataY, _, _ = readCIFAR10(orginialDatasetPath)
		print("Preprocessing data")
		cluster = 10520
		dataPath = dataFolderPath+dataset+'/Preprocessed'
		toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel = shuffleAndSplitData(dataX, dataY,cluster)
		toTrainDataSave, toTestDataSave    = preprocessingCIFAR(toTrainData, toTestData)
		shadowDataSave, shadowTestDataSave = preprocessingCIFAR(shadowData, shadowTestData)
	elif(dataset == 'News'):
		newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes')  )
		newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes') )
		X = np.concatenate((newsgroups_train.data , newsgroups_test.data), axis=0)
		y = np.concatenate((newsgroups_train.target , newsgroups_test.target), axis=0)
		vectorizer = TfidfVectorizer()
		X = vectorizer.fit_transform(X)
		X = X.toarray()
		print("Preprocessing data")
		print(X.shape)
		cluster = 4500
		dataPath = dataFolderPath+dataset+'/Preprocessed'
		toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel = shuffleAndSplitData(X, y,cluster)
		toTrainDataSave, toTestDataSave    = preprocessingNews(toTrainData, toTestData)
		shadowDataSave, shadowTestDataSave = preprocessingNews(shadowData, shadowTestData)

			
	try:
		os.makedirs(dataPath)
	except OSError:
		pass
	
	np.savez(dataPath + '/targetTrain.npz', toTrainDataSave, toTrainLabel)
	np.savez(dataPath + '/targetTest.npz',  toTestDataSave, toTestLabel)
	np.savez(dataPath + '/shadowTrain.npz', shadowDataSave, shadowLabel)
	np.savez(dataPath + '/shadowTest.npz',  shadowTestDataSave, shadowTestLabel)
	
	print("Preprocessing finished\n\n")



def initializeTargetModel(dataset,num_epoch,dataFolderPath= './data/',modelFolderPath = './model/',classifierType = 'cnn'):
	dataPath = dataFolderPath+dataset+'/Preprocessed'
	attackerModelDataPath = dataFolderPath+dataset+'/attackerModelData'
	modelPath = modelFolderPath + dataset
	try:
		os.makedirs(attackerModelDataPath)
		os.makedirs(modelPath)
	except OSError:
		pass
	print("Training the Target model for {} epoch".format(num_epoch))
	targetTrain, targetTrainLabel  = load_data(dataPath + '/targetTrain.npz')
	targetTest,  targetTestLabel   = load_data(dataPath + '/targetTest.npz')
	attackModelDataTarget, attackModelLabelsTarget, targetModelToStore = trainTarget(classifierType,targetTrain, targetTrainLabel, X_test=targetTest, y_test=targetTestLabel, splitData= False, inepochs=num_epoch, batch_size=100) 
	np.savez(attackerModelDataPath + '/targetModelData.npz', attackModelDataTarget, attackModelLabelsTarget)
	np.savez(modelPath + '/targetModel.npz', *lasagne.layers.get_all_param_values(targetModelToStore))
	return attackModelDataTarget, attackModelLabelsTarget

def initializeShadowModel(dataset,num_epoch,dataFolderPath= './data/',modelFolderPath = './model/',classifierType = 'cnn'):
	dataPath = dataFolderPath+dataset+'/Preprocessed'
	attackerModelDataPath = dataFolderPath+dataset+'/attackerModelData'
	modelPath = modelFolderPath + dataset
	try:
		os.makedirs(modelPath)
	except OSError:
		pass
	print("Training the Shadow model for {} epoch".format(num_epoch))
	shadowTrainRaw, shadowTrainLabel  = load_data(dataPath + '/shadowTrain.npz')
	targetTestRaw,  shadowTestLabel   = load_data(dataPath + '/shadowTest.npz')
	attackModelDataShadow, attackModelLabelsShadow, shadowModelToStore = trainTarget(classifierType, shadowTrainRaw, shadowTrainLabel, X_test=targetTestRaw, y_test=shadowTestLabel, splitData= False, inepochs=num_epoch, batch_size=100) 
	np.savez(attackerModelDataPath + '/shadowModelData.npz', attackModelDataShadow, attackModelLabelsShadow)
	np.savez(modelPath + '/shadowModel.npz', *lasagne.layers.get_all_param_values(shadowModelToStore))
	return attackModelDataShadow, attackModelLabelsShadow
	
	

	

def generateAttackData(dataset, classifierType, dataFolderPath ,pathToLoadData ,num_epoch ,preprocessData ,trainTargetModel ,trainShadowModel,topX=3 ):
	attackerModelDataPath = dataFolderPath+dataset+'/attackerModelData'
	if(preprocessData):
		initializeData(dataset,pathToLoadData)
	
	if(trainTargetModel):	
		targetX, targetY = initializeTargetModel(dataset,num_epoch,classifierType =classifierType )
	else:
		targetX, targetY = load_data(attackerModelDataPath + '/targetModelData.npz')
	
	if(trainShadowModel):	
		shadowX, shadowY = initializeShadowModel(dataset,num_epoch,classifierType =classifierType)	
	else:
		shadowX, shadowY = load_data(attackerModelDataPath + '/shadowModelData.npz')
	
	targetX = clipDataTopX(targetX,top=topX)
	shadowX = clipDataTopX(shadowX,top=topX)
	return targetX, targetY, shadowX, shadowY
	
def attackerOne(dataset= 'CIFAR10',classifierType = 'cnn',dataFolderPath='./data/',pathToLoadData = './data/cifar-10-batches-py-official',num_epoch = 50,preprocessData=True,trainTargetModel = True,trainShadowModel=True):
	targetX, targetY, shadowX, shadowY = generateAttackData(dataset,classifierType,dataFolderPath,pathToLoadData,num_epoch,preprocessData,trainTargetModel,trainShadowModel)
	print("Training the attack model for the first adversary")
	trainAttackModel(targetX, targetY, shadowX, shadowY)  
	
	
def attackerTwo(dataset1= 'CIFAR10',dataset2= 'News',classifierType1 = 'cnn',classifierType2 = 'nn',dataFolderPath='./data/',pathToLoadData = './data/cifar-10-batches-py-official',num_epoch = 50,preprocessData=True,trainTargetModel = True,trainShadowModel=True):
	Dataset1X, Dataset1Y, _, _ = generateAttackData(dataset1,classifierType1,dataFolderPath,pathToLoadData,num_epoch,preprocessData,trainTargetModel,trainShadowModel)
	Dataset2X, Dataset2Y, _, _ = generateAttackData(dataset2,classifierType2,dataFolderPath,pathToLoadData,num_epoch,preprocessData,trainTargetModel,trainShadowModel)
	print("Training the attack model for the second adversary")
	trainAttackModel(Dataset1X, Dataset1Y, Dataset2X, Dataset2Y)  
	
def attackerThree(dataset= 'CIFAR10',classifierType = 'cnn',dataFolderPath='./data/',pathToLoadData = './data/cifar-10-batches-py-official',num_epoch = 50,preprocessData=True,trainTargetModel = True):
	targetX, targetY, _, _ = generateAttackData(dataset,classifierType,dataFolderPath,pathToLoadData,num_epoch,preprocessData,trainTargetModel,trainShadowModel=False,topX=1)
	print('AUC = {}'.format(roc_auc_score(targetY,targetX)))

if(opt.adv =='1'):
	attackerOne(dataset= opt.dataset,classifierType = opt.classifierType,dataFolderPath=opt.dataFolderPath,pathToLoadData = opt.pathToLoadData,num_epoch = opt.num_epoch,preprocessData=opt.preprocessData,trainTargetModel = opt.trainTargetModel, trainShadowModel = opt.trainShadowModel)
	
elif(opt.adv =='2'):
	attackerTwo(dataset1= opt.dataset,dataset2= opt.dataset2,classifierType1 = opt.classifierType,classifierType2 = opt.classifierType2,dataFolderPath=opt.dataFolderPath,pathToLoadData = opt.pathToLoadData,num_epoch = opt.num_epoch, preprocessData = opt.preprocessData, trainTargetModel = opt.trainTargetModel, trainShadowModel = opt.trainShadowModel)

elif(opt.adv =='3'):
	attackerThree(dataset= opt.dataset,classifierType =opt.classifierType,dataFolderPath=opt.dataFolderPath,pathToLoadData = opt.pathToLoadData,num_epoch = opt.num_epoch,preprocessData=opt.preprocessData,trainTargetModel = opt.trainTargetModel)
	