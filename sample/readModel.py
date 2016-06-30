import os, scipy.io
import numpy as np
from poseParamsBounds import getBounds

import sys
import inspect

def getCurDir():
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.insert(0, os.path.join(getCurDir(), "../lib"))
from shapemodel import shapepose

def getModelPath():
    return os.path.join(getCurDir(), '../data/HumanShape')

def getFacesPath():
    return os.path.join(getCurDir(), '../3dParty/humanshape/fitting/')

'''
Computes ratio between the result model size and the input model size and location of the mean point
* in_personModel is the person model
* in_modelHeight is the result height of the mean model
* out_sizeRatio  is the result ratio
* out_location   is the location of the mean point of the mean person mesh before resize
'''
def getModelStatistics(in_personModel, in_modelHeight):
    poseParams = np.zeros((1, 31), dtype = np.float64)
    nShapeParams = in_personModel['eVectors'].shape[1]
    shapeParams = np.zeros((1, nShapeParams), dtype = np.float64)
    curModel, jointArray = shapepose(poseParams, shapeParams,
            in_personModel['eVectors'], in_personModel['modelPath'])
    out_sizeRatio = in_modelHeight / (np.max(curModel[:, 2]) - np.min(curModel[:, 2]))
    out_location = np.mean(curModel, axis = 0).reshape((1, 3))
    return out_sizeRatio, out_location

def ReadShape(in_type):
    dataPath = os.path.join(getModelPath(), in_type)

    evectorsFileName = os.path.join(dataPath, "evectors.mat")
    evaluesFileName = os.path.join(dataPath, "evalues.mat")

    evectors = scipy.io.loadmat(evectorsFileName)['evectors']
    evalues = scipy.io.loadmat(evaluesFileName)['evalues']
    return evectors, evalues, dataPath

def ReadFaces():
    dataPath = getFacesPath()
    faceFileName = os.path.join(dataPath, "facesShapeModel.mat")
    return scipy.io.loadmat(faceFileName)['faces']

def getPoseBounds():
    tmp = 1e3
    out_bounds = getBounds()
    mask = np.where(out_bounds['min'] < -tmp)[0]
    out_bounds['min'][mask] = -tmp
    mask = np.where(out_bounds['max'] > tmp)[0]
    out_bounds['max'][mask] = tmp
    return out_bounds

def ReadModel(in_modelName, in_modelHeight):
    nPCA = 20
    out_personModel = {'eVectors': [], 'eValues': [], 'modelPath': '',
            'sizeRatio': np.array([]), 'meanLocation': [], 'poseBounds': []}
    out_personModel['eVectors'], out_personModel['eValues'], out_personModel['modelPath'] = ReadShape(in_modelName)
    out_personModel['faceArray'] = ReadFaces().astype(np.uint64) - 1
    out_personModel['eVectors'] = out_personModel['eVectors'][0 : nPCA, :].copy('F').transpose()
    out_personModel['eValues'] = out_personModel['eValues'][:, 0 : nPCA]
    out_personModel['poseBounds'] = getPoseBounds()
    sizeRatio, location = getModelStatistics(out_personModel, in_modelHeight)
    out_personModel['sizeRatio'] = sizeRatio
    out_personModel['meanLocation'] = location
    return out_personModel
