from scipy.io import loadmat, savemat
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

def loadDataset(dataset):
    if(dataset == 'IndianPines'):
        data_gt = loadmat('Dataset/Indian_pines_gt.mat')
        data_gt = data_gt['indian_pines_gt']
        X = loadmat('Dataset/Indian_pines_corrected.mat')
        X = X['indian_pines_corrected']
    elif dataset == 'Salinas':
        data_gt = loadmat('Dataset/Salinas_gt.mat')
        data_gt = data_gt['salinas_gt']
        X = loadmat('Dataset/Salinas_corrected.mat')
        X = X['salinas_corrected']
    elif dataset == 'PaviaU':
        data_gt = loadmat('Dataset/PaviaU_gt.mat')
        data_gt = data_gt['paviaU_gt']
        X = loadmat('Dataset/PaviaU.mat')
        X = X['paviaU']
    elif dataset == 'KSC':
        data_gt = loadmat('Dataset/KSC_gt.mat')
        data_gt = data_gt['KSC_gt']
        X = loadmat('Dataset/KSC.mat')
        X = X['KSC']
    return X, data_gt

def createPatches(X, y, windowSize = 5, removeZeros = True):
    margin = int((windowSize-1) /2)
    newX = np.zeros((X.shape[0] + 2*margin, X.shape[1] + 2*margin, X.shape[2], X.shape[3]), dtype = np.float64)
    newX[margin:margin + X.shape[0], margin:margin + X.shape[1], :] = X
    patchIndex = 0
    patches = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2],X.shape[3]), dtype = np.float64)
    patchLabels = np.zeros((X.shape[0] * X.shape[1]))
    for r in range(margin, newX.shape[0] - margin):
        for c in range(margin, newX.shape[1] - margin):
            newPatch = newX[r-margin:r+margin+1, c-margin:c+margin+1]
            patches[patchIndex,:,:,:,:] = newPatch
            patchLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex += 1
    if removeZeros:
        patches = patches[patchLabels>0,:,:,:,:]
        patchLabels = patchLabels[patchLabels>0]
        patchLabels -= 1
    return patches, patchLabels

def oversampleWeakClasses(X, y):
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts
    newX = X[y == uniqueLabels[0], :, :, :, :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y== label,:,:,:,:].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :, :]
    newY = newY[rand_perm]
    return newX, newY

def create_combined_data(data, lbp_image):
    shape = list(data.shape)
    combData = np.zeros((shape[0],shape[1],shape[2],2), dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                combData[i,j,k,0] = data[i,j,k]
                combData[i,j,k,1] = lbp_image[i,j,k]
    return combData

def prepareFinalClassificationData(data, lbp_image, data_gt, windowSize = 9):
    combData = create_combined_data(data, lbp_image)
    XPatches, yPatches = createPatches(combData, data_gt, windowSize=windowSize, removeZeros = False)
    hsi = XPatches[:,:,:,:,0]
    hsi1 = hsi[...,np.newaxis]
    lbp = []
    margin = int((windowSize-1)/2)
    for i in range(XPatches.shape[0]):
        lbp.append(XPatches[i,margin,margin,:,1])
    lbp = np.array(lbp)
    lbp = lbp[...,np.newaxis]
    return hsi, hsi1, lbp, yPatches

def create_train_val_test_sets(data, lbp_image, data_gt, testRatio, windowSize = 9):
    combData = create_combined_data(data, lbp_image)
    XPatches, yPatches = createPatches(combData, data_gt, windowSize=windowSize, removeZeros = True)
    X_train, X_test, y_train, y_test = train_test_split(XPatches, yPatches, test_size = testRatio, random_state = 4)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.2, random_state = 4)
    X_train, y_train = oversampleWeakClasses(X_train, y_train)
    # X_train = AugmentData(X_train)
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    hsi_train, hsi_val, hsi_test = prepare_hsi_sets(X_train, X_val, X_test)
    lbp_train, lbp_val, lbp_test = prepare_lbp_sets(X_train, X_val, X_test, windowSize = windowSize)
    return hsi_train, hsi_val, hsi_test, lbp_train, lbp_val, lbp_test, y_train, y_val, y_test

def prepare_hsi_sets(X_train, X_val, X_test):
    hsi_train = X_train[:,:,:,:,0]
    hsi_val = X_val[:,:,:,:,0]
    hsi_test = X_test[:,:,:,:,0]
    return hsi_train, hsi_val, hsi_test

def prepare_lbp_sets(X_train, X_val, X_test, windowSize = 9):
    lbp_train = []
    lbp_val = []
    lbp_test = []
    margin = int((windowSize-1)/2)
    for i in range(X_train.shape[0]):
        lbp_train.append(X_train[i,margin,margin,:,1])
    lbp_train = np.array(lbp_train)
    lbp_train = lbp_train[...,np.newaxis]
    for i in range(X_val.shape[0]):
        lbp_val.append(X_val[i,margin,margin,:,1])
    lbp_val = np.array(lbp_val)
    lbp_val = lbp_val[...,np.newaxis]
    for i in range(X_test.shape[0]):
        lbp_test.append(X_test[i,margin,margin,:,1])
    lbp_test = np.array(lbp_test)
    lbp_test = lbp_test[...,np.newaxis]
    
    return lbp_train, lbp_val, lbp_test
