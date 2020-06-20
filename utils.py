import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import spectral

def evaluateModel(gtValues, predictions, class_cnt, reshape = True):
    temp = 0
    if reshape:
        gtValues = np.reshape(gtValues, (-1))
        predictions = np.reshape(predictions,(-1))
        temp = 1
    accuracy = (np.sum(gtValues == predictions)) / (gtValues.shape[0]) * 100
    c_m = confusion_matrix(gtValues,predictions)
    print('Confusion matrix\n')
    class_accuracy = []
    for i in range(temp,class_cnt+temp):
        sum = 0
        for j in range(temp,class_cnt+temp):
            print(c_m[i][j], end = '\t')
            sum += c_m[i][j]
        class_accuracy.append(c_m[i][i]/sum*100)
        print(c_m[i][i]/sum*100, end = '\t')
        print('\n')
    print('Overall Accuracy : ', accuracy)
    kappa_score = cohen_kappa_score(gtValues, predictions) * 100
    print('Average Accuracy : ', np.sum(class_accuracy)/class_cnt)
    print('Kappa coefficient : ', kappa_score)

def showFalseColor(data, bands, size = None):
    if(size != None):
        size = (size,size)
    spectral.imshow(data, bands = bands, figsize = size)

def showClassificationMap(data_gt, size = None):
    if(size != None):
        size = (size,size)
    spectral.imshow(classes = data_gt, figsize = size)

def computeCFD(final_predictions, data_gt):
    p = [0,0,0,0,0]
    sample_count = 0
    gtValues = np.reshape(data_gt, (-1))
    L = 4
    cfd = 0
    for i in range(final_predictions.shape[1]):
        if(gtValues[i] != 0):
            sample_count += 1
            count = 0
            for j in range(L):
                count += int(gtValues[i] != final_predictions[j][i])
            p[count] += 1

    p = np.array(p, dtype = np.float16)
    p[:] = p[:]/sample_count
    for i in range(1,L):
        cfd += (L-i)/(L-1)*p[i]
    cfd = cfd / (1-p[0])
    return cfd
