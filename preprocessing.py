from sklearn.decomposition import PCA
import numpy as np
from skimage.feature import local_binary_pattern
import scipy

def applyPCA(X, n_comp):
    shape = X.shape
    X = np.reshape(X,(-1, X.shape[2]))
    pca = PCA(n_comp)
    Xpca = pca.fit_transform(X)
    Xpca = np.reshape(Xpca, (shape[0], shape[1], n_comp))
    return Xpca, pca

def create_ulbp(data, method, P = 8, r = 1):
    lbp_image = np.zeros(data.shape, dtype = np.float32)
    for i in range(data.shape[-1]):
        lbp_image[:,:,i] = local_binary_pattern(data[:,:,i], P, r, method = method)
    return lbp_image

def AugmentData(X_train):
    for i in range(int(X_train.shape[0]/2)):
        patch = X_train[i,:,:,:,:]
        num = np.random.randint(0,2)
        if (num == 0):
            flipped_patch = np.flipud(patch)
        if (num == 1):
            flipped_patch = np.fliplr(patch)
        if (num == 2):
            no = np.random.randrange(-180,180,30)
            flipped_patch = scipy.ndimage.interpolation.rotate(patch, no,axes=(1, 0),
                                                               reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=False)
    patch2 = flipped_patch
    X_train[i,:,:,:,:] = patch2

    return X_train
