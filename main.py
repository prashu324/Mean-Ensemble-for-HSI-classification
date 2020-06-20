from dataset_Preparation import *
from preprocessing import *
from autoencoder import *
from individual_Classifiers import *
from utils import *
from sklearn.preprocessing import StandardScaler

dataset = 'IndianPines'
X, data_gt = loadDataset(dataset)

class_cnt = max(np.unique(data_gt.reshape([-1])))
windowSize = 9
testRatio = 0.9
optimizer = 'adam'
batch_size = 32
epochs = 80
apply_pca = False
n_comp = 10
code_sz = 30
method = 'uniform'

if(apply_pca == True):
    X_pca, pca = applyPCA(X = X, n_comp = n_comp)
    data = X_pca
else:
    data = X

data = np.array(data)
temp = np.reshape(data, (-1, data.shape[-1]))
standardScaler = StandardScaler()
temp = standardScaler.fit_transform(temp)

# Preparing Train and test sets for Autoencoder
X_train, X_test, y_train, y_test = train_test_split(temp, temp, test_size = 0.6)

# Building, training and encoding data using autoencoder
data, autoencoder, encoder, decoder = build_autoencoder(temp, X.shape[-1], code_sz, X_train, y_train, X_test, y_test)
data = np.reshape(data, (X.shape[0], X.shape[1], code_sz))

# Extracting Uniform LBP
lbp_image = create_ulbp(data, method, 8, 1)

# Preparing train, validation and test sets
hsi_train, hsi_val, hsi_test, lbp_train, lbp_val, lbp_test, y_train, y_val, y_test = create_train_val_test_sets(data, lbp_image, data_gt, testRatio = testRatio, windowSize = windowSize)
hsi_train1 = hsi_train[...,np.newaxis]
hsi_val1 = hsi_val[...,np.newaxis]
hsi_test1 = hsi_test[...,np.newaxis]

# Preparing dataset for final classification
hsi_final, hsi_final1, lbp_final, y_final = prepareFinalClassificationData(data, lbp_image, data_gt, windowSize)

# Creating four individual classifiers
# Creating, Training and testing Model1
model1 = build_model1(hsi_train, class_cnt)
model1 = trainModel(model1, [hsi_train], y_train, [hsi_val], y_val, epochs = epochs, batch_size = batch_size)
print("-------------Evaluating on test set--------")
test_pred_prob1 = model1.predict([hsi_test], batch_size = 512)
test_predictions1 = np.argmax(test_pred_prob1, axis = 1)
evaluateModel(y_test, test_predictions1, class_cnt, reshape = False)
print("-------------Evaluating on entire dataset--------")
final_pred_prob1 = model1.predict([hsi_final], batch_size = 512)
final_predictions1 = np.argmax(final_pred_prob1, axis = 1) + 1
final_predictions1[y_final == 0] = 0
final_predictions1 = np.reshape(final_predictions1, data_gt.shape)
evaluateModel(data_gt, final_predictions1, class_cnt, reshape = True)

# Creating, Training and testing Model2
model2 = build_model2(hsi_train = hsi_train, lbp_train = lbp_train, class_cnt = class_cnt)
model2 = trainModel(model2, [lbp_train, hsi_train], y_train, [lbp_val, hsi_val], y_val, epochs = epochs, batch_size = batch_size)
print("-------------Evaluating on test set--------")
test_pred_prob2 = model2.predict([lbp_test, hsi_test], batch_size = 512)
test_predictions2 = np.argmax(test_pred_prob2, axis = 1)
evaluateModel(y_test, test_predictions2, class_cnt, reshape = False)
print("-------------Evaluating on entire dataset--------")
final_pred_prob2 = model2.predict([lbp_final, hsi_final], batch_size = 512)
final_predictions2 = np.argmax(final_pred_prob2, axis = 1) + 1
final_predictions2[y_final == 0] = 0
final_predictions2 = np.reshape(final_predictions2, data_gt.shape)
evaluateModel(data_gt, final_predictions2, class_cnt, reshape = True)

# Creating, Training and testing Model3
model3 = build_model3(hsi_train1, class_cnt)
model3 = trainModel(model3, [hsi_train1], y_train, [hsi_val1], y_val, epochs = epochs, batch_size = batch_size)
print("-------------Evaluating on test set--------")
test_pred_prob3 = model3.predict([hsi_test1], batch_size = 512)
test_predictions3 = np.argmax(test_pred_prob3, axis = 1)
evaluateModel(y_test, test_predictions3, class_cnt, reshape = False)
print("-------------Evaluating on entire dataset--------")
final_pred_prob3 = model3.predict([hsi_final1], batch_size = 512)
final_predictions3 = np.argmax(final_pred_prob3, axis = 1) + 1
final_predictions3[y_final == 0] = 0
final_predictions3 = np.reshape(final_predictions3, data_gt.shape)
evaluateModel(data_gt, final_predictions3, class_cnt, reshape = True)

# Creating, Training and testing Model4
model4 = build_model4(lbp_train, hsi_train1, class_cnt)
model4 = trainModel(model4, [lbp_train, hsi_train1], y_train, [lbp_val, hsi_val1], y_val, epochs = epochs, batch_size = batch_size)
print("-------------Evaluating on test set--------")
test_pred_prob4 = model4.predict([lbp_test, hsi_test1], batch_size = 512)
test_predictions4 = np.argmax(test_pred_prob4, axis = 1)
evaluateModel(y_test, test_predictions4, class_cnt, reshape = False)
print("-------------Evaluating on entire dataset--------")
final_pred_prob4 = model4.predict([lbp_final, hsi_final1], batch_size = 512)
final_predictions4 = np.argmax(final_pred_prob4, axis = 1) + 1
final_predictions4[y_final == 0] = 0
final_predictions4 = np.reshape(final_predictions4, data_gt.shape)
evaluateModel(data_gt, final_predictions4, class_cnt, reshape = True)

showClassificationMap(final_predictions1, size=5)
showClassificationMap(final_predictions2, size=5)
showClassificationMap(final_predictions3, size=5)
showClassificationMap(final_predictions4, size=5)

# Preparing data for creating ensemble of individual classifiers
final_pred_prob = np.zeros((4,data_gt.shape[0] * data_gt.shape[1], class_cnt), dtype=np.float32)
final_pred_prob[0,:,:] = final_pred_prob1
final_pred_prob[1,:,:] = final_pred_prob2
final_pred_prob[2,:,:] = final_pred_prob3
final_pred_prob[3,:,:] = final_pred_prob4

final_predictions1 = np.reshape(final_predictions1, (-1))
final_predictions2 = np.reshape(final_predictions2, (-1))
final_predictions3 = np.reshape(final_predictions3, (-1))
final_predictions4 = np.reshape(final_predictions4, (-1))

final_predictions = np.zeros((4,data_gt.shape[0] * data_gt.shape[1]), dtype=np.float32)
final_predictions[0,:] = final_predictions1
final_predictions[1,:] = final_predictions2
final_predictions[2,:] = final_predictions3
final_predictions[3,:] = final_predictions4

# Computing the value of CFD of individual classifiers
cfd = computeCFD(final_predictions, data_gt)
print(cfd)

# Preparing data for 5 different combination strategies
max_ensemble_pred_prob = np.zeros((data_gt.shape[0] * data_gt.shape[1], class_cnt), dtype = np.float32)
min_ensemble_pred_prob = np.zeros((data_gt.shape[0] * data_gt.shape[1], class_cnt), dtype = np.float32)
med_ensemble_pred_prob = np.zeros((data_gt.shape[0] * data_gt.shape[1], class_cnt), dtype = np.float32)
mean_ensemble_pred_prob = np.zeros((data_gt.shape[0] * data_gt.shape[1], class_cnt), dtype = np.float32)
majority_voting_votes = np.zeros((data_gt.shape[0] * data_gt.shape[1], class_cnt), dtype = np.int16)

for i in range(final_pred_prob.shape[1]):
    max_ensemble_pred_prob[i] = [np.max(final_pred_prob[:,i,j]) for j in range(class_cnt)]
    min_ensemble_pred_prob[i] = [np.min(final_pred_prob[:,i,j]) for j in range(class_cnt)]
    med_ensemble_pred_prob[i] = [np.median(final_pred_prob[:,i,j]) for j in range(class_cnt)]
    mean_ensemble_pred_prob[i] = [np.mean(final_pred_prob[:,i,j]) for j in range(class_cnt)]
    for j in range(final_pred_prob.shape[0]):
        majority_voting_votes[i,np.argmax(final_pred_prob[j,i,:])] += 1

max_ensemble_predictions = np.argmax(max_ensemble_pred_prob, axis = 1) + 1
max_ensemble_predictions[y_final == 0] = 0
max_ensemble_predictions = np.reshape(max_ensemble_predictions, data_gt.shape)

min_ensemble_predictions = np.argmax(min_ensemble_pred_prob, axis = 1) + 1
min_ensemble_predictions[y_final == 0] = 0
min_ensemble_predictions = np.reshape(min_ensemble_predictions, data_gt.shape)

med_ensemble_predictions = np.argmax(med_ensemble_pred_prob, axis = 1) + 1
med_ensemble_predictions[y_final == 0] = 0
med_ensemble_predictions = np.reshape(med_ensemble_predictions, data_gt.shape)

mean_ensemble_predictions = np.argmax(mean_ensemble_pred_prob, axis = 1) + 1
mean_ensemble_predictions[y_final == 0] = 0
mean_ensemble_predictions = np.reshape(mean_ensemble_predictions, data_gt.shape)

voting_ensemble_predictions = np.argmax(majority_voting_votes, axis = 1) + 1
voting_ensemble_predictions[y_final == 0] = 0
voting_ensemble_predictions = np.reshape(voting_ensemble_predictions, data_gt.shape)

print("------------Evaluating Max ensemble-----------")
evaluateModel(data_gt, max_ensemble_predictions, class_cnt, reshape = True)
print("------------Evaluating Min ensemble-----------")
evaluateModel(data_gt, min_ensemble_predictions, class_cnt, reshape = True)
print("------------Evaluating Median ensemble-----------")
evaluateModel(data_gt, med_ensemble_predictions, class_cnt, reshape = True)
print("------------Evaluating Mean ensemble-----------")
evaluateModel(data_gt, mean_ensemble_predictions, class_cnt, reshape = True)
print("------------Evaluating Majority voting ensemble-----------")
evaluateModel(data_gt, voting_ensemble_predictions, class_cnt, reshape = True)

savemat('data/predictions_indianPines.mat', {'predictions' : np.uint8(mean_ensemble_predictions)})
