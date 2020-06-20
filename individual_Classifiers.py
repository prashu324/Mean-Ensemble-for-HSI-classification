from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, Conv3D, Reshape, MaxPooling2D, Conv1D, MaxPooling1D, Flatten, Dense, Input, concatenate, Dropout, Average, MaxPooling3D, Activation, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import load_model, Model, Sequential

def trainModel(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('temp/.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    history = model.fit(x_train, y_train,epochs = epochs, batch_size = batch_size, validation_data = (x_val, y_val),
                        callbacks = [earlyStopping, mcp_save, reduce_lr_loss])
    return model

def build_model1(hsi_train, class_cnt):
    input2 = Input(shape = hsi_train[0].shape)
    conv4 = Conv2D(128, (3,3), activation = 'relu')(input2)
    conv5 = Conv2D(128, (3,3))(conv4)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.4)(conv5)
    conv5 = Dropout(0.4)(conv5)
    conv6 = Conv2D(128, (3,3), strides = (2,2))(conv5)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.4)(conv6)
    #conv7 = Conv2D(128, (5,5))(conv6)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Activation('relu')(conv7)
    #conv7 = Dropout(0.4)(conv7)
    flatten2 = Flatten()(conv6)
    flatten3 = Dropout(0.4)(flatten2)
    dense1 = Dense(150, activation = 'relu')(flatten3)
    output1 = Dense(class_cnt, activation = 'softmax')(dense1)

    model1 = Model(inputs = [input2], output = output1)
    model1.summary()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr = 0.0001)
    model1.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model1

def build_model2(hsi_train, lbp_train, class_cnt):
    input1 = Input(shape = lbp_train[0].shape)
    conv1 = Conv1D(32, 3, activation = 'relu')(input1)
    maxPool1 = MaxPooling1D(pool_size = 2)(conv1)
    conv2 = Conv1D(32, 3, activation = 'relu')(maxPool1)
    maxPool2 = MaxPooling1D(pool_size = 2)(conv2)
    conv3 = Conv1D(32, 3, strides = 2, activation = 'relu')(maxPool2)
    #maxPool3 = MaxPooling1D(pool_size = 2)(conv3)
    #maxPool3 = Dropout(0.25)(maxPool3)
    flatten1 = Flatten()(conv3)

    input2 = Input(shape = hsi_train[0].shape)
    conv4 = Conv2D(128, (3,3), activation = 'relu')(input2)
    conv5 = Conv2D(128, (3,3))(conv4)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.4)(conv5)
    conv5 = Dropout(0.4)(conv5)
    conv6 = Conv2D(128, (3,3), strides = (2,2))(conv5)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.4)(conv6)
    #conv7 = Conv2D(128, (5,5))(conv6)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Activation('relu')(conv7)
    #conv7 = Dropout(0.4)(conv7)
    flatten2 = Flatten()(conv6)

    flatten3 = concatenate([flatten1, flatten2])
    flatten3 = Dropout(0.4)(flatten3)
    dense1 = Dense(150, activation = 'relu')(flatten3)
    output1 = Dense(class_cnt, activation = 'softmax')(dense1)

    model2 = Model(inputs = [input1, input2], output = output1)
    model2.summary()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr = 0.0001)
    model2.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model2

def build_model3(hsi_train1, class_cnt):
    input4 = Input(shape = hsi_train1[0].shape)
    conv8= Conv3D(32, (3,3,5), activation = 'relu')(input4)
    conv8 = MaxPooling3D(pool_size = (1,1,2))(conv8)
    conv9 = Conv3D(32, (3,3,5), activation = 'relu')(conv8)
    conv9 = Dropout(0.4)(conv9)
    conv9 = Dropout(0.4)(conv9)
    reshaped = Reshape((5,5,9*32))(conv9)
    conv10 = Conv2D(128, (3,3), strides = (2,2), activation = 'relu', input_shape = ())(reshaped)
    conv10 = Dropout(0.4)(conv10)
    #conv11 = Conv2D(128, (5,5), activation = 'relu', input_shape = ())(conv10)
    #conv11 = Dropout(0.4)(conv11)
    flatten5 = Flatten()(conv10)
    flatten6 = Dropout(0.4)(flatten5)
    output2 = Dense(class_cnt, activation = 'softmax')(flatten6)

    model3 = Model(inputs = [input4], output = output2)
    model3.summary()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr = 0.0001)
    model3.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model3

def build_model4(lbp_train, hsi_train1, class_cnt):
    input3 = Input(shape = lbp_train[0].shape)
    conv13 = Conv1D(32, 3, activation = 'relu')(input3)
    maxPool4 = MaxPooling1D(pool_size = 2)(conv13)
    conv14 = Conv1D(32, 3, activation = 'relu')(maxPool4)
    maxPool5 = MaxPooling1D(pool_size = 2)(conv14)
    conv15 = Conv1D(32, 3, strides = 2, activation = 'relu')(maxPool5)
    #maxPool6 = MaxPooling1D(pool_size = 2)(conv15)
    #maxPool6 = Dropout(0.25)(maxPool6)
    flatten4 = Flatten()(conv15)

    input4 = Input(shape = hsi_train1[0].shape)
    conv8= Conv3D(32, (3,3,5), activation = 'relu')(input4)
    conv8 = MaxPooling3D(pool_size = (1,1,2))(conv8)
    conv9 = Conv3D(32, (3,3,5), activation = 'relu')(conv8)
    conv9 = Dropout(0.4)(conv9)
    conv9 = Dropout(0.4)(conv9)
    reshaped = Reshape((5,5,9*32))(conv9)
    conv10 = Conv2D(128, (3,3), strides = (2,2), activation = 'relu', input_shape = ())(reshaped)
    conv10 = Dropout(0.4)(conv10)
    #conv11 = Conv2D(128, (5,5), activation = 'relu', input_shape = ())(conv10)
    #conv11 = Dropout(0.4)(conv11)
    flatten5 = Flatten()(conv10)

    flatten6 = concatenate([flatten4, flatten5])
    flatten6 = Dropout(0.4)(flatten6)
    output2 = Dense(class_cnt, activation = 'softmax')(flatten6)

    model4 = Model(inputs = [input3, input4], output = output2)
    model4.summary()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr = 0.0001)
    model4.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model4
    