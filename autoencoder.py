from keras.layers import Dense, Input
from keras.models import Model, Sequential

def build_encoder_decoder(inp_sz, code_sz):
    encoder = Sequential()
    encoder.add(Dense(80, input_dim = inp_sz))
    encoder.add(Dense(50))
    encoder.add(Dense(code_sz))
    
    decoder = Sequential()
    decoder.add(Dense(50, input_dim = code_sz))
    decoder.add(Dense(80))
    decoder.add(Dense(inp_sz))
    return encoder, decoder

def build_autoencoder(data, inp_size, code_sz, X_train, y_train, X_test, y_test):
    encoder, decoder = build_encoder_decoder(inp_size, code_sz)
    inp = Input(shape = (inp_size,))
    code = encoder(inp)
    recons = decoder(code)
    autoencoder = Model(inp, recons)
    autoencoder.compile(optimizer = 'adam', loss = 'msle', metrics = ['accuracy'])
    print(autoencoder.summary())
    history = autoencoder.fit(X_train, y_train, verbose = 1, epochs = 20, batch_size = 128, validation_data = (X_test, y_test))
    result = encoder.predict(data)
    return result, autoencoder, encoder, decoder
