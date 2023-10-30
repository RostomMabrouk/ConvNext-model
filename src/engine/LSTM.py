from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i+n_steps
        if end_ix >len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def main():
    sequence = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    n_steps = 3
    X, y = split_sequence(sequence, n_steps)
    print(X)
    print('------------------------------------------------')
    #vanilla LSTM
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    model.fit(X,y, epochs=200, verbose=0)
    X_input =array([70,80,90])
    X_input = X_input.reshape(1, n_steps, n_features)
    yhat = model.predict(X_input, verbose= 0)
    print(yhat)

    print('------------------------------------------------')
    #Stacked LSTM
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X,y, epochs=200, verbose=1)
    X_input =array([70,80,90])
    X_input = X_input.reshape(1, n_steps, n_features)
    yhat = model.predict(X_input, verbose= 1)
    print(yhat)

    print('------------------------------------------------')
    #Bidirectionnal LSTM
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X,y, epochs=200, verbose=0)
    X_input =array([70,80,90])
    X_input = X_input.reshape(1, n_steps, n_features)
    yhat = model.predict(X_input, verbose= 0)
    print(yhat)
    print('------------------------------------------------')
    #CNN-LSTM
    n_features = 1
    n_steps = 4
    n_seq = 2
    X, y = split_sequence(sequence, n_steps)
    n_steps = 2
    X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X,y, epochs=500, verbose=0)
    X_input =array([70,80,90])
    X_input = X_input.reshape(1, n_seq, 1, n_steps, n_features)
    yhat = model.predict(X_input, verbose= 1)
    print(yhat)


if __name__=='__main__':
    main()

