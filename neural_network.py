import numpy as np
import pandas as pd
import cv2

data = pd.read_csv('csv/train.csv')

data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def feed_forward(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def back_propagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def gradient_descent(alpha, iterations):
    try:
        W1 = np.load('model/w1.npy')
        b1 = np.load('model/b1.npy')
        W2 = np.load('model/w2.npy')
        b2 = np.load('model/b2.npy')

        return W1, b1, W2, b2
    except:
        W1, b1, W2, b2 = init_params()
        for i in range(iterations):
            Z1, A1, Z2, A2 = feed_forward(W1, b1, W2, b2, X_train)
            dW1, db1, dW2, db2 = back_propagation(Z1, A1, Z2, A2, W1, W2, X_train, Y_train)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = get_predictions(A2)
                print("Accuracy: ", get_accuracy(predictions, Y_train))

        np.save('model/w1.npy', W1)
        np.save('model/b1.npy', b1)
        np.save('model/w2.npy', W2)
        np.save('model/b2.npy', b2)

        return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = feed_forward(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255

def get_formatted_img_from_path(path):
    img = cv2.imread(path, 0)
    img_reverted= cv2.bitwise_not(img)
    new_img = img_reverted / 255.0

    flat = []
    for i in new_img.flat:
        flat.append([i])

    return np.array(flat)

def translate_output(out):
    if (out == 6):
        print('Primeiro número.')
    elif (out == 4):
        print('Segundo número.')
    else:
        print('Número não reconhecido.')

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y