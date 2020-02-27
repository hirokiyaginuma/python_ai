import numpy as np

INPUT_LAYER_SIZE = 1
HIDDEN_LAYER_SIZE = 2
OUTPUT_LAYER_SIZE = 2

def init_weights():
    Wh = np.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) * \
                np.sqrt(2.0/INPUT_LAYER_SIZE)
    Wo = np.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * \
                np.sqrt(2.0/HIDDEN_LAYER_SIZE)

    return Wh, Wo

def init_bias():
    Bh = np.full((1, HIDDEN_LAYER_SIZE), 0.1)
    Bo = np.full((1, OUTPUT_LAYER_SIZE), 0.1)
    return Bh, Bo

def relu(Z):
    return np.maximum(0, Z)

def relu_prime(Z):
    '''
    Z - weighted input matrix

    Returns gradient of Z where all
    negative values are set to 0 and
    all positive values set to 1
    '''
    Z[Z < 0] = 0
    Z[Z > 0] = 1
    return Z

def cost(yHat, y):
    cost = np.sum((yHat - y)**2) / 2.0
    return cost

def cost_prime(yHat, y):
    return yHat - y

def feed_forward(X):
    '''
    X    - input matrix
    Zh   - hidden layer weighted input
    Zo   - output layer weighted input
    H    - hidden layer activation
    y    - output layer
    yHat - output layer predictions
    '''

    Wh, Wo = init_weights()
    Bh, Bo = init_bias()

    print(X)
    print(Wh)
    print(Bh)
    print()

    # Hidden layer
    Zh = np.dot(X, Wh) + Bh
    H = relu(Zh)

    print(H)
    print(Wo)
    print(Bo)
    print()

    # Output layer
    Zo = np.dot(H, Wo) + Bo
    yHat = relu(Zo)

    return yHat

if __name__ == '__main__':

    '''
    print(np.random.randn(2, 2))
    print(np.sqrt(2.0 / 1))
    print()

    Wh = np.random.randn(2, 2) * \
         np.sqrt(2.0 / 1)
    Wo = np.random.randn(2, 2) * \
         np.sqrt(2.0 / 2)

    print(Wh)
    print(Wo)
    print()


    Bh = np.full((1, 2), 0.1)
    Bo = np.full((1, 2), 0.1)
    print(Bh)
    print(Bo)
    print(Bh + Bo)
    print()
    '''

    X = np.array([[3.0], [2.0]])

    # print(np.dot(X, Wh))

    print(feed_forward(X))
    print()

    """
    2 by 1 * 1 by 2 + 1 by 2
    
    2 by 2 * 2 by 2 + 1 by 2
    
    2 by 2
    """

    '''
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 1.0])
    print(x + y)
    '''