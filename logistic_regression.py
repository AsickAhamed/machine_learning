import numpy as np


def sigmoid(x,w,b):
    z = np.dot(x,w) + b
    f_wb = 1 / (1 + np.exp(-z))

    return f_wb

def compute_cost(X, y, w, b):
    
    f_wb = sigmoid(X,w,b)
    m = X.shape[0]
    loss = 0
    for i in range(m):
        loss += (- y[i] * np.log(f_wb[i])) - ((1-y[i]) * (1 - np.log(f_wb[i])))
        
    return loss/m



def gradient_descent(X, y, w, b):

    m,n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    f_wb = sigmoid(X, w, b)
    for i in range(m):
        error = f_wb[i] - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i,j]
        dj_db += error
    
    return dj_dw/m, dj_db/m


def compute_gradient_descent(X, y, w, b, alpha, num_iters):

    m,n = X.shape
    j_history = []
    p_history = []

    for i in range(num_iters):
        dj_dw, dj_db = gradient_descent(X, y, w, b)

        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        j = compute_cost(X,y,w,b)
        j_history.append(j)
        p_history.append([w,b])

        if i%100 == 0:
            print(f"Iteration: {i} | cost: {j} |  ")

    print(f"w : {w} | b : {b}")




if __name__ == "__main__":




   

    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    w_tmp  = np.zeros_like(X_train[0])
    b_tmp = 0.
    alph = 0.1
    iters = 10000

    compute_gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)