import numpy as np

def compute_cost(X, y, w, b):

    m = X.shape[0]
    f_wb = np.dot(X,w) + b
    error = np.square(f_wb - y)
    J = np.sum(error)/ (2*m)
   
    return J


def gradient_descent(X, y, w, b):

    f_wb = np.dot(X,w) + b
    m= X.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        
        dj_dw += (f_wb[i] - y[i])* X[i]
        
        dj_db += (f_wb[i] - y[i])
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db


def compute_gradient_descent(X, y, w, b, alpha, num_iters):


    j_history = []
    p_history = []

    for i in range(num_iters):

        dj_dw, dj_db = gradient_descent(X, y, w, b)

        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        
        j = compute_cost(X,y,w,b)
        p_history.append([w, b])
        j_history.append(j)

        if i%1000 == 0:
            print(f"Iteration: {i} | cost: {j} |  ")


    


    return w, b, j_history, p_history 



if __name__ == "__main__":

   
    x_train = np.array([1.0, 2.0])   #features
    y_train = np.array([300.0, 500.0])   #target value
    alpha = 1.0e-2
    w_in = 0
    b_in = 0
    num_iters = 10000

    w_out,b_out,j_history, p_history =  compute_gradient_descent(x_train, y_train, w_in, b_in, alpha, num_iters)
    print(f"w : {w_out} | b : {b_out}")

