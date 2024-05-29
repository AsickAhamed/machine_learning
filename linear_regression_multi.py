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

        if i%10000 == 0:
            print(f"Iteration: {i} | cost: {j} |  ")

    print(f"w : {w} | b : {b}")
    


    return w, b, j_history, p_history 



if __name__ == "__main__":

    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])
    # w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
    initial_w = np.zeros(4)
    print(initial_w)
    initial_b = 0.
    iterations = 1000
    alpha = 5.0e-7
    # run gradient descent 
    w_final, b_final, J_hist, p = compute_gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    alpha, iterations)
    

    for i in range(3):
        print(f"prediction : {np.dot(X_train[i],w_final) + b_final}, target value : {y_train[i]}")





