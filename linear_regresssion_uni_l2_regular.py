import numpy as np

def compute_cost(X, y, w, b, lambda_):
    m = X.shape[0]
    f_wb = w * X + b
    error = np.square(f_wb - y)
    reg_term = (lambda_ / (2 * m)) * np.sum(w ** 2)
    J = np.sum(error) / (2 * m) + reg_term
    return J


def gradient_descent(X, y, w, b, lambda_):
    m = X.shape[0]
    f_wb = w * X + b
    dj_dw = (np.sum((f_wb - y) * X) / m) + (lambda_ / m) * w
    dj_db = np.sum(f_wb - y) / m
    return dj_dw, dj_db


def compute_gradient_descent(X, y, w, b, alpha, num_iters, lambda_):
    j_history = []
    p_history = []

    for i in range(num_iters):
        dj_dw, dj_db = gradient_descent(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        j = compute_cost(X, y, w, b, lambda_)
        p_history.append([w, b])
        j_history.append(j)

        if i % 1000 == 0:
            print(f"Iteration: {i} | cost: {j} | ")

    return w, b, j_history, p_history


    


    return w, b, j_history, p_history 


if __name__ == "__main__":
    x_train = np.array([1.0, 2.0]).reshape(-1, 1)
    y_train = np.array([300.0, 500.0]).reshape(-1, 1)
    alpha = 1.0e-2
    w_in = 0.0
    b_in = 0.0
    num_iters = 10000
    lambda_ = 0.1  # Regularization strength

    x_train = x_train.flatten()
    y_train = y_train.flatten()

    w_out, b_out, j_history, p_history = compute_gradient_descent(
        x_train, y_train, w_in, b_in, alpha, num_iters, lambda_
    )
    print(f"w : {w_out} | b : {b_out}")

