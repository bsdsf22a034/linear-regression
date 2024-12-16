import numpy as np

def calculate_mean(values):
    return np.mean(values)

def calculate_slope(X,Y,mean_X,mean_Y):
    numerator=np.sum((X -mean_X)*(Y-mean_Y))
    denominator=np.sum((X-mean_X)**2)
    return numerator/denominator

def calculate_intercept(mean_X,mean_Y,slope):
    return mean_Y-slope*mean_X

def predict(X,theta_0,theta_1):
    return theta_0+theta_1*X

def calculate_mse(Y,Y_pred):
    return np.mean((Y-Y_pred)**2)

def gradient_descent(X,Y,theta_0, theta_1, learning_rate, iterations):
    m=len(X)
    for _ in range(iterations):
        Y_pred=predict(X, theta_0, theta_1)
        error=Y_pred - Y
        theta_0_gradient=(2/m) * np.sum(error)
        theta_1_gradient=(2/m) * np.sum(error * X)
        theta_0 -= learning_rate * theta_0_gradient
        theta_1 -= learning_rate * theta_1_gradient
    return theta_0, theta_1

def fit_linear_regression(X, Y, learning_rate=0.01, iterations=1000):
    mean_X=calculate_mean(X)
    mean_Y=calculate_mean(Y)
    slope=calculate_slope(X, Y, mean_X, mean_Y)
    intercept=calculate_intercept(mean_X, mean_Y, slope)
    theta_0, theta_1=gradient_descent(X, Y, intercept, slope, learning_rate, iterations)
    return theta_0, theta_1

def test_model(X, Y, learning_rate=0.01, iterations=1000):
    theta_0, theta_1 = fit_linear_regression(X, Y, learning_rate, iterations)
    Y_pred=predict(X, theta_0, theta_1)
    mse=calculate_mse(Y, Y_pred)
    print(f"Learned Model Parameters: Theta_0 (Intercept) = {theta_0}, Theta_1 (Slope) = {theta_1}")
    print(f"Mean Squared Error (MSE) = {mse}")
    return theta_0, theta_1, mse

X=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y=np.array([30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000])

test_model(X, Y, learning_rate=0.01, iterations=1000)
