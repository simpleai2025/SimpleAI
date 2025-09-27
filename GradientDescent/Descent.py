import numpy as np
'''
Gradient descent algorithm for problems where the solution equation is:
y = mx + b
'''
def gradient_descent(x,y,learning_rate = 0.0001,num_iter = 1000) -> None:

    m = b = 0
    n = len(x)
    
    for _ in range(num_iter):

        y_predicted = x * m + b # Prediction

        # Code to adjust the weights using the derivate of the cost function
        # and find the local minimun.
        dm = -(2/n) * sum(x*(y - y_predicted))
        db = -(2/n) * sum(y - y_predicted)

        m = m - learning_rate * dm
        b = b - learning_rate * db

    print("{},{}".format(m,b))

'''
Stochastic gradient descent algorithm for problems where the solution is:
y = mx + q
The difference is that in the stochastic algorithm the weights are adjusted based 
on the derivate of only one data of the dataset, picked randomly.
'''
def stochastic_gradient_descent(x,y,learning_rate = 0.0001,num_iter = 1000) -> None:

    m = b = 0

    for _ in range(num_iter):
        
        #I picked a random data from the dataset
        i = np.random.randint(0,len(x))
        y_predicted = m * x[i] + b

        dm = -2 * (x[i]*(y[i] - y_predicted))
        db = -2 * (y[i] - y_predicted)

        m = m - learning_rate * dm
        b = b - learning_rate * db

    print("{},{}".format(m,b))

'''
Mini batch gradient descent algorithm for problems where the solution is:
y = mx + b
The difference is that is the other algorithm, for each iteration, the weights are adjusted 
based on the derivate of one data of the dataset or all the dataset.
Here we consider only a part of the dataset to adjust the weights.
For this reason there is the variable group size.
'''
def mini_batch_gradient_descent(x,y,learning_rate = 0.0001, num_iter = 1000, 
                                group_size = 50) -> None:
    
    m = b = 0

    for _ in range(num_iter):

        start_i = np.random.randint(0,len(x) - group_size)
        #Code to create the random group
        group_x = np.array(x[start_i:start_i+group_size])
        group_y = np.array(y[start_i:start_i+group_size])

        y_predicted = m * group_x + b

        dm = -(2/group_size) * sum(group_x*(group_y - y_predicted))
        db = -(2/group_size) * sum(group_y - y_predicted)

        m = m - learning_rate * dm
        b = b - learning_rate * db

    print("{},{}".format(m,b))
        

if __name__ == '__main__':
    x = np.random.randint(0,100,100)
    y = np.array([num*2 + 1 for num in x])
    print("GRADIENT DESCENT:")
    gradient_descent(x,y)
    print("STOCHASTIC GRADIEDNT DESCENT:")
    stochastic_gradient_descent(x,y)
    print("MINI BATCH GRADIENT DESCENT:")
    mini_batch_gradient_descent(x,y)