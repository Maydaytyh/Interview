import numpy as np
def init_parameters(layers_dim):
    L = len(layers_dim)
    parameters = {}
    for i in range(1,L):
        parameters['W'+str(i)] = np.random.random([layers_dim[i],layers_dim[i-1]])
        parameters['b'+str(i)] = np.zeros([layers_dim[i],1])
    return parameters

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def diff_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def forward(x,parameters):
    a = []
    z = []
    caches = {}
    a.append(x)
    z.append(x)
    layers = len(parameters) // 2
    for i in range(1,layers):
        z_temp = parameters["W"+str(i)]@x + parameters["b"+str(i)]
        z.append(z_temp)
        a.append(sigmoid(z_temp))
    z_temp = parameters["W"+str(layers)]@a[layers-1] + parameters["b"+str(layers)]
    z.append(z_temp)
    a.append(z_temp)
    caches["z"] = z
    caches["a"] = a
    return caches,a[layers]

def backward(y,caches,parameters,al):
    layers = len(parameters) // 2
    grades = {}
    m = y.shape[1]

    grades["dz"+str(layers)] = al - y
    grades["dW"+str(layers)] = grades["dz"+str(layers)]@caches["a"][layers-1].T/m
    grades["db"+str(layers)] = np.sum(grades["dz"+str(layers)],axis = 1,keepdims = True)/m

    for i in reversed(range(1,layers)):
        grades["dz"+str(i)] = parameters["W"+str(i+1)].T @ grades["dz"+str(i+1)] * diff_sigmoid(caches["z"][i])
        grades["dW"+str(i)] = grades["dz"+str(i)]@caches["a"][i-1].T/m
        grades["db"+str(i)] = np.sum(grades["dz"+str(i)],axis=1,keepdims=True)/m
    return grades

def update_grades(parameters,grades,lr):
    layers = len(parameters) // 2
    for i in range(1,layers):
        parameters["W"+str(i)] -= lr * grades["dW"+str(i)]
        parameters["b"+str(i)] -= lr * grades["db"+str(i)]
    return parameters

def compute_loss(al,y):
    return np.mean(np.square(al-y))

def load_data():
    x = np.arange(0.0,1.0,0.01)
    y = 20 * np.sin(2*np.pi*x)
    return x,y

x,y = load_data()
x = x.reshape(1,100)
y = y.reshape(1,100)

parameters = init_parameters([1,100,1])
al = 0
for i in range(40000):
    caches, al = forward(x,parameters)
    grades = backward(y,caches,parameters,al)
    parameters = update_grades(parameters,grades,0.3)
    if i % 100 == 0:
        loss = compute_loss(al,y)
        print(f"Epoch:{i},loss:{loss}")
