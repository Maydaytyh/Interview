from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

X, Y = fetch_california_housing(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)

ones = np.ones(shape=(X.shape[0],1))
X = np.hstack([X,ones])
validate_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = validate_size,shuffle=True)


def get_batch(batchsize, X, Y):
    assert X.shape[0] % batchsize == 0
    batchnum = X.shape[0] // batchsize
    X_new = X.reshape((batchnum,batchsize,X.shape[1]))
    Y_new = Y.reshape((batchnum,batchsize,))

    for i in range(batchnum):
        yield X_new[i,:,:], Y_new[i,:]

def mse(X,Y,W):
    return 0.5*np.mean(np.square(X@W - Y))

def diff_mse(X,Y,W):
    return X.T@(X@W-Y)/X.shape[0]

lr = 0.001
num_epochs = 100
batch_size = 64
validate_every = 4
def plot_loss(loss_train, loss_validate, validate_every):
        # Your implementation here
    plt.figure()
    plt.plot(loss_train,label = 'train loss')
    plt.plot(range(0,len(loss_train),validate_every),loss_validate,label = 'validate loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def train(num_epochs, batch_size, validate_every,W0,X_train,Y_train,X_test,Y_test):
    loop = tqdm(range(num_epochs))
    loss_train = []
    loss_validate = []
    W = W0
    for epoch in loop:
        loss_train_epoch = 0
        for x_batch,y_batch in get_batch(batch_size,X_train,Y_train):
            loss_batch = mse(x_batch,y_batch,W)
            # print(loss_batch)
            loss_train_epoch += loss_batch*x_batch.shape[0]/X_train.shape[0]
            grad = diff_mse(x_batch,y_batch,W)
            W = W - lr*grad
        loss_train.append(loss_train_epoch)
        if epoch % validate_every == 0:
            loss_validate_epoch = mse(X_test,Y_test,W)
            loss_validate.append(loss_validate_epoch)
            print(f'Epoch:{epoch},train loss:{loss_train_epoch},validate loss:{loss_validate_epoch}')
    # Define the plot_loss function or import it from a module
    
    plot_loss(np.array(loss_train),np.array(loss_validate),validate_every)

W0 = np.random.random(size = (X.shape[1],))
train(num_epochs,batch_size,validate_every,W0,X_train,Y_train,X_test,Y_test)