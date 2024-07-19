import numpy as np
class BatchNormalization:
    def __init__(self, dim, epsilon=1e-8):
        self.epsilon = epsilon
        self.gamma = np.ones((dim, 1))
        self.beta = np.zeros((dim, 1))
        
        # For moving averages during inference
        self.moving_mean = np.zeros((dim, 1))
        self.moving_variance = np.ones((dim, 1))
        
    def forward(self, X, training=True, momentum=0.9):
        if training:
            # Step 1: Compute the mean of the batch
            mean = np.mean(X, axis=1, keepdims=True)
            
            # Step 2: Compute the variance of the batch
            variance = np.var(X, axis=1, keepdims=True)
            
            # Step 3: Normalize the batch
            X_normalized = (X - mean) / np.sqrt(variance + self.epsilon)
            
            # Step 4: Scale and shift
            out = self.gamma * X_normalized + self.beta
            
            # Update moving averages
            self.moving_mean = momentum * self.moving_mean + (1 - momentum) * mean
            self.moving_variance = momentum * self.moving_variance + (1 - momentum) * variance
        else:
            # During inference, use the moving averages
            X_normalized = (X - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)
            out = self.gamma * X_normalized + self.beta
        
        return out
    
    def backward(self, d_out, X):
        m = X.shape[1]

        mean = np.mean(X, axis=1, keepdims=True)
        variance = np.var(X, axis=1, keepdims=True)
        X_normalized = (X - mean) / np.sqrt(variance + self.epsilon)
        
        # Gradients of scale and shift parameters
        d_gamma = np.sum(d_out * X_normalized, axis=1, keepdims=True)
        d_beta = np.sum(d_out, axis=1, keepdims=True)
        
        # Gradient of the normalized input
        d_X_normalized = d_out * self.gamma
        
        # Gradient of the variance
        d_variance = np.sum(d_X_normalized * (X - mean) * -0.5 * np.power(variance + self.epsilon, -1.5), axis=1, keepdims=True)
        
        # Gradient of the mean
        d_mean = np.sum(d_X_normalized * -1 / np.sqrt(variance + self.epsilon), axis=1, keepdims=True) + d_variance * np.sum(-2 * (X - mean), axis=1, keepdims=True) / m
        
        # Gradient of the input
        d_X = d_X_normalized / np.sqrt(variance + self.epsilon) + d_variance * 2 * (X - mean) / m + d_mean / m
        
        return d_X, d_gamma, d_beta
