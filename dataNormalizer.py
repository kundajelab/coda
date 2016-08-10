import numpy as np

class DataNormalizer(object):
    """
    This class learns input scaling parameters and uses those parameters to apply input scaling
    to given data. It contains the fit() and transform() methods. 

    We suppose four types of input scaling:
        1) 'ZCA': Zero-mean, features linearly transformed to have unit covariance
        2) 'Z': Zero-mean, each feature independently scaled to unit variance
        3) '01': Zero-mean, each feature independently scaled to lie within [-1, 1]
        4) 'identity': Nothing happens to the input

    If not using 'ZCA', the transform() method does not create a copy. Instead, it modifies the 
    argument passed to it. If using 'ZCA', this behavior is ambiguous because of the use of 
    np.reshape, and you should assume that the argument passed to it could randomly be either 
    transformed or not (and so you should not make further use of the argument).

    All of these transformations are affine transformations. To represent them, each instance of the 
    class has two variables, W and b, which roughly correspond to the scale and translation factors
    for the different transformations.

    Sample usage: 
        normalizer = DataNormalizer('01')
        normalizer.fit(X_train)
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)
    """

    def __init__(self, mode):
        self.b = None
        self.W = None
        self.mode = mode
        if mode not in ['ZCA', 'Z', '01', 'identity']:
            raise ValueError, "mode=%s must be 'ZCA', 'Z', '01', or 'identity'" % mode


    def fit(self, X_orig):
        """
        Learns scaling parameters on the X_orig dataset. Does not modify X_orig.
        """        
        if len(X_orig.shape) != 2 and len(X_orig.shape) != 3:
            raise ValueError, "X must be either a 3-tensor of shape num_examples x seq_length x \
                               num_input_marks, or a 2-tensor of shape num_examples x num_input_marks"
        if self.mode == 'identity':
            return None        

        X = np.copy(X_orig)
        num_input_marks = X.shape[-1]

        # If X is a 3-tensor, reshape X such that it is a 2-tensor of shape 
        # (num_examples * seq_length) x num_input_marks. 
        if len(X.shape) == 3:    
            X = np.reshape(X, (-1, num_input_marks))
        
        self.b = np.mean(X, axis=0) 

        X -= self.b

        if self.mode == 'ZCA':
            sigma = np.dot(X.T, X) / X.shape[0]
            U, S, V = np.linalg.svd(sigma)
            self.W = np.dot(
                np.dot(U, np.diag(1 / np.sqrt(S + 1e-5))),
                U.T)
        elif self.mode == 'Z':
            self.W = np.empty(num_input_marks)
            for idx in range(num_input_marks):
                self.W[idx] = np.std(X[:, idx])
        elif self.mode == '01':
            self.W = np.empty(num_input_marks)
            for idx in range(num_input_marks):
                self.W[idx] = np.max(np.abs(X[:, idx]))

        return None            


    def transform(self, X):
        if len(X.shape) != 2 and len(X.shape) != 3:
            raise ValueError, "X must be either a 3-tensor of shape num_examples x seq_length x \
                               num_input_marks, or a 2-tensor of shape num_examples x num_input_marks"

        if self.mode == 'identity':
            return X
            
        assert self.b is not None
        assert self.W is not None

        num_input_marks = X.shape[-1]
        orig_shape = X.shape

        if self.mode == 'ZCA':            
            X = np.reshape(X, (-1, num_input_marks))
            if self.W.shape[1] != X.shape[1]:
                raise ValueError, "When doing a ZCA transform, X and W must have the same number of columns."
            X = np.dot(
                X - self.b,
                self.W.T)
            X = np.reshape(X, orig_shape)
        elif self.mode in ['Z', '01']:
            if (len(self.b) != num_input_marks) or (len(self.W) != num_input_marks):
                print("X.shape: ", X.shape)
                print("b.shape: ", self.b.shape)
                print("W.shape: ", self.W.shape)
                raise ValueError, "The shapes of X, b, and W must all share the same last dimension."                
            for idx in range(num_input_marks):
                X[..., idx] = (X[..., idx] - self.b[idx]) / self.W[idx]

        return X
