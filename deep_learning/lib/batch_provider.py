import numpy as np

class BatchProvider():
    """
    This is a helper class to conveniently access mini batches of training, testing and validation data
    """

    def __init__(self, X, Y, indices):  # indices don't always cover all of X and Y (e.g. in the case of val set)

        self.X = X
        self.Y = Y
        self.indices = indices

    def next_batch(self, batch_size):
        """
        Get a single random batch
        """

        batch_indices = np.random.choice(self.indices, batch_size, replace=False)

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(batch_indices)

        X_batch = self.X[batch_indices, ...]
        Y_batch = self.Y[batch_indices, ...]
        
        # for some reason border pixels that are exactly 0.0 cause numerical instability
        # not sure why but this fixes the problem

        if X_batch.ndim == 4:  # only do this if working on ACDC data
            X_batch[X_batch == 0.0] = np.mean(X_batch)

        return X_batch, Y_batch

    def iterate_batches(self, batch_size):
        """
        Get a range of batches. Use as argument of a for loop like you would normally use 
        the range() function. 
        """

        np.random.shuffle(self.indices)
        N = self.indices.shape[0]

        for b_i in range(0, N, batch_size):

            if b_i + batch_size > N:
                continue

            # HDF5 requires indices to be in increasing order
            batch_indices = np.sort(self.indices[b_i:b_i + batch_size])

            X_batch = self.X[batch_indices, ...]
            Y_batch = self.Y[batch_indices, ...]
            
            # for some reason border pixels that are exactly 0.0 cause numerical instability
            # not sure why but this fixes the problem

            if X_batch.ndim == 4:  # only do this if working on ACDC data
                X_batch[X_batch == 0.0] = np.mean(X_batch)

            yield X_batch, Y_batch