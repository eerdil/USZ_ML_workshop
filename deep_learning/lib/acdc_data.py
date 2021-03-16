
import numpy as np
import h5py

from batch_provider import BatchProvider

# Load data using h5py
data_path = './deep_learning/data/acdc_preprocessed.hdf5'
data = h5py.File(data_path, 'r')

# Extract pointers to the test and train data
images_train = data['images_train']
images_test = data['images_test']

masks_train = data['masks_train']
masks_test = data['masks_test']

# Extract the number of training and testing points
N_train = images_train.shape[0]
N_test = images_test.shape[0]

# Create a shuffled range of indices for both training and testing data
train_indices = np.arange(N_train)
np.random.shuffle(train_indices)

test_indices = np.arange(N_test)
np.random.shuffle(test_indices)

# Split another 20% of the training data for validation purposes
val_fraction = 0.2
N_val = int(val_fraction*N_train)
val_indices = np.random.choice(train_indices, N_val, replace=False)

# Reset train to whatever isn't val
train_indices = np.setdiff1d(train_indices, val_indices)
N_train = train_indices.shape[0]


# Create the batch providers
train = BatchProvider(images_train, masks_train, train_indices)
validation = BatchProvider(images_train, masks_train, val_indices)
test = BatchProvider(images_test, masks_test, test_indices)


if __name__ == '__main__':

    # If the program is called as main, perform some debugging operations

    print('DEBUGGING OUTPUT')
    print('training')
    for ii in range(2):
        X_tr, Y_tr = train.next_batch(10)
        print(np.mean(X_tr))
        print(np.mean(Y_tr))
        print('--')

    print('test')
    for ii in range(2):
        X_te, Y_te = test.next_batch(10)
        print(np.mean(X_te))
        print(np.mean(Y_te))
        print('--')

    print('validation')
    for ii in range(2):
        X_va, Y_va = validation.next_batch(10)
        print(np.mean(X_va))
        print(np.mean(Y_va))
        print('--')