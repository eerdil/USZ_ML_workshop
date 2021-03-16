
import numpy as np
from batch_provider import BatchProvider
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set data path
data_path = '../data/pima-indians-diabetes_data.txt'

table = np.genfromtxt(data_path, delimiter=',', dtype="|U5")

y = table[:,-1]
X = table[:,:-1]

X = np.where(X=='?', np.nan, X)

X = np.float32(X)
y = np.uint8(y)

y = y[~np.isnan(X).any(axis=1)]
X = X[~np.isnan(X).any(axis=1)]

X_raw = X
y_raw = y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# important: only learn the scaling on the train data. In practice we might not have the test data, so 
# this would be like looking into the future. We cannot use any attributes from the test data to train the
# algorithm. 

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

N_train = X_train.shape[0]
N_test = X_test.shape[0]
N_val = X_val.shape[0]

# Create a shuffled range of indices for both training and testing data
train_indices = np.arange(N_train)
test_indices = np.arange(N_test)
val_indices = np.arange(N_val)

# Create the batch providers
train = BatchProvider(X_train, y_train, train_indices)
validation = BatchProvider(X_test, y_test, val_indices)
test = BatchProvider(X_val, y_val, test_indices)


if __name__ == '__main__':

    print('DEBUGGING OUTPUT')
    pass
