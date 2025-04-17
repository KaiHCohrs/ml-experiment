import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer
from sklearn.model_selection import train_test_split

from jax import vmap, random, jit
from jax import numpy as jnp
from functools import partial

def build_data_loaders(dataset_config, data):
    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=dataset_config['test_portion'], random_state=dataset_config['seed'])

    # Create scalers for inputs and targets
    scaler_inputs = StandardScaler()
    scaler_output = MaxAbsScaler()

    # Fit scalers on training data
    scaler_inputs.fit(train_data.loc[:,dataset_config['inputs']].values)
    scaler_output.fit(train_data.loc[:,dataset_config['targets']].values.reshape(-1, 1))

    # Create custom datasets for training and validation
    train_dataset = NNModelDataset(train_data, scaler_inputs, scaler_output, dataset_config['var_inputs'], dataset_config['var_targets'])
    val_dataset = NNModelDataset(val_data, scaler_inputs, scaler_output, dataset_config['var_inputs'], dataset_config['var_targets'])

    # Create custom data loaders
    batch_size = dataset_config['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, [scaler_inputs, scaler_output]

class NNModelDataset(Dataset):
    def __init__(self, data, scaler_inputs, scaler_output, var_inputs, var_targets):
        self.data = data
        self.scaler_inputs = scaler_inputs
        self.scaler_output = scaler_output
        self.var_inputs = var_inputs
        self.var_targets = var_targets

        # Standardize inputs
        self.X = self.scaler_inputs.transform(self.data.loc[:,var_inputs].values)

        # Normalize targets
        self.y = self.scaler_output.transform(self.data.loc[:,var_targets].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.X[idx]), torch.Tensor(self.y[idx])



class CustomBootstrapLoader(Dataset):
    def __init__(
        self,
        X,
        y,
        batch_size=128,
        ensemble_size=32,
        split=0.8,
        rng_key=random.PRNGKey(1234),
    ):
        #'Initialization'
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.split = split
        self.key = rng_key

        if self.N < self.batch_size:
            self.batch_size = self.N

        # Create the bootstrapped partitions
        keys = random.split(rng_key, ensemble_size)
        if split < 1:
            self.data_train, self.data_val = vmap(self.__bootstrap, (None, None, 0))(
                X, y, keys
            )
            (self.X_train, self.y_train) = self.data_train
        else:
            self.data_train, self.data_val = vmap(
                self.__bootstrap_train_only, (None, None, 0)
            )(X, y, keys)
            (self.X_train, self.y_train) = self.data_train

        # Each bootstrapped data-set has its own normalization constants
        self.norm_const = vmap(self.normalization_constants, in_axes=(0, 0))(
            self.X_train, self.y_train
        )

        # For analysis reasons
        self.norm_const_val = vmap(self.normalization_constants, in_axes=(0, 0))(
            *self.data_val
        )

    def normalization_constants(self, X, y):
        mu_X, sigma_X = X.mean(0), X.std(0)
        mu_y, sigma_y = jnp.zeros(
            y.shape[1],
        ), jnp.abs(
            y
        ).max(0) * jnp.ones(
            y.shape[1],
        )

        return (mu_X, sigma_X), (mu_y, sigma_y)

    def __bootstrap(self, X, y, key):
        # TODO Proper Bootstrap is happening outside. In here we take the whole dataset and split it
        idx = random.choice(key, self.N, (self.N,), replace=False)
        idx_train = idx[: jnp.floor(self.N * self.split).astype(int)]
        idx_test = idx[jnp.floor(self.N * self.split).astype(int) :]

        inputs_train = X[idx_train, :]
        targets_train = y[idx_train, :]

        inputs_test = X[idx_test, :]
        targets_test = y[idx_test, :]

        return (inputs_train, targets_train), (inputs_test, targets_test)

    def __bootstrap_train_only(self, X, y, key):
        idx = random.choice(key, self.N, (self.N,), replace=False).sort()

        inputs_train = X[idx]
        targets_train = y[idx]

        inputs_test = X[idx]
        targets_test = y[idx]

        return (inputs_train, targets_train), (inputs_test, targets_test)

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key, X, y, norm_const):
        "Generates data containing batch_size samples"
        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        X = X[idx, :]
        y = y[idx, :]
        X = (X - mu_X) / sigma_X
        y = (y - mu_y) / sigma_y
        return X, y

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        keys = random.split(self.key, self.ensemble_size)
        inputs, targets = vmap(self.__data_generation, (0, 0, 0, 0))(
            keys, self.X_train, self.y_train, self.norm_const
        )
        return inputs, targets