import pandas as pd
import tensorflow as tf
if tf.__version__ < '2.0':
    import warnings  # noqa
    warnings.warn('OutlierDetector requires tensorflow>=2.X, %s was found.\n\nTry:\npip install tensorflow>=2.0'
                  % tf.__version__, UserWarning)
import numpy as np
from .losses import FairODLoss


class OutlierDetector():
    """
    Fairness-aware Outlier Detector.
    This outlier detector trains an autoencoder to detect outliers in an unsupervised setting.
    It takes into account fairness by training the model using a custom loss (FairODLoss) composed of 3 terms.
    For details of this function refer to the losses module.

    Args:
        alpha (float): weight parameter for the first two terms of the FairODLoss function. As default, 1.0.
        gamma (float): weight parameter for the third term of the FairODLoss function. As default, 0.0.
        base_loss (function): loss function that computes the reconstruction errors. As default, MSE.
        optimizer (tf.keras.optimizer): As default, adam optimizer.
        model (tf.keras.model): Autoencoder model. Default is none and if this is the case, we build a simple
                                autoencoder of 3 dense layers.
        threshold (double): the value to set the outlier threshold at. Default, none.
                            It will be used to classify samples as outliers or not.

    An example can be found in the notebook tutorial FairOD_Example.
    """

    def __init__(self, alpha=1.0, gamma=0.0,
                 base_loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
                 optimizer=tf.keras.optimizers.Adam(), model=None,
                 threshold=None):

        self.loss_fn = FairODLoss(base=base_loss, alpha=alpha, gamma=gamma)
        self.optimizer = optimizer
        self.model = model
        self.threshold = threshold

    def __build_autoencoder(self, n_inputs):
        """
        This function builds a simple autoencoder model when one is not provided.

        Args:
            n_inputs (int): number of features passed into the model.

        Returns:
            tf.keras.Model
        """
        # The paper suggests the following embedding dimensions dependent on number of input features
        n_hidden = 8
        if n_inputs < 100:
            n_hidden = 2

        inputs = tf.keras.Input(shape=(n_inputs,))
        encoder = tf.keras.layers.Dense(n_inputs)(inputs)
        embedding = tf.keras.layers.Dense(n_hidden)(encoder)
        decoder = tf.keras.layers.Dense(n_inputs, activation='tanh')(embedding)

        model = tf.keras.Model(inputs=inputs, outputs=decoder)

        return model

    def fit(self, X, pv, batch_size=512, epochs=5, val_X=None, val_pv=None, stopping_after=None):
        """
        This function carries out the training of the model.

        Args:
            X (np.array or pd.DataFrame): training dataset
            pv (np.array or pd.DataFrame): protected variables of training sataset
            batch_size (int)
            epochs (int)
            val_X (np.array or pd.DataFrame): validation dataset
            val_pv (np.array or pd.DataFrame): protected variables of validation dataset
            stopping_after (int): number of epochs during which, if the loss value continues to increse, then we stop the training

        Returns:
            2 np.arrays of length number of epochs, containing the training and validation losses.
            If the validation datasets are not provided, then the second array is empty.
        """
        # If no model was provided, build a simple autoencoder
        if self.model is None:
            self.model = self.__build_autoencoder(len(X.columns))

        # Compile model
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        # Prepare training datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X, pv))
        train_dataset = train_dataset.batch(batch_size)

        # Initialise array to store loss during training
        train_loss_results = np.zeros(shape=(epochs,), dtype=np.float32)

        # If we have a validation dataset, do the above steps again
        val_loss_results = None
        if val_X is not None:
            # Prepare datasets
            val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_pv))
            val_dataset = val_dataset.batch(batch_size)
            # Initialise array to store loss during validation
            val_loss_results = np.zeros(shape=(epochs,), dtype=np.float32)

        # Training
        counter = 0
        for epoch in range(epochs):
            # Reset loss avg at each epoch
            train_loss_avg = tf.keras.metrics.Mean()

            # Iterate over batches
            for step, (X_batch, pv_batch) in enumerate(train_dataset):
                loss_value = self.__training_step(X_batch, pv_batch, train_loss_avg)

            # Save the training loss
            train_loss_results[epoch] = loss_value

            # Iterate over validation batches if the validation dataset was provided
            if val_X is not None:
                # Reset validation loss avg at the end of epoch
                val_loss_avg = tf.keras.metrics.Mean()

                # Run a validation loop at the end of each epoch
                for X_batch_val, pv_batch_val in val_dataset:
                    val_loss_value = self.__validation_step(X_batch_val, pv_batch_val, val_loss_avg)

                # Save validation loss
                val_loss_results[epoch] = val_loss_value

            # Early stopping callback:
            # we stop the training if the loss value increses during x=stopping_after epochs
            if stopping_after:
                if counter > stopping_after:
                    break
                if loss_value > train_loss_results[epoch - 1]:
                    counter += 1
                else:
                    counter = 0

        return train_loss_results, val_loss_results

    def __training_step(self, X, pv, train_loss_avg):
        """
        This function carries out 1 training step in the fit method.

        Args:
            X (np.array or pd.DataFrame): training dataset of size batch_size
            pv (np.array or pd.DataFrame): protected variables also size batch_size
            train_loss_avg (tf.keras.metrics.Mean): function used to compute average loss per training batch

        Returns:
            loss value of training step
        """
        with tf.GradientTape() as tape:
            X_pred = self.model(X, training=True)
            loss_value = self.loss_fn(X, X_pred, pv)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # Update and return avg loss
        train_loss_avg(loss_value)
        return train_loss_avg.result()

    def __validation_step(self, X, pv, val_loss_avg):
        """
        Similar to the above function but using validation dataset.
        """
        X_pred = self.model(X, training=False)
        test_loss_value = self.loss_fn(X, X_pred, pv)
        # Update and return avg loss
        val_loss_avg(test_loss_value)
        return val_loss_avg.result()

    def _predict(self, X):
        """
        This function does a predict of sample passed as argument, ie it attempts to reconstruct samples.
        """
        return self.model.predict(X)

    def predict_scores(self, X):
        """
        This function computes the reconstruction errors (or outlier scores), that is, the MSE per sample.
        """
        X_pred = self._predict(X)
        return tf.reduce_mean((X - X_pred) ** 2, axis=1)

    def predict_outliers(self, X):
        """
        This function classifies the samples into the outlier or inlier group. This is done using their
        resconstruction error and the outlier threshold set when instanciating the class.

        Args:
            X (np.array or pd.DataFrame): dataset to carry out outlier detection

        Returns:
            tf.Tensor: binary tensor where the samples with 1 are the classified outliers
        """
        outlier_scores = self.predict_scores(X)

        # If no threshold was passed, then find the 90th percentile
        if self.threshold is None:
            self.threshold = np.percentile(outlier_scores.numpy(), 90)

        outliers = outlier_scores >= self.threshold
        return tf.cast(outliers, tf.int64)
