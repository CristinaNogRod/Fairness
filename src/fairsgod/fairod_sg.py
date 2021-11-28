import pandas as pd
import tensorflow as tf
import numpy as np
from .losses import FairODLoss

from sklearn.metrics import roc_auc_score



class SGOutlierDetector():
    """
    Fairness-aware Outlier Detector.

    This outlier detector combines an autoencoder and a scoring network to detect anomalies in an unsupervised setting.
    It takes into account fairness by training the model using a custom loss (FairODLoss) composed of 3 terms.
    For details on FairODLoss function refer to the losses module.

    The aim of the scoring network is to learn the disparity of normal and abnormal data and 
    enhance this disparity during the network training. 

    Args:
        alpha (float): weight parameter for the first two terms of the FairODLoss function. As default, 1.0.
        gamma (float): weight parameter for the third term of the FairODLoss function. As default, 0.0.
        base_loss (function): loss function that computes the reconstruction errors. As default, MSE.
        epsilon (float): hyperparameter of scorer_loss function. As default, 0.015.
        a (float): hyperparameter of scorer_loss function. As default, 6.
        lambda_a (float): hyperparameter of scorer_loss function. As default, 1.0.
        lambda_se (float): weight parameter for the score-guided regularization term. As default, 0.9.
        optimizer (tf.keras.optimizer): As default, adam optimizer.
        model (tf.keras.model): Autoencoder model & Scoring network. Default is none and if this is the case, we build a simple
                                autoencoder and scoring network.
    """

    def __init__(self, 
                 alpha=0.1, 
                 gamma=0.5,
                 epsilon=0.015, a=6, 
                 lambda_a=1.0, 
                 lambda_se=0.9,
                 base_loss=None,
                 optimizer=None, 
                 embedding_dim=30,
                 ae=None, encoder=None, scorer=None
    ):
        """Constructor"""
        self.loss_fn = FairODLoss(base=base_loss, alpha=alpha, gamma=gamma)

        self.optimizer = tf.keras.optimizers.Adam() if not optimizer else optimizer
        self.ae = ae
        self.encoder = encoder
        self.scorer = scorer

        self.epsilon = epsilon
        self.a = a
        self.lambda_a = lambda_a
        self.lambda_se = lambda_se
        self.mu0 = 1e-5

        self.embedding_dim = embedding_dim

    def _build_models(self, n_inputs):
        """
        This function builds an autoencoder model and a scoring network when these are not provided.

        Args:
            n_inputs (int): number of features passed into the model.

        Returns:
            tf.keras.Model
        """
       # autoencoder
        inputs = tf.keras.Input(shape=(n_inputs,))
        encoder = tf.keras.layers.Dense(n_inputs, activation='relu', name='enc_l1')(inputs)
        embedding = tf.keras.layers.Dense(self.embedding_dim, activation='tanh', name='enc_embedding')(encoder)
        decoder = tf.keras.layers.Dense(80, activation='relu', name='dec_l1')(embedding)
        decoder = tf.keras.layers.Dense(n_inputs, activation='tanh', name='dec_out')(decoder)

        model = tf.keras.Model(inputs=inputs, outputs=decoder)
        encoder_model = tf.keras.Model(inputs=inputs, outputs=embedding)
        
        # scorer network
        inp_scorer = tf.keras.Input(shape=(self.embedding_dim,))
        scorer_layer = tf.keras.layers.Dense(self.embedding_dim, activation='relu', name='scorer_l1')(inp_scorer)
        scorer_layer = tf.keras.layers.Dense(30, activation='linear', name='scorer_l2')(scorer_layer)
        scorer_layer = tf.keras.layers.Dense(1, activation='linear', name='scorer_l3')(scorer_layer)

        scorer_model = tf.keras.Model(inputs=inp_scorer, outputs=scorer_layer)
        
        return model, encoder_model, scorer_model

    def fit(self, X, pv, batch_size=256, epochs=5, val_X=None, val_pv=None, stopping_after=None):
        """
        This function carries out the training of the networks.

        Args:
            X (np.array or pd.DataFrame): training dataset
            pv (np.array or pd.DataFrame): protected variables of training sataset
            batch_size (int)
            epochs (int)
            val_X (np.array or pd.DataFrame): validation dataset
            val_pv (np.array or pd.DataFrame): protected variables of validation dataset
            stopping_after (int): number of epochs during which, if the loss value continues to increase, then we stop the training

        Returns:
            2 np.arrays of length number of epochs, containing the training and validation losses.
            If the validation datasets are not provided, then the second array is empty.
        """
        # If models are not provided, build them
        if self.ae is None:
            self.ae, self.encoder, self.scorer = self._build_models(len(X.columns))

        # TODO: Added
        if type(pv) == tuple:
            y = pv[0]
            pv = pv[1]

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
            train_loss_results[epoch] = train_loss_avg.result()

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
            X_pred = self.ae(X, training=True)
            embeddings = self.encoder(X, training=True)
            scores = self.scorer(embeddings, training=True)

            recon_errors = tf.keras.losses.mse(X, X_pred)

            loss_value = self.loss_fn(X, X_pred, pv)
            loss_scoring = self.__scorer_loss(recon_errors, scores)

            loss_value = tf.cast(loss_value, tf.float32)
            loss_scoring = tf.cast(loss_scoring, tf.float32)

            loss_total = loss_value + self.lambda_se * tf.reduce_mean(loss_scoring)

        grads = tape.gradient(loss_total, sources = self.ae.trainable_weights + self.scorer.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.ae.trainable_weights + self.scorer.trainable_weights))

        # Update and return avg loss
        #train_loss_avg(loss_value)
        train_loss_avg(tf.reduce_mean(loss_value))
        return tf.reduce_mean(loss_value)#train_loss_avg.result()

    def __validation_step(self, X, pv, val_loss_avg):
        """
        Similar to the above function but using validation dataset.
        """
        X_pred = self.ae(X, training=True)
        embeddings = self.encoder(X, training=True)
        scores = self.scorer(embeddings, training=True)

        recon_errors = tf.keras.losses.mse(X, X_pred)

        loss_value = self.loss_fn(X, X_pred, pv)
        loss_scoring = self.__scorer_loss(recon_errors, scores)

        loss_value = tf.cast(loss_value, tf.float32)
        loss_scoring = tf.cast(loss_scoring, tf.float32)

        loss_total = tf.reduce_mean(loss_scoring)
        #loss_total = loss_value + self.lambda_se * tf.reduce_mean(loss_scoring)

        # Update and return avg loss
        #train_loss_avg(loss_value)
        val_loss_avg(loss_scoring)
        return val_loss_avg.result()

    def __scorer_loss(self, reconstruction_loss, score):
        """
        This function computes the score_guide regularization term.
        The aim is to force the distribution of the normal data points towards a mean of 0 and for anomalous data points towards mean of a.

        Args:
            reconstruction_loss (tf.Tensor): MSE obtained from autoencoder output.
            score (tf.Tensor): output of the scorer_loss function.
            mu0 (float): hyperparameter of scorer_loss function. As default, 1e-5.
                         Model is not sensitive to this parameter so can be left as the default.
        
        Returns:
            
        """
        mask = tf.cast(reconstruction_loss < self.epsilon, dtype=tf.float64)
        losses = tf.zeros_like(reconstruction_loss,  dtype=tf.float64)
        score = tf.squeeze(tf.cast(score, dtype=tf.float64))

        losses += tf.squeeze(tf.cast(tf.abs(score - self.mu0), dtype=tf.float64)) * mask
        losses += tf.squeeze(tf.math.maximum(0.0, self.a - score)) * (1 - mask) * self.lambda_a

        return losses

    def _predict(self, X):
        """
        This function does a predict of sample passed as argument, ie it attempts to reconstruct samples.
        """
        return self.ae.predict(X)

    def predict_scores(self, X):
        """
        This function computes the reconstruction errors (or outlier scores), that is, the MSE per sample.
        """
        embed = self.encoder.predict(X)
        return self.scorer(embed)
        #X_pred = self._predict(X)
        #return tf.reduce_mean((X - X_pred) ** 2, axis=1)

    def predict_outliers(self, X, threshold=10):
        """
        This function classifies the samples into the outlier or inlier group. This is done using their
        resconstruction error and the outlier threshold set when instanciating the class.

        Args:
            X (np.array or pd.DataFrame): dataset to carry out outlier detection

        Returns:
            tf.Tensor: binary tensor where the samples with 1 are the classified outliers
        """
        outlier_scores = self.predict_scores(X)
        outliers = outlier_scores >= threshold
        return tf.cast(outliers, tf.int64)

    def get_params(self, *args, **kwargs):
        return {
            'epsilon':self.epsilon,
            'lambda_se':self.lambda_se,
            'a':self.a,
            'lambda_a':self.lambda_a,
            'alpha':self.loss_fn.alpha,
            'gamma':self.loss_fn.gamma
        }

    def set_params(self, **params):
        self.epsilon=params['epsilon'] if 'epsilon' in params else self.epsilon
        self.lambda_se=params['lambda_se']
        self.a=params['a']
        self.lambda_a=params['lambda_a']
        self.loss_fn = FairODLoss(base=None, 
                                  alpha=params['alpha'] if 'alpha' in params else self.loss_fn.alpha, 
                                  gamma=params['gamma'] if 'gamma' in params else self.loss_fn.gamma
        )
        return self

    def score(self, X, y):
        if type(y) == tuple:
            pv = y[1]
            y = y[0]
        ypred = self.predict_scores(X).numpy()
        return roc_auc_score(y, ypred)