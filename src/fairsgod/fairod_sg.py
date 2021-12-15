import pandas as pd
import tensorflow as tf
import numpy as np
from .losses import FairODLoss,  _StatisticalParity_Loss, _GroupFidelity_Loss

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
                 epsilon_p=90, 
                 a=6, 
                 lambda_a=1.0, 
                 lambda_se=0.9,
                 base_loss=None,
                 optimizer=None, 
                 embedding_dim=None,
                 ae=None, encoder=None, scorer=None
    ):
        """Constructor"""
        self.spl = _StatisticalParity_Loss()
        self.gfl = _GroupFidelity_Loss()
        self.alpha = alpha
        self.gamma = gamma


        self.optimizer = None
        self.ae = ae
        self.encoder = encoder
        self.scorer = scorer

        self.epsilon = None
        self.epsilon_p = epsilon_p
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
        #initializer = tf.keras.initializers.Constant(value=.01)
        initializer = tf.keras.initializers.RandomNormal()

        if self.embedding_dim is None:
            if n_inputs < 40:
                self.embedding_dim=2
            if 40 <= n_inputs < 100:
                self.embedding_dim=5
            if n_inputs >= 100:
                self.embedding_dim=10
        
        if n_inputs < 40:
            layers = [10, self.embedding_dim]
        if 40 <= n_inputs < 100:
            layers = [40, 20, self.embedding_dim]
        if n_inputs >= 100:
            layers = [40, 20, 10, self.embedding_dim]
    

       # autoencoder
        inputs = tf.keras.Input(shape=(n_inputs,))
        encoder = inputs
        for i, ln in enumerate(layers[:-1]):
            encoder = tf.keras.layers.Dense(ln, activation='relu', name=f'enc_l{i}', kernel_initializer=initializer)(encoder) 
        embedding = tf.keras.layers.Dense(layers[-1], activation='linear', name='enc_embedding', kernel_initializer=initializer)(encoder)

        dec_inp = tf.keras.Input(shape=(layers[-1]))
        decoder = dec_inp
        for i, ln in enumerate(layers[:0:-1]):
            decoder = tf.keras.layers.Dense(20, activation='relu', name=f'dec_{i}', kernel_initializer=initializer)(dec_inp)
        
        decoder = tf.keras.layers.Dense(n_inputs, activation='linear', name='dec_out', kernel_initializer=initializer)(decoder)

        encoder_model = tf.keras.Model(inputs=inputs, outputs=embedding)
        decoder_model = tf.keras.Model(inputs=dec_inp, outputs=decoder)

        outs = decoder_model(encoder_model.output)
        model = tf.keras.Model(inputs=inputs, outputs=outs)

        # scorer network
        inp_scorer = tf.keras.Input(shape=(layers[-1]))
        scorer_layer = tf.keras.layers.Dense(10, activation='linear', name='scorer_l2', kernel_initializer=initializer)(inp_scorer)
        scorer_layer = tf.keras.layers.Dropout(.2)(scorer_layer)
        scorer_layer = tf.keras.layers.Dense(1, activation='linear', name='scorer_l3', kernel_initializer=initializer)(scorer_layer)

        scorer_model = tf.keras.Model(inputs=inp_scorer, outputs=scorer_layer)
        
        return model, encoder_model, decoder_model, scorer_model

    def fit(self, X, pv, batch_size=256, epochs=5, val_X=None, val_pv=None, early_stop=None, verbose=True):
        # If models are not provided, build them
        if self.ae is None:
            self.ae, self.encoder, self.decoder, self.scorer = self._build_models(len(X.columns))

        self._training_finished = False

        if type(pv) == tuple:
            y = pv[0]
            pv = pv[1]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

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
            if self._training_finished:
                break

            if verbose:
                print(f"EPOCH {epoch}")

            # Reset loss avg at each epoch
            train_loss_avg = tf.keras.metrics.Mean()

            # Reset used epsilon after each epoch
            self.epsilon = None

            # Iterate over batches
            for step, (X_batch, pv_batch) in enumerate(train_dataset):
                loss_value = self.__training_step(X_batch, pv_batch, train_loss_avg, verbose=verbose)

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

            if early_stop is not None:
                outs = []
                for step, (X_batch, pv_batch) in enumerate(train_dataset):
                    scores = self.predict_scores(X_batch).numpy().tolist()
                    outs.extend(scores)
                
                scores_mean = np.mean(scores)
                if scores_mean > self.a / 2:
                    print(f"Early stopping... (Scores mean: {scores_mean})")
                    self._training_finished = True

        return train_loss_results, val_loss_results

    def __training_step(self, X, pv, train_loss_avg, verbose=False):
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
            embeddings = self.encoder(X, training=True)
            X_pred = self.decoder(embeddings)
            scores = self.scorer(embeddings, training=True)

            recon_errors = tf.keras.losses.mse(X, X_pred)

            l2_error = tf.cast(
                #InvalidArgumentError: cannot compute Sub as input #1(zero-based) was expected to be a int64 tensor but is a double tensor [Op:Sub]
                 tf.norm(X - tf.cast(X_pred, tf.float64), axis=1), 
                 tf.float32
            )

            # nasty
            if self.epsilon is None:
                iepsilon = np.percentile(l2_error.numpy(), self.epsilon_p)
                self.epsilon = iepsilon

            loss_scoring = self.__scorer_loss(l2_error, scores)

            loss_scoring = tf.cast(loss_scoring, tf.float32)

            # FairOD Terms
            sx = scores
            spl_term = self.spl(sx, pv)
            gfl_term = self.gfl(sx, pv)

            #self.lambda_se = 1
            loss_total =  recon_errors + self.lambda_se * tf.reduce_mean(loss_scoring)

            if verbose:
                print("  LOSS_TOTAL {} - LOSS_REC {} - LOSS_SCORING {} - EPSILON NORM {}".format(
                    tf.reduce_mean(loss_total).numpy(),
                    tf.reduce_mean(recon_errors).numpy(),
                    tf.reduce_mean(loss_scoring).numpy(),
                    tf.reduce_mean(self.epsilon).numpy(),
                ))

            # Fair Extension:
            loss_total = self.alpha * loss_total + (1 - self.alpha) * spl_term + self.gamma * gfl_term


        grads = tape.gradient(loss_total, sources=self.encoder.trainable_weights + self.decoder.trainable_weights + self.scorer.trainable_weights)
        grads = [(tf.clip_by_value(grad, -.5, .5)) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights + self.decoder.trainable_weights + self.scorer.trainable_weights))
        

        # Update and return avg loss
        train_loss_avg(loss_total)
        return tf.reduce_mean(loss_total)

    def __validation_step(self, X, pv, val_loss_avg):
        """
        Similar to the above function but using validation dataset.
        """
        X_pred = self.ae(X, training=True)
        embeddings = self.encoder(X, training=True)
        scores = self.scorer(embeddings, training=True)

        recon_errors = tf.keras.losses.mse(X, X_pred)

        #loss_value = self.loss_fn(X, X_pred, pv)
        loss_scoring = self.__scorer_loss(recon_errors, scores)

        #loss_value = tf.cast(loss_value, tf.float32)
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

        ref = tf.random.normal((10000,), mean=0, stddev=1)
        mu0 = tf.cast(tf.reduce_mean(ref), dtype=tf.float64)
        
        inlier_loss = tf.cast(
            tf.abs(score - mu0), 
            dtype=tf.float64
        )

        outlier_loss = tf.math.maximum(0.0, self.a - score)

        losses += inlier_loss * (mask)
        losses += outlier_loss * (1-mask) * self.lambda_a

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

    def predict_outliers(self, X, threshold=None):
        """
        This function classifies the samples into the outlier or inlier group. This is done using their
        resconstruction error and the outlier threshold set when instanciating the class.

        Args:
            X (np.array or pd.DataFrame): dataset to carry out outlier detection

        Returns:
            tf.Tensor: binary tensor where the samples with 1 are the classified outliers
        """
        outlier_scores = self.predict_scores(X)
        outliers = outlier_scores >= self.epsilon
        return tf.cast(outliers, tf.int64)

    def get_params(self, *args, **kwargs):
        return {
            'epsilon_p':self.epsilon_p,
            'lambda_se':self.lambda_se,
            'a':self.a,
            'lambda_a':self.lambda_a,
            'alpha':self.alpha,
            'gamma':self.gamma
        }

    def set_params(self, **params):
        self.epsilon=params['epsilon_p'] if 'epsilon_p' in params else self.epsilon_p
        self.lambda_se=params['lambda_se']
        self.a=params['a']
        self.lambda_a=params['lambda_a'],
        self.alpha=params['alpha'],
        self.alpha=params['gamma']
        return self

    def score(self, X, y):
        if type(y) == tuple:
            pv = y[1]
            y = y[0]
        ypred = self.predict_scores(X).numpy()
        return roc_auc_score(y, ypred)