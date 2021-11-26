import tensorflow as tf



class FairODLoss(tf.keras.losses.Loss):
    """
    Computes the weighted sum of the base, statistical parity and group fidelity loss terms.

    This function can be used in fairness-aware outlier detectors. The aim is to include fairness criteria during training.
    The first term (by default the MSE) computes the reconstruction error of the samples.
    The second term (statistical parity) aims to minimise the correlation between the reconstruction error and a protected variable (PV).
    The last term (group fidelity) aims to maintain the rankings of the base model.
    (The base model only uses the first term in its training loss function).

    Args:
        base (function): loss function that computes what we call reconstruction errors.
        alpha (float): weighting parameter to adjust importance of MSE and statistical parity loss terms.
        gamma (float): weighting parameter to adjust importance of group fidelity loss term.

    Return:
        tf.Tensor: A tf.Tensor usable as a loss in tensorflow.
    """

    def __init__(self, base=None, alpha=1, gamma=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.base = base if base else tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.spl = _StatisticalParity_Loss()
        self.gfl = _GroupFidelity_Loss()

    def __call__(self, y_true, y_pred, pv):
        y_true = tf.cast(y_true, tf.float32)
        sx = self.base(y_true, y_pred)
        return self.alpha * tf.reduce_sum(sx) + (1 - self.alpha) * self.spl(sx, pv) + self.gamma * self.gfl(sx, pv)


class _StatisticalParity_Loss(tf.keras.losses.Loss):
    """
    Statistical Parity Loss.

    This loss aims to minimise the absolute correlation between the reconstruction errors (or outlier scores)
        and the PV (Protected Variable).
    Thus, the outlier score is independent of the value of the PV.

    Returns:
        tf.Tensor: A tf.Tensor usable as a loss in tensorflow.

    (Note the PV is passed as the y_pred argument and the reconstruction error as the y_true, as in usual Keras losses.)
    """

    def __init__(self):
        super().__init__()

    def call(self, sx, pv):
        mu_s = tf.math.reduce_mean(sx)
        sum_s = (sx - mu_s)
        var_s = tf.math.reduce_sum((sx - mu_s) ** 2)

        pv = tf.cast(pv, tf.float32)
        mu_pv = tf.math.reduce_mean(pv)
        sum_pv = (pv - mu_pv)
        var_pv = tf.math.reduce_sum((pv - mu_pv) ** 2)

        denom = (tf.math.sqrt(var_s * var_pv))
        # minimising 1.+ sqrt( product of standard deviations ) is the same as minimising just the product
        if denom == 0:
            denom = denom + 1.

        return tf.math.abs((tf.math.reduce_sum(sum_s * sum_pv)) / denom)


class _GroupFidelity_Loss(tf.keras.losses.Loss):
    """
    Group Fidelity Loss.

    This loss uses a listwise ranking loss criterion based on the Discounted Cumulative Gain (DCG) measure.
    It ensures that the rankings within the groups are maintained with respect to base's model ranking.
    The base's model ranking is the ranking of the samples based on their reconstruction error.

    Returns:
        tf.Tensor: A tf.Tensor usable as a loss in tensorflow.
    """

    def __init__(self):
        super().__init__()

    def call(self, sx, pv):
        sx_maj = sx[pv == 0]
        sx_min = sx[pv == 1]

        # The next few lines avoid computing the loss for empty sx_min or sx_maj
        term_min = 0.0
        term_maj = 0.0
        if tf.reduce_sum(sx_min) > 0:
            term_min = self.compute_ndcg(sx_min)
        if tf.reduce_sum(sx_maj) > 0:
            term_maj = self.compute_ndcg(sx_maj)

        return term_maj + term_min

    def compute_idcg(self, sx):
        """
        This function computes the normalisation term for the NDCG term in compute_sum().

        Args:
            sx (tf.Tensor): reconstruction errors or outlier scores tensor

        Return:
            tf.Tensor: used in when compute_sum() is called.
        """
        j = tf.linspace(1., len(sx), len(sx))
        return tf.math.reduce_sum((2 ** sx - 1) / tf.experimental.numpy.log2(1 + j))

    def compute_ndcg(self, sx):
        """
        This function computes a modified version of the NDCG (Normalised Discounted Cumulative Gains).

        Args:
            sx (tf.Tensor): reconstruction errors or outlier scores tensor

        Return:
            tf.Tensor: used in when GroupFidelity_loss() is called.
        """
        denom_sum = tf.math.reduce_sum(tf.math.sigmoid(sx - tf.expand_dims(sx, axis=1)), axis=1)
        log_term = tf.experimental.numpy.log2(1 + denom_sum)

        NDCG = tf.math.reduce_sum((2 ** sx - 1) / (log_term * self.compute_idcg(sx)))
        return 1 - NDCG
