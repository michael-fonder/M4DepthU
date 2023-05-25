import tensorflow as tf

def masked_reduce_mean(err, gt_depth, mask=None):
    if mask is None:
        mask = tf.cast(tf.greater(gt_depth, 1e-6), tf.float32)
    return tf.reduce_sum(tf.math.multiply_no_nan(err,mask))/tf.maximum(tf.reduce_sum(mask),1)


class RootMeanSquaredError(tf.keras.metrics.Mean):

    def __init__(self, name='RMSE', **kwargs):
        super(RootMeanSquaredError, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.square(y_true-y_pred)
        error = tf.sqrt(masked_reduce_mean(error, y_true))
        return super(RootMeanSquaredError, self).update_state(error)

class RootMeanSquaredLogError(tf.keras.metrics.Mean):

    def __init__(self, name='RMSE_log', **kwargs):
        super(RootMeanSquaredLogError, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.cast(tf.greater(y_true, 0.), tf.float32)
        y_true = tf.math.log(y_true+1e-6)
        y_pred = tf.math.log(y_pred+1e-6)
        error = tf.square(y_true-y_pred)
        error = tf.sqrt(masked_reduce_mean(error, y_true))

        return super(RootMeanSquaredLogError, self).update_state(error)

class AbsRelError(tf.keras.metrics.Mean):

    def __init__(self, name='AbsRel', **kwargs):
        super(AbsRelError, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.math.abs(y_true - y_pred) / (y_true+1e-6)
        error = masked_reduce_mean(error, y_true)
        return super(AbsRelError, self).update_state(error)


class SqRelError(tf.keras.metrics.Mean):

    def __init__(self, name='SqRel', **kwargs):
        super(SqRelError, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.math.squared_difference(y_true, y_pred) / (y_true+1e-6)
        error = masked_reduce_mean(error, y_true)
        return super(SqRelError, self).update_state(error)


class ThresholdRelError(tf.keras.metrics.Mean):

    def __init__(self, threshold, name='Delta', **kwargs):
        self.threshold = threshold
        super(ThresholdRelError, self).__init__(name=name + str(threshold), **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        thresh = tf.maximum((y_true / y_pred), (y_pred / y_true))
        error = tf.cast(tf.math.less(thresh, 1.25 ** self.threshold), tf.float32)
        error = masked_reduce_mean(error, y_true)
        return super(ThresholdRelError, self).update_state(error)

class AuSE(tf.keras.metrics.Metric):
    '''
        Parent class for AuSE performance metrics.
        This code is designed to compute the AuSE over the whole dataset.
    '''
    def __init__(self, name='AuSE', direction='DESCENDING', **kwargs):
        super(AuSE, self).__init__(name=name, **kwargs)
        self.uncert_array = tf.Variable(initial_value=tf.zeros([0]), trainable=False, validate_shape=False, name="uncert_array",
                                        dtype=tf.float32, shape = tf.TensorShape(None))
        self.error_array = tf.Variable(initial_value=tf.zeros([0]), trainable=False, validate_shape=False, name="error_array",
                                        dtype=tf.float32, shape = tf.TensorShape(None))
        self.direction = direction

    def get_perf(self, y_true, y_pred):
        ''' Needs to be implemented for the desired performance metric '''
        return 0

    def post_mean(self, value):
        return value

    def reset_states(self):
        self.uncert_array.assign(tf.zeros([0]))
        self.error_array.assign(tf.zeros([0]))

    def update_state(self, y_true, y_pred, uncert, mask=None):
        if mask is None:
            mask = tf.ones_like(y_true)

        element_to_keep = tf.where(tf.reshape(tf.cast(mask, tf.bool), [-1]))

        def get_elements(map, indices):
            return tf.gather(tf.reshape(map, [-1]), indices)[:, 0]

        perf = self.get_perf(y_true, y_pred)
        tmp = tf.concat([self.error_array, get_elements(perf, element_to_keep)], axis=0)
        self.error_array.assign(tmp)

        tmp = tf.concat([self.uncert_array, get_elements(uncert, element_to_keep)], axis=0)
        self.uncert_array.assign(tmp)

    def result(self):
        with tf.device("/cpu:0"):
            nbre_bins = 999
            sparsification_stop = int(0.02*nbre_bins) # stop sparsification once 2 percent of the samples are remaining
            def get_sparsification_curve(sorted_vector):
                remainder = tf.shape(self.error_array)[0] % nbre_bins
                results = [tf.reduce_mean(sorted_vector)]

                bins = tf.stack(tf.split(sorted_vector[remainder:], nbre_bins), axis=0)
                bins = tf.reduce_mean(bins, axis=[1])

                for i in range(nbre_bins-sparsification_stop):
                    results.append(self.post_mean(tf.reduce_mean(bins[i:])))

                return tf.stack(results)

            oracle_sort = tf.argsort(self.error_array, direction=self.direction)
            oracle_sorted = tf.gather(tf.identity(self.error_array), oracle_sort)
            oracle_sparsification = get_sparsification_curve(oracle_sorted)

            uncert_sort = tf.argsort(self.uncert_array, direction='DESCENDING')
            uncert_sorted = tf.gather(tf.identity(self.error_array), uncert_sort)
            uncert_sparsification = get_sparsification_curve(uncert_sorted)

            ause = tf.reduce_mean(tf.abs(uncert_sparsification-oracle_sparsification))

        return ause*(float(nbre_bins-sparsification_stop)/float(nbre_bins))


class AuSEAbsRelError(AuSE):

    def __init__(self, name='AuSE_AbsRel', **kwargs):
        super(AuSEAbsRelError, self).__init__(name=name, **kwargs)

    def get_perf(self, y_true, y_pred):
        return tf.math.abs(y_true - y_pred) / (y_true+1e-6)

class AuSERootMeanSquaredLogError(AuSE):

    def __init__(self, name='AuSE_RMSE_log', **kwargs):
        super(AuSERootMeanSquaredLogError, self).__init__(name=name, **kwargs)

    def get_perf(self, y_true, y_pred):
        y_true_log = tf.math.log(y_true+1e-6)
        y_pred_log = tf.math.log(y_pred+1e-6)
        return tf.square(y_true_log-y_pred_log)

    def post_mean(self, value):
        return tf.sqrt(value)

class AuSEThresholdRelError(AuSE):

    def __init__(self, threshold, name='AuSE_Delta', **kwargs):
        super(AuSEThresholdRelError, self).__init__(name=name, direction='ASCENDING', **kwargs)
        self.threshold = threshold

    def get_perf(self, y_true, y_pred):
        thresh = tf.maximum((y_true / y_pred), (y_pred / y_true))
        return tf.cast(tf.math.less(thresh, 1.25 ** self.threshold), tf.float32)