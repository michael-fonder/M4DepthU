"""
----------------------------------------------------------------------------------------
Copyright (c) 2023 - Michael Fonder, University of Liège (ULiège), Belgium.

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License along with this
program. If not, see < [ https://www.gnu.org/licenses/ | https://www.gnu.org/licenses/ ] >.
----------------------------------------------------------------------------------------
"""

import tensorflow as tf
from tensorflow import keras as ks
from utils.depth_operations import *
from collections import namedtuple

M4depthUAblationParameters = namedtuple('M4depthUAblationParameters', ('DINL', 'SNCV', 'time_recurr', 'normalize_features', 'subdivide_features', 'level_memory', 'uncertainty_head_layers', 'uncertainty'),
                                        defaults=(True, True, True, True, True, True, 0, 'relative'))

M4depthULossParameters = namedtuple('M4depthULossParameters', ('lh_weight', 'uncertainty_weight'),
                                    defaults=(0.05, 1.))

class DomainNormalization(ks.layers.Layer):
    # Normalizes a feature map according to the procedure presented by
    # Zhang et.al. in "Domain-invariant stereo matching networks".

    def __init__(self, regularizer_weight=0.0004):
        super(DomainNormalization, self).__init__()
        self.regularizer_weight = regularizer_weight

    def build(self, input_shape):
        channels = input_shape[-1]

        self.scale = self.add_weight(name="scale", shape=[1, 1, 1, channels], dtype='float32',
                                     initializer=tf.ones_initializer(), trainable=True)
        self.bias = self.add_weight(name="bias", shape=[1, 1, 1, channels], dtype='float32',
                                    initializer=tf.zeros_initializer(), trainable=True)

        # Add regularization loss on the scale factor
        regularizer = tf.keras.regularizers.L2(self.regularizer_weight)
        self.add_loss(regularizer(self.scale))

    def call(self, f_map):
        mean = tf.math.reduce_mean(f_map, axis=[1, 2], keepdims=True, name=None)
        var = tf.math.reduce_variance(f_map, axis=[1, 2], keepdims=True, name=None)
        normed = tf.math.l2_normalize((f_map - mean) / (var + 1e-12), axis=-1)
        return self.scale * normed + self.bias


class FeaturePyramid(ks.layers.Layer):
    # Encoder of the network
    # Builds a pyramid of feature maps.

    def __init__(self, settings, regularizer_weight=0.0004, trainable=True):
        super(FeaturePyramid, self).__init__(trainable=trainable)

        self.use_dinl = settings["ablation"].DINL
        self.out_sizes = [16, 32, 64, 96, 128, 192][:settings["nbre_lvls"]]

        init = ks.initializers.HeNormal()
        reg = ks.regularizers.L1(l1=regularizer_weight)
        self.conv_layers_s1 = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in self.out_sizes
        ]
        self.conv_layers_s2 = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(2, 2), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in self.out_sizes
        ]

        self.dn_layers = [DomainNormalization(regularizer_weight=regularizer_weight) for nbre_filters in self.out_sizes]

    @tf.function  # (jit_compile=True)
    def call(self, images):
        feature_maps = images
        outputs = []
        for i, (conv_s1, conv_s2, dn_layer) in enumerate(zip(self.conv_layers_s1, self.conv_layers_s2, self.dn_layers)):
            tmp = conv_s1(feature_maps)
            if self.use_dinl and i == 0:
                tmp = dn_layer(tmp)
            tmp = tf.nn.leaky_relu(tmp, 0.1)

            tmp = conv_s2(tmp)
            feature_maps = tf.nn.leaky_relu(tmp, 0.1)
            outputs.append(feature_maps)

        return outputs


class DispRefiner(ks.layers.Layer):
    # Sub-network in charge of refining an input parallax estimate

    def __init__(self, settings, regularizer_weight=0.0004):
        super(DispRefiner, self).__init__()

        init = ks.initializers.HeNormal()
        reg = ks.regularizers.L1(l1=regularizer_weight)

        local_head_cnt = settings.uncertainty_head_layers+1
        head_conv_channels = [128, 128, 96, 64, 32, 16]

        p_conv_channels = (head_conv_channels + [5])[-local_head_cnt:]   # Layers allocated to the parallax head
        u_conv_channels = (head_conv_channels + [1])[-local_head_cnt:]   # Layers allocated to the uncertainty head
        pre_conv_channels = (head_conv_channels + [1])[:-local_head_cnt] # Layers shared between the two heads

        # Initialize layers for the parallax refiner subnetwork

        self.prep_conv_layers = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in pre_conv_channels
        ]

        self.est_d_conv_layers = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in p_conv_channels
        ]

        self.est_a_conv_layers = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in u_conv_channels
        ]

    @tf.function
    def call(self, feature_map):

        prev_out = tf.identity(feature_map)

        # Common branch of the parallax refiner subnetwork
        for i, conv in enumerate(self.prep_conv_layers):
            prev_out = conv(prev_out)
            prev_out = tf.nn.leaky_relu(prev_out, 0.1)

        # Separate the refining for the two distinct heads
        prev_outs = [prev_out, prev_out]

        for i, convs in enumerate(zip(self.est_d_conv_layers, self.est_a_conv_layers)):
            for j, (prev, conv) in enumerate(zip(prev_outs, convs)):
                prev_outs[j] = conv(prev)

                if i < len(self.est_d_conv_layers) - 1:  # Don't activate last convolution output
                    prev_outs[j] = tf.nn.leaky_relu(prev_outs[j], 0.1)

        return {"para": prev_outs[0][:,:,:,:1], "uncertainty_para": prev_outs[1], "mem": prev_outs[0][:,:,:,1:]}


class DepthEstimatorLevel(ks.layers.Layer):
    # Stackable level for the decoder of the architecture
    # Outputs both a depth and a parallax map

    # IMPORTANT note: parallax values are encoded and estimated in the log space. Since the difference between the parallax
    # and its inverse is simply a change of sign in the log space, estimating the parallax or its inverse is the same task for the network.
    # Therefore, for code simplicity, we always ask the network to estimate parallax, even when it should theoretically
    # infer the inverse parallax, such as required for the probabilitic uncertainty.

    def __init__(self, settings, depth, regularizer_weight=0.0004):
        super(DepthEstimatorLevel, self).__init__()

        self.is_training = settings["is_training"]
        self.ablation = settings["ablation"]

        self.disp_refiner = DispRefiner(self.ablation, regularizer_weight=regularizer_weight)
        self.init = True
        self.lvl_depth = depth
        self.lvl_mul = depth-3

    def build(self, input_shapes):
        # Init. variables required to store the state of the level between two time steps when working in an online fashion
        self.shape = input_shapes["curr_f_maps"]

        f_maps_init = tf.zeros_initializer()
        d_maps_init = tf.ones_initializer()
        if (not self.is_training):
            self.prev_f_maps = self.add_weight(name="prev_f_maps", shape=self.shape, dtype='float32',
                                               initializer=f_maps_init, trainable=False, use_resource=False)
            self.depth_prev_t = self.add_weight(name="depth_prev_t", shape=self.shape[:3] + [1], dtype='float32',
                                                initializer=d_maps_init, trainable=False, use_resource=False)
            self.lvl_uncertainty_depth = self.add_weight(name="uncertainty_depth", shape=self.shape[:3] + [1], dtype='float32',
                                                initializer=d_maps_init, trainable=False, use_resource=False)
        else:
            print("Skipping temporal memory instanciation")

    @tf.function
    def call(self, inputs):
        # Deserialization of inputs
        editable_inputs = inputs.copy()
        expected_vars = ["curr_f_maps", "prev_l_est", "rot", "trans", "camera", "new_traj", "prev_f_maps", "prev_t_data"]
        for var in expected_vars:
            if not var in inputs:
                editable_inputs[var] = None

        curr_f_maps = editable_inputs["curr_f_maps"]
        prev_l_est  = editable_inputs["prev_l_est"]
        rot         = editable_inputs["rot"]
        trans       = editable_inputs["trans"]
        camera      = editable_inputs["camera"]
        new_traj    = editable_inputs["new_traj"]
        prev_f_maps = editable_inputs["prev_f_maps"]
        prev_t_data = editable_inputs["prev_t_data"]

        with tf.name_scope("DepthEstimator_lvl"):
            b, h, w, c = self.shape

            # Set dictionnary key to use for depth-related uncertainty
            if self.ablation.uncertainty == "relative":
                uncert_depth_choice = "rel_uncertainty_depth"
            else:
                uncert_depth_choice = "uncertainty_depth"

            # Disable feature vector subdivision if required
            if self.ablation.subdivide_features:
                nbre_cuts = 2**(self.lvl_depth//2)
            else:
                nbre_cuts = 1

            # Disable feature vector normalization if required
            if self.ablation.normalize_features:
                vector_processing = lambda f_map : tf.linalg.normalize(f_map, axis=-1)[0]
            else:
                vector_processing = lambda f_map : f_map

            # Preparation of the feature maps for to cost volumes
            curr_f_maps = vector_processing(tf.reshape(curr_f_maps, [b,h,w,nbre_cuts,-1]))
            curr_f_maps = tf.concat(tf.unstack(curr_f_maps, axis=3), axis=3)
            if prev_f_maps is not None:
                prev_f_maps = vector_processing(tf.reshape(prev_f_maps, [b,h,w,nbre_cuts,-1]))
                prev_f_maps = tf.concat(tf.unstack(prev_f_maps, axis=3), axis=3)

            # Manage level temporal memory
            if (not self.is_training) and prev_f_maps is None and prev_t_data is None:
                prev_t_depth = self.depth_prev_t
                prev_f_maps = self.prev_f_maps
                prev_u = self.lvl_uncertainty_depth
                prev_t_data = True
            elif not prev_t_data is None:
                prev_u = prev_t_data[uncert_depth_choice]
                prev_t_depth = prev_t_data["depth"]

            if prev_l_est is None:
                # Initial state of variables
                para_prev_l = tf.ones([b, h, w, 1])
                depth_prev_l = 1000. * tf.ones([b, h, w, 1])
                other_prev_l = tf.zeros([b, h, w, 4])
                acc_prev_l = tf.ones([b, h, w, 1])
            else:
                other_prev_l = tf.compat.v1.image.resize_bilinear(prev_l_est["other"], [h, w])
                para_prev_l = tf.compat.v1.image.resize_bilinear(prev_l_est["para"], [h, w]) * 2.
                depth_prev_l = tf.compat.v1.image.resize_bilinear(prev_l_est["depth"], [h, w])
                acc_prev_l = tf.compat.v1.image.resize_bilinear(prev_l_est["uncertainty_para"], [h, w]) * 2.

            # Reinitialize temporal memory if sample is part of a new sequence
            # Note : sequences are supposed to be synchronized over the whole batch
            if prev_t_data is None or new_traj[0]:
                curr_l_est = {"para": para_prev_l, "other": other_prev_l, "uncertainty_para": acc_prev_l}
            else:
                with tf.name_scope("preprocessor"):

                    if self.ablation.uncertainty == "relative":
                        prev_t_para_data = prev_d2para(prev_t_depth, rot, trans, camera, rel_uncertainty = prev_u)
                        prev_t_para_data = tf.concat([prev_t_para_data["para"], prev_t_para_data["rel_uncertainty_para"]], axis=-1)
                    else:
                        prev_t_para_data = prev_d2para(prev_t_depth, rot, trans, camera, uncertainty = prev_u)
                        prev_t_para_data = tf.concat([prev_t_para_data["para"], prev_t_para_data["uncertainty_para"]], axis=-1)

                    prev_t_para_data_reproj, _ = reproject(prev_t_para_data, depth_prev_l, rot, trans, camera)
                    cv = get_parallax_sweeping_cv(curr_f_maps, prev_f_maps, para_prev_l, rot, trans, camera, 4,
                                                  nbre_cuts=nbre_cuts)

                    with tf.name_scope("input_prep"):
                        input_features = [cv, tf.math.log(para_prev_l*2**self.lvl_mul), tf.math.log(acc_prev_l*2**self.lvl_mul)]

                        if self.ablation.level_memory:
                            input_features.append(other_prev_l)
                        else:
                            print("Ignoring level memory")

                        if self.ablation.SNCV:
                            autocorr = cost_volume(curr_f_maps, curr_f_maps, 3, nbre_cuts=nbre_cuts)
                            input_features.append(autocorr)
                        else:
                            print("Skipping sncv")

                        if self.ablation.time_recurr:
                            input_features.append(tf.math.log(prev_t_para_data_reproj[:,:,:,:1]*2**self.lvl_mul))
                            if uncert_depth_choice == "uncertainty_depth":
                                input_features.append(tf.math.log(prev_t_para_data_reproj[:,:,:,1:]*2**self.lvl_mul))
                            else:
                                input_features.append(tf.math.log(prev_t_para_data_reproj[:,:,:,1:]))
                        else:
                            print("Skipping time recurrence")

                        f_input = tf.concat(input_features, axis=3)

                with tf.name_scope("depth_estimator"):
                    prev_out = self.disp_refiner(f_input)

                    para_curr_l = tf.exp(tf.clip_by_value(prev_out["para"], -7., 7.))/2**self.lvl_mul
                    uncert_curr_l = tf.exp(tf.clip_by_value(prev_out["uncertainty_para"], -7., 7.))/(2**self.lvl_mul)
                    curr_l_est = {
                        "other": tf.identity(prev_out["mem"]),
                        "para": tf.identity(para_curr_l),
                        "uncertainty_para": tf.identity(uncert_curr_l)
                    }

            # Derive uncertainty related metrics
            if self.ablation.uncertainty == "relative":
                curr_l_est["rel_uncertainty_para"] = curr_l_est["uncertainty_para"] / tf.stop_gradient(curr_l_est["para"])
                depth_data = parallax2depth(curr_l_est["para"], rot, trans, camera, rel_uncertainty=curr_l_est["rel_uncertainty_para"])
            else:
                curr_l_est["rel_uncertainty_para"] = curr_l_est["uncertainty_para"] * tf.stop_gradient(curr_l_est["para"])
                depth_data = parallax2depth(curr_l_est["para"], rot, trans, camera, inv_uncertainty=curr_l_est["uncertainty_para"])
            curr_l_est = {**curr_l_est, **depth_data}

            # Set values for first sample of the trajectory
            if prev_t_data is None or new_traj[0]:
                curr_l_est["depth"] = tf.identity(depth_prev_l, name="estimated_depth")
                curr_l_est["rel_uncertainty_depth"] = tf.ones_like(depth_prev_l, name="estimated_uncertainty")
                curr_l_est["uncertainty_depth"] = tf.stop_gradient(depth_prev_l, name="estimated_uncertainty")

            # Update level memory
            if not self.is_training:
                self.prev_f_maps.assign(curr_f_maps)
                self.depth_prev_t.assign(curr_l_est["depth"])
                self.lvl_uncertainty_depth.assign(curr_l_est[uncert_depth_choice])

            return curr_l_est


class DepthEstimatorPyramid(ks.layers.Layer):
    # Decoder part of the architecture
    # Requires the feature map pyramid(s) produced by the encoder as input

    def __init__(self, settings, deconv_levels=None, regularizer_weight=0.0004, trainable=True):
        super(DepthEstimatorPyramid, self).__init__(trainable=trainable)
        # self.trainable = trainable
        if deconv_levels==None:
            self.deconv_levels = settings["nbre_lvls"]
        else:
            self.deconv_levels = deconv_levels
        self.levels = [
            DepthEstimatorLevel(settings, i+1, regularizer_weight=regularizer_weight) for i in range(settings["nbre_lvls"])
        ]
        self.is_training = settings["is_training"]
        self.is_unsupervised = False #settings["unsupervised"]

    @tf.function
    def call(self, inputs):
        f_maps_pyrs = inputs["f_maps_pyrs"]
        traj_samples = inputs["traj_samples"]
        camera = traj_samples[0]["camera"]

        d_est_seq = []
        for seq_i, (f_pyr_curr, sample) in enumerate(zip(f_maps_pyrs, traj_samples)):
            with tf.name_scope("DepthEstimator_seq"):
                print("Seq sample %i" % seq_i)
                rot = sample['rot']
                trans = sample['trans']

                cnter = float(len(self.levels))
                d_est_curr = None

                # Loop over all the levels of the pyramid
                # Note : the deepest level has to be handled slightly differently due to the absence of deeper level
                for l, (f_maps_curr, level) in enumerate(zip(f_pyr_curr[::-1], self.levels[::-1])):
                    f_maps_prev = None
                    d_est_prev = None
                    if l >= self.deconv_levels:
                        continue

                    if seq_i != 0:
                        f_maps_prev = f_maps_pyrs[seq_i - 1][-l - 1]
                        d_est_prev = d_est_seq[-1][-l - 1]

                    local_camera = camera.copy()
                    local_camera["f"] /= 2. ** cnter
                    local_camera["c"] /= 2. ** cnter

                    if l != 0:
                        d_est = d_est_curr[-1].copy()
                    else:
                        d_est= None

                    local_rot = rot
                    local_trans = trans
                    new_traj = sample["new_traj"]

                    # Level inputs serialization
                    tmp_inputs = {
                        "curr_f_maps":f_maps_curr,
                        "prev_l_est":None,
                        "rot":local_rot,
                        "trans":local_trans,
                        "camera":local_camera,
                        "new_traj":new_traj,
                        "prev_f_maps":f_maps_prev,
                        "prev_t_data":d_est_prev
                    }
                    # Remove None's (required for not breaking tf)
                    lvl_inputs = {k: v for k, v in tmp_inputs.items() if v is not None}

                    if d_est_curr == None:
                        d_est_curr = [level(lvl_inputs)]
                    else:
                        lvl_inputs["prev_l_est"] = d_est
                        d_est_curr.append(level(lvl_inputs))
                    cnter -= 1.

                d_est_seq.append(d_est_curr[::-1])
        return d_est_seq


def _masked_reduce_mean(array, mask, axis=None):
    return tf.reduce_sum(array * mask, axis=axis) / (tf.reduce_sum(mask, axis=axis) + 1e-12)

def downscale_map(input, size, sparse=False, method=tf.image.ResizeMethod.BILINEAR):
    h, w = size

    if sparse:
        b, h_g, w_g = input.get_shape().as_list()[0:3]
        tmp = tf.reshape(input, [b, h, h_g // h, w, w_g // w, 1])
        mask = tf.cast(tf.greater(tmp, 0), tf.float32)

        # resize ground-truth by taking holes into account
        tmp = tf.reshape(input, [b, h, h_g // h, w, w_g // w, 1])
        resized = _masked_reduce_mean(tmp, mask, axis=[2, 4])

        # get valid data points
        sparsity_mask = tf.cast(tf.greater(tf.reduce_sum(mask, axis=[2, 4]), 0.), tf.float32)
    else:
        resized = tf.image.resize(input, [h, w], method=method)
        sparsity_mask = tf.ones_like(resized)

    return resized, sparsity_mask

class M4DepthU(ks.models.Model):
    """Tensorflow model of M4Depth"""

    def __init__(self, depth_type="map", nbre_levels=6, is_training=False, ablation_settings=None, loss_settings=None, get_all_scales=None, deconv_levels=None):
        super(M4DepthU, self).__init__()

        if ablation_settings is None:
            self.ablation_settings = M4depthUAblationParameters()
        else:
            self.ablation_settings = ablation_settings

        if loss_settings is None:
            self.loss_settings = M4depthULossParameters()
        else:
            self.loss_settings = loss_settings

        if get_all_scales is None:
            self.get_all_scales = is_training
        else:
            self.get_all_scales = get_all_scales

        if deconv_levels==None:
            self.deconv_levels = nbre_levels
        else:
            self.deconv_levels = deconv_levels

        self.model_settings = {
            "nbre_lvls": nbre_levels,
            "is_training": is_training,
            "ablation" : self.ablation_settings
        }

        if self.ablation_settings.uncertainty == "relative":
            self.uncert_depth_choice = "rel_uncertainty_depth"
        else:
            self.uncert_depth_choice = "uncertainty_depth"

        self.depth_range = [0.01, 200.]

        self.depth_type = depth_type

        self.encoder = FeaturePyramid(self.model_settings, regularizer_weight=0.)
        self.d_estimator = DepthEstimatorPyramid(self.model_settings, deconv_levels=self.deconv_levels,
                                                 regularizer_weight=0.)

        self.step_counter = tf.Variable(initial_value=tf.zeros_initializer()(shape=[], dtype='int64'), trainable=False)
        self.summaries = []

    @tf.function
    def call(self, data):
        # traj_samples = data[0]
        # camera = data[1]

        traj_samples = self.__unstack_trajectory__(data)


        with tf.name_scope("M4DepthU"):
            self.step_counter.assign_add(1)

            f_maps_pyrs = []
            for sample in traj_samples:
                f_maps_pyrs.append(self.encoder(sample['RGB_im']))

            inputs = {
                "f_maps_pyrs": f_maps_pyrs,
                "traj_samples": traj_samples
            }
            d_maps_pyrs = self.d_estimator(inputs)
            if self.get_all_scales:
                return d_maps_pyrs
            else:
                h, w = traj_samples[-1]['RGB_im'].get_shape().as_list()[1:3]
                invalid_uncertainty = tf.cast(tf.equal(d_maps_pyrs[-1][0]["rel_uncertainty_para"], 0.), dtype=tf.float32)
                return {"depth": tf.image.resize(d_maps_pyrs[-1][0]["depth"], [h, w],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                        "para": tf.image.resize(d_maps_pyrs[-1][0]["para"]*2, [h, w],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                        "rel_uncertainty_depth": tf.image.resize(d_maps_pyrs[-1][0]["rel_uncertainty_depth"], [h, w],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                        "rel_uncertainty_para": tf.image.resize(d_maps_pyrs[-1][0]["rel_uncertainty_para"], [h, w],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                        "uncertainty_depth": tf.image.resize(d_maps_pyrs[-1][0]["uncertainty_depth"], [h, w],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                        "uncertainty_para": tf.image.resize(d_maps_pyrs[-1][0]["uncertainty_para"], [h, w],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                        "accuracy_depth": tf.image.resize(d_maps_pyrs[-1][0]["accuracy_depth"]-invalid_uncertainty, [h, w],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                        }

    def __build_network_inputs__(self, dictionnay):
        input_keys = ["RGB_im", "new_traj", "rot", "trans", "camera"]
        inputs = {}
        for key in input_keys:
            inputs[key] = dictionnay[key]
        return inputs

    def __unstack_trajectory__(self, inputs):
        data_format = len(inputs["RGB_im"].get_shape().as_list())
        if data_format == 5:
            seq_len = inputs["RGB_im"].get_shape().as_list()[1]
            list_of_samples = [{} for i in range(seq_len)]
            for key, value in inputs.items():
                if key != "camera":
                    value_list = tf.unstack(value, axis=1)
                    for i, item in enumerate(value_list):
                        list_of_samples[i][key] = item
                else:
                    for i in range(seq_len):
                        list_of_samples[i]["camera"] = value
        else:
            list_of_samples = [inputs]

        return list_of_samples

    @tf.function
    def train_step(self, data):
        self.model_settings["is_training"] = True
        with tf.name_scope("train_scope"):
            with tf.GradientTape() as tape:

                preds = self(self.__build_network_inputs__(data))
                # Rearrange samples produced by the dataloader
                traj_samples = self.__unstack_trajectory__(data)

                gts = []
                for sample in traj_samples:
                    gts.append({"depth":tf.clip_by_value(sample["depth"], 0., self.depth_range[1]),
                                "para": depth2parallax(tf.clip_by_value(sample["depth"], self.depth_range[0], self.depth_range[1]), sample["rot"], sample["trans"], data["camera"])})

                if self.ablation_settings.uncertainty == "relative":
                    loss, summary_values = self.m4depthu_custom_tailored_loss(gts, preds)
                else:
                    loss, summary_values = self.m4depthu_baseline_loss(gts, preds)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        with tf.name_scope("summaries"):
            tf.summary.scalar("depth_loss", summary_values[0], step=self.step_counter, description=None)
            tf.summary.scalar("acc_loss", summary_values[1], step=self.step_counter, description=None)

            max_d = 200.
            gt_d_clipped = tf.clip_by_value(traj_samples[-1]['depth'], 1., max_d)
            tf.summary.image("RGB_im", traj_samples[-1]['RGB_im'], step=self.step_counter)
            im_reproj, _ = reproject(traj_samples[-2]['RGB_im'], traj_samples[-1]['depth'],
                                     traj_samples[-1]['rot'], traj_samples[-1]['trans'], data["camera"])
            tf.summary.image("camera_prev_t_reproj", im_reproj, step=self.step_counter)

            tf.summary.image("depth_gt", tf.math.log(gt_d_clipped) / tf.math.log(max_d), step=self.step_counter)
            for i, est in enumerate(preds[-1]):
                if i==0:
                    tf.summary.image("rel_uncertainty", tf.nn.tanh(tf.abs(est["rel_uncertainty_para"])), step=self.step_counter)
                d_est_clipped = tf.clip_by_value(est["depth"], 1., max_d)
                self.summaries.append(
                    [tf.summary.image, "depth_lvl_%i" % i, tf.math.log(d_est_clipped) / tf.math.log(max_d)])
                tf.summary.image("depth_lvl_%i" % i, tf.math.log(d_est_clipped) / tf.math.log(max_d),
                                 step=self.step_counter)

        with tf.name_scope("metrics"):
            gt = gts[-1]["depth"]
            est = tf.image.resize(preds[-1][0]["depth"], gt.get_shape()[1:3],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            max_d = 80.
            gt = tf.clip_by_value(gt, 0.00, max_d)
            est = tf.clip_by_value(est, 0.001, max_d)
            self.compiled_metrics.update_state(gt, est)
            out_dict = {m.name: m.result() for m in self.metrics}
            out_dict["loss"] = loss

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return out_dict

    @tf.function
    def test_step(self, data):
        # expects one sequence element at a time (batch dim required and is free to set)"
        self.model_settings["is_training"] = False

        preds = self(self.__build_network_inputs__(data))
        est = preds["depth"]

        # If sequence was received as input, compute performance metrics only on its last frame (required for KITTI benchmark))
        data_format = len(data["depth"].get_shape().as_list())
        if data_format == 5:
            gt = data["depth"][:,-1,:,:,:]
            new_traj=False
        else:
            gt = data["depth"]
            new_traj = data["new_traj"]

        with tf.name_scope("metrics"):
            # Compute performance scores
            max_d = 80.
            infinity_mask = tf.cast(tf.less_equal(gt, max_d), tf.float32)

            gt = gt * infinity_mask
            est = tf.clip_by_value(est, 0.001, max_d) * infinity_mask

            if not new_traj:
                self.compiled_metrics.update_state(gt, est)

        # Return a dict mapping metric names to current value.
        out_dict = {m.name: m.result() for m in self.metrics}
        return out_dict

    @tf.function
    def predict_step(self, data):
        # expects one sequence element at a time (batch dim is required and is free to be set)"
        self.model_settings["is_training"] = False
        preds = self(self.__build_network_inputs__(data))

        with tf.name_scope("metrics"):
            est = preds

            return_data = {
                "image": data["RGB_im"],
                "depth": est["depth"],
                "uncertainty": est[self.uncert_depth_choice],
                "new_traj": data["new_traj"]
            }
        return return_data

    def restore(self, path):
        checkpoint = tf.train.Checkpoint(self)
        checkpoint.restore(path)

    @tf.function
    def m4depthu_custom_tailored_loss(self, gts, preds):
        '''
        Implements the custom tailored loss function to get uncertainty estimates for M4Depth
        '''

        with tf.name_scope("loss_function"):
            # Clip and convert depth
            def preprocess(input):
                return tf.math.log(tf.clip_by_value(input, self.depth_range[0], self.depth_range[1]))

            # Compute the loss for the uncertainty
            def custom_tailored_uncertainty_loss(gt_para, est_para, est_err, mask, desired_error_rate=0.05):
                est_para = tf.stop_gradient(est_para) # Prevent gradient interference with depth estimation

                # Increase weight of bounds that were too low
                gt_err = tf.stop_gradient(gt_para - est_para)

                log_lh = tf.abs(gt_err) / (est_err + 1e-12) + desired_error_rate * tf.math.log(est_err+ 1e-12)

                loss_term = _masked_reduce_mean(log_lh, mask)

                return loss_term

            l1_loss_term = 0.
            uncert_loss_term = 0.
            for gt, pred_pyr in zip(gts[1:], preds[1:]):  # Iterate over sequence
                nbre_points = 0.

                gt_preprocessed = preprocess(gt["depth"])
                gt_h, gt_w = gt_preprocessed.get_shape().as_list()[1:3]

                for i, pred in enumerate(pred_pyr):  # Iterate over the outputs produced by the different levels
                    pred_depth = preprocess(pred["depth"])
                    pref_acc = pred["uncertainty_para"]

                    # Compute loss term
                    b, h, w = pred_depth.get_shape().as_list()[:3]
                    nbre_points += h * w

                    # ensure gt sparsity if using velodyne measurements
                    if self.depth_type == "velodyne":
                        mask = tf.cast(tf.greater(gt["depth"], 0.), tf.float32)
                        gt_para = gt["para"] * mask
                        gt_preprocessed *= mask
                    else:
                        gt_para = gt["para"]

                    gt_depth_resized, mask = downscale_map(gt_preprocessed, [h,w], sparse=(self.depth_type == "velodyne"))
                    gt_para_resized = downscale_map(gt_para, [h,w], sparse=(self.depth_type == "velodyne"))[0] * (float(h)/float(gt_h))

                    # compute loss only on data points

                    l1_loss_lvl = _masked_reduce_mean(tf.abs(gt_depth_resized - pred_depth), mask)
                    uncert_loss_lvl = custom_tailored_uncertainty_loss(gt_para_resized, pred['para'], pref_acc, mask,
                                                                   desired_error_rate=self.loss_settings.lh_weight)

                    l1_loss_term += (0.64 / (2. ** (i - 1))) * l1_loss_lvl / float(len(gts) - 1)
                    uncert_loss_term += (0.64 / (2. ** (i - 1))) * uncert_loss_lvl / float(len(gts) - 1)

            tot_loss = l1_loss_term + self.loss_settings.uncertainty_weight * uncert_loss_term
            return tot_loss, [l1_loss_term, uncert_loss_term]

    @tf.function
    def m4depthu_baseline_loss(self, gts, preds, step=None):
        '''
        Implements the probabilistic baseline loss function to get uncertainty estimates for M4Depth
        '''

        with tf.name_scope("loss_function"):

            # Clip and convert depth
            def preprocess_log(input):
                return tf.math.log(tf.clip_by_value(input, self.depth_range[0], self.depth_range[1]))

            def preprocess_lin(input):
                return tf.clip_by_value(input, self.depth_range[0], self.depth_range[1] * 2.)

            log_l1_loss_term = 0.
            conf_loss_term = 0.
            for gt, pred_pyr in zip(gts[1:], preds[1:]):  # Iterate over sequence
                nbre_points = 0.

                gt_pp_log = preprocess_log(gt["depth"])
                gt_pp_lin = preprocess_lin(gt["depth"])
                gt_h, gt_w = gt_pp_log.get_shape().as_list()[1:3]

                for i, pred in enumerate(pred_pyr):  # Iterate over the outputs produced by the different levels
                    est_d_log = preprocess_log(pred["depth"])
                    est_d_lin = preprocess_lin(pred["depth"])
                    est_u = pred["uncertainty_depth"]

                    # Compute loss term
                    b, h, w = est_d_log.get_shape().as_list()[:3]
                    nbre_points += h * w

                    # ensure gt sparsity if using velodyne measurements
                    if self.depth_type == "velodyne":
                        mask = tf.cast(tf.greater(gt["depth"], 0.), tf.float32)
                        gt_pp_log *= mask

                    gt_d_log, mask = downscale_map(gt_pp_log, [h, w], sparse=(self.depth_type == "velodyne"))
                    gt_d_lin, mask = downscale_map(gt_pp_lin, [h, w], sparse=(self.depth_type == "velodyne"))

                    # compute loss only on data points

                    l1_loss_log = tf.abs(gt_d_log - est_d_log)
                    l1_loss_lin = tf.abs(gt_d_lin - est_d_lin) * (float(h) / float(gt_h))
                    lin_mask = tf.cast(tf.less(gt_d_lin, self.depth_range[1]), tf.float32) # mask pixels past a given distance

                    uncert_loss = tf.stop_gradient(
                        l1_loss_lin) / est_u + self.loss_settings.lh_weight * tf.math.log(est_u)
                    log_l1_loss_term += (0.64 / (2. ** (i - 1))) * _masked_reduce_mean(l1_loss_log, mask) / float(
                        len(gts) - 1)
                    conf_loss_term += (0.64 / (2. ** (i - 1))) * _masked_reduce_mean(uncert_loss,
                                                                                     mask * lin_mask) / float(
                        len(gts) - 1)

            tot_loss = log_l1_loss_term + self.loss_settings.uncertainty_weight * conf_loss_term
            return tot_loss, [log_l1_loss_term, conf_loss_term]
