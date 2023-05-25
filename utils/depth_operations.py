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

The functions in this file implement the equations presented in the paper. Please refer to it
for details about the operations performed here.
"""

import tensorflow as tf
from utils import dense_image_warp

def get_rot_mat(rot):
    """ Converts a rotation vector into a rotation matrix

    If the vector is of length 3 an "xyz"  small rotation sequence is expected
    If the vector is of length 4 an "wxyz" quaternion is expected
    """

    b, c = rot.get_shape().as_list()
    if c == 3:
        ones = tf.ones([b])
        matrix = tf.stack((ones, -rot[:, 2], rot[:, 1],
                           rot[:, 2], ones, -rot[:, 0],
                           -rot[:, 1], rot[:, 0], ones), axis=-1)

        output_shape = tf.concat((tf.shape(input=rot)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)
    elif c == 4:
        w, x, y, z = tf.unstack(rot, axis=-1)
        tx = 2.0 * x
        ty = 2.0 * y
        tz = 2.0 * z
        twx = tx * w
        twy = ty * w
        twz = tz * w
        txx = tx * x
        txy = ty * x
        txz = tz * x
        tyy = ty * y
        tyz = tz * y
        tzz = tz * z
        matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                           txy + twz, 1.0 - (txx + tzz), tyz - twx,
                           txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                          axis=-1)  # pyformat: disable
        output_shape = tf.concat((tf.shape(input=rot)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)
    else:
        raise ValueError('Rotation must be expressed as a small angle (x,y,z) or a quaternion (w,x,y,z)')


@tf.function
def get_coords_2d(map, camera):
    """ Creates a grid containing pixel coordinates normalized by the camera focal length """

    b, h, w, c = map.get_shape().as_list()
    h_range = tf.range(0., h, 1.0, dtype=tf.float32) + 0.5
    w_range = tf.range(0., w, 1.0, dtype=tf.float32) + 0.5
    grid_x, grid_y = tf.meshgrid(w_range, h_range)
    mesh = tf.reshape(tf.stack([grid_x, grid_y], axis=2), [1, h, w, 2]) - tf.reshape(camera["c"], [b, 1, 1, 2])

    coords_2d = tf.concat([tf.divide(mesh, tf.reshape(camera["f"], [b, 1, 1, 2])), tf.ones([b, h, w, 1])], axis=-1)
    coords_2d = tf.expand_dims(coords_2d, -1)
    return coords_2d, mesh


@tf.function
def reproject(map, depth, rot, trans, camera):
    """ Spatially warps (reprojects) an input feature map according to given depth map, motion and camera properties """

    with tf.name_scope("reproject"):
        # Test the shape of the inputs
        b, h, w, c = map.get_shape().as_list()
        b, h1, w1, c = depth.get_shape().as_list()
        if w != w1 or h != h1:
            raise ValueError('Height and width of map and depth should be the same')

        # Reshape motion data in a format compatible for vector math
        fx = camera["f"][:, 0]
        fy = camera["f"][:, 1]

        proj_mat = []
        for i in range(b):
            proj_mat.append([[fx[i], 0., 0.], [0., fy[i], 0.], [0., 0., 1.]])
        proj_mat = tf.convert_to_tensor(proj_mat)

        rot_mat = get_rot_mat(rot)
        transformation_mat = tf.concat([rot_mat, tf.expand_dims(trans, -1)], -1)

        # Fuse projection matrix K with transformation matrix
        combined_mat = tf.linalg.matmul(proj_mat, transformation_mat)
        combined_mat = tf.reshape(combined_mat, [b, 1, 1, 3, 4])

        # Get the relative coordinates for each point of the map
        coords, mesh = get_coords_2d(map, camera)
        pos_vec = tf.expand_dims(tf.concat([coords[:, :, :, :, 0] * depth, tf.ones([b, h, w, 1])], axis=-1), axis=-1)

        # Compute corresponding coordinates in related frame
        proj_pos = tf.linalg.matmul(combined_mat, pos_vec)
        proj_coord = proj_pos[:, :, :, :2, 0] / proj_pos[:, :, :, 2:, 0]
        rot_pos = tf.linalg.matmul(combined_mat[:, :, :, :, :3], pos_vec[:, :, :, :3, :])
        rot_coord = rot_pos[:, :, :, :2, 0] / rot_pos[:, :, :, 2:, 0]

        flow = tf.reverse(proj_coord - mesh, axis=[-1])

    return dense_image_warp(map, flow), [proj_coord - rot_coord, rot_coord]


@tf.function
def recompute_depth(depth, rot, trans, camera, mesh=None):
    """ Recomputes perceived depth according to given camera motion and specifications """

    with tf.compat.v1.name_scope("recompute_depth"):
        depth = tf.identity(depth)
        b, h, w, c = depth.get_shape().as_list()

        # Reshape motion data in a format compatible for vector math
        trans_vec = tf.reshape(-trans, [b, 1, 1, 3, 1])
        rot_mat = get_rot_mat(rot)[:, -1:, :]

        # Get the relative coordinates for each point of the map
        if mesh is None:
            h_range = tf.range(0., h, 1.0, dtype=tf.float32) + 0.5
            w_range = tf.range(0., w, 1.0, dtype=tf.float32) + 0.5
            grid_x, grid_y = tf.meshgrid(w_range, h_range)
            mesh = tf.reshape(tf.stack([grid_x, grid_y], axis=2), [1, h, w, 2]) - tf.reshape(camera["c"], [b, 1, 1, 2])

        coords_2d = tf.concat([tf.divide(mesh, tf.reshape(camera["f"], [b, 1, 1, 2])), tf.ones([b, h, w, 1])], axis=-1)
        pos_vec = tf.expand_dims(coords_2d, -1)

        # Recompute depth
        trans_vec = tf.linalg.matmul(tf.reshape(rot_mat, [b, 1, 1, 1, 3]), trans_vec)
        proj_pos_rel = tf.linalg.matmul(tf.reshape(rot_mat, [b, 1, 1, 1, 3]), pos_vec)
        new_depth = tf.stop_gradient(proj_pos_rel[:, :, :, :, 0]) * depth + tf.stop_gradient(trans_vec[:, :, :, :, 0])
        return tf.clip_by_value(new_depth, 0.1, 2000.)


@tf.function
def get_rel_uncertainty_depth(depth, rel_u_para, const):
    """
    Converts the relative uncertainty on parallax into relative uncertainty on depth using our custom-tailored method
    """

    para_fac = 1 + rel_u_para
    depth_fac = para_fac + const * (1 - para_fac) / depth
    rel_uncertainty_depth = depth_fac - 1.
    return rel_uncertainty_depth


@tf.function
def get_rel_uncertainty_para(depth, rel_u_depth, const):
    """
    Converts the relative uncertainty on depth into relative uncertainty on parallax using our custom-tailored method
    """

    depth_fac = rel_u_depth + 1.
    tmp = const / depth
    para_fac = (depth_fac - tmp) / (1 - tmp)
    rel_uncertainty_para = para_fac - 1
    return rel_uncertainty_para


@tf.function
def get_uncertainty_depth(u_inv_para, const):
    """
    Converts the relative uncertainty on inverse parallax into relative uncertainty on depth using the probabilistic approach
    """

    return tf.abs(const)*u_inv_para


@tf.function
def get_uncertainty_inv_para(u_depth, const):
    """
    Converts the relative uncertainty on depth into relative uncertainty on the inverse parallax using the probabilistic approach
    """

    return u_depth/tf.abs(const)


@tf.function
def parallax2depth(para, rot, trans, camera, rel_uncertainty=None, inv_uncertainty=None):
    """
    Converts a parallax map into a depth map according to given camera motion and specifications
    Also converts the given uncertainty accordingly.

    """

    with tf.compat.v1.name_scope("parallax2depth"):
        b, h, w = para.get_shape().as_list()[0:3]

        coords2d, _ = get_coords_2d(para, camera)

        para = tf.maximum(tf.reshape(para, [b, h * w, 1, 1]), 1e-5)
        coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])
        rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
        t = tf.reshape(trans, [b, 1, 3, 1])
        f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b, 1])], axis=1), [b, 1, 3, 1])

        rot_coords = rot_mat @ coords2d
        alpha = rot_coords[:, :, -1:, :]
        proj_coords = rot_coords * f_vec / alpha
        scaled_t = t * f_vec

        delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 0, 0]
        delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 1, 0]

        sqrt_value = tf.reshape(tf.sqrt(delta_x ** 2 + delta_y ** 2), [b, h * w, 1, 1])

        depth = (sqrt_value / para - scaled_t[:, :, -1:, :]) / alpha
        depth = tf.math.maximum(depth, 1e-5)  # prevent depth from going negative, therefore breaking our hypothesis

        depth = tf.reshape(depth, [b, h, w, 1])

        if rel_uncertainty is None and inv_uncertainty is None:
            return depth
        else:
            if not rel_uncertainty is None:
                const = tf.reshape(- scaled_t[:, :, -1:, :] / alpha, [b, h, w, 1])
                uncert = get_rel_uncertainty_depth(tf.stop_gradient(depth), rel_uncertainty, const)
                accuracy = 1. / (1. + uncert)
                return {"depth"                 : tf.identity(depth),
                        "rel_uncertainty_depth" : tf.identity(uncert),
                        "uncertainty_depth"     : tf.identity(uncert*depth),
                        "accuracy_depth"        : tf.identity(accuracy)}
            else:
                const = tf.reshape(sqrt_value / alpha, [b, h, w, 1])
                uncert = get_uncertainty_depth(inv_uncertainty, tf.maximum(const, 1e-6))
                accuracy = 1. / (1. + uncert/depth)
                return {"depth"                 : tf.identity(depth),
                        "rel_uncertainty_depth" : tf.identity(uncert/tf.stop_gradient(depth)),
                        "uncertainty_depth"     : tf.identity(uncert),
                        "accuracy_depth"        : tf.identity(accuracy)}


@tf.function
def depth2parallax(depth, rot, trans, camera, rel_uncertainty=None, uncertainty=None):
    """
    Converts a depth map into a parallax map according to given camera motion and specifications.
    Also converts the given uncertainty accordingly.

    """

    with tf.compat.v1.name_scope("depth2parallax"):
        b, h, w = depth.get_shape().as_list()[0:3]

        coords2d, _ = get_coords_2d(depth, camera)

        coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])
        rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
        t = tf.reshape(trans, [b, 1, 3, 1])
        f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b, 1])], axis=1), [b, 1, 3, 1])

        rot_coords = rot_mat @ coords2d
        alpha = rot_coords[:, :, -1:, :]
        proj_coords = rot_coords * f_vec / alpha
        scaled_t = t * f_vec

        delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 0, 0]
        delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 1, 0]

        sqrt_value = tf.reshape(tf.sqrt(delta_x ** 2 + delta_y ** 2), [b, h * w, 1, 1])

        para = sqrt_value / (tf.reshape(depth, [b, h * w, 1, 1]) * alpha + scaled_t[:, :, -1:, :])

        para = tf.maximum(tf.reshape(para, [b, h, w, 1]), 1e-6)

        if rel_uncertainty is None and uncertainty is None:
            return para
        else:
            if not rel_uncertainty is None:
                const = tf.reshape(- scaled_t[:, :, -1:, :] / alpha, [b, h, w, 1])
                uncert = get_rel_uncertainty_para(depth, rel_uncertainty, const)
                accuracy = 1. / (1. + uncert)
                return {"para"                 : tf.identity(para),
                        "rel_uncertainty_para" : tf.identity(uncert),
                        "uncertainty_para"     : tf.identity(uncert*para),
                        "accuracy_para"        : tf.identity(accuracy)}
            else:
                const = tf.reshape(sqrt_value / alpha, [b, h, w, 1])
                uncert = get_uncertainty_inv_para(uncertainty, const)
                accuracy = 1. / (1. + uncert/para)
                return {"para"                 : tf.identity(para),
                        "rel_uncertainty_para" : tf.identity(uncert*para),  # uncert on the inverse parallax
                        "uncertainty_para"     : tf.identity(uncert),
                        "accuracy_para"        : tf.identity(accuracy)}


@tf.function
def prev_d2para(prev_d, rot, trans, camera, rel_uncertainty=None, uncertainty=None):
    """ Converts depth map corresponding to previous time step into the parallax map corresponding to current time step """

    with tf.compat.v1.name_scope("prev_d2para"):
        b, h, w = prev_d.get_shape().as_list()[0:3]

        coords2d, _ = get_coords_2d(prev_d, camera)

        coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])
        t = tf.reshape(trans, [b, 1, 3, 1])
        f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b, 1])], axis=1), [b, 1, 3, 1])

        coords2d = coords2d * f_vec
        scaled_t = t * f_vec

        delta = (scaled_t - t[:, :, -1:, :] * coords2d)
        numerator = tf.norm(delta[:, :, :2, :], axis=2, keepdims=True)

        para = tf.norm(delta[:, :, :2, :] / (tf.reshape(prev_d, [b, h * w, 1, 1]) - t[:, :, -1:, :]), axis=2)
        para = tf.clip_by_value(para, tf.math.exp(-7.), tf.math.exp(7.))
        para = tf.stop_gradient(tf.reshape(para, [b, h, w, 1]))

        # convert uncertainty related to previous depth estimate
        if rel_uncertainty is None and uncertainty is None:
            return para
        else:
            if not rel_uncertainty is None:
                depth_fac = 1. + rel_uncertainty
                numerator = tf.reshape(numerator, [b, h, w, 1])
                const = -t[:, :, -1:, :]

                # Compute parallax value equivalent to depth*rel_uncert, then divide by expected para to get the inverse
                # of the parallax factor (see illustration in paper)
                inv_para_fac = tf.math.divide_no_nan(numerator, (tf.abs(prev_d * depth_fac + const) * para))
                uncert = 1. / tf.maximum(inv_para_fac, 1e-6) - 1.

                uncert = tf.abs(uncert)  # necessary to avoid issue in log later
                uncert = tf.maximum(uncert, 1e-6)

                accuracy = 1 / (1 + uncert)
                return {"para"                 : tf.stop_gradient(para),
                        "rel_uncertainty_para" : tf.stop_gradient(uncert),
                        "uncertainty_para"     : tf.stop_gradient(uncert*para),
                        "accuracy_para"        : tf.stop_gradient(accuracy)}
            else:
                const = tf.reshape(tf.norm(delta[:, :, :2, :], axis=2), [b, h, w, 1])
                uncert = uncertainty/tf.maximum(const, 1e-6)
                accuracy = 1. / (1. + uncert * para)
                return {"para"                 : tf.stop_gradient(para),
                        "rel_uncertainty_para" : tf.stop_gradient(uncert*para), # uncert on the inverse parallax
                        "uncertainty_para"     : tf.stop_gradient(uncert),
                        "accuracy_para"        : tf.stop_gradient(accuracy)}


def tile_in_batch(map, nbre_copies):
    map_shape = map.get_shape().as_list()
    map = tf.expand_dims(map, axis=0)
    map = tf.tile(map, [nbre_copies] + [1 for i in map_shape])
    return tf.reshape(map, [nbre_copies * map_shape[0]] + map_shape[1:-1] + [
        -1])  # out. shape is the following: [nbre_copies*map_shape[0]]+map_shape[1:])


@tf.function
def get_parallax_sweeping_cv(c1, c2, para, rot, trans, camera, search_range, additional_data=None, nbre_cuts=1):
    """ Computes the PSCV as presented in the paper introducing M4Depth """

    with tf.compat.v1.name_scope("PSCV"):
        # Prepare inputs
        nbre_copies = 2 * search_range + 1
        expl_range = tf.reshape(tf.range(-search_range, search_range + 1, 1.0, dtype=tf.float32), [-1, 1, 1, 1, 1])
        if not additional_data is None:
            b, h, w, c_add = additional_data.get_shape().as_list()[0:4]

        b, h, w, c = c1.get_shape().as_list()[0:4]

        para = tile_in_batch(para, nbre_copies)
        para = tf.reshape(para, [nbre_copies, -1, w, h, 1])
        para = tf.reshape(para + expl_range, [-1, h, w, 1])  # [nbre_copies*b,h,w,1]
        para = tf.clip_by_value(para, 1e-6, 1e6)

        # Compute para independent factors
        coords2d, _ = get_coords_2d(c1, camera)
        coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])

        rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
        t = tf.reshape(trans, [b, 1, 3, 1])
        f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b, 1])], axis=1), [b, 1, 3, 1])

        rot_coords = rot_mat @ coords2d
        alpha = rot_coords[:, :, -1:, :]
        proj_coords = rot_coords * f_vec / alpha
        scaled_t = t * f_vec

        delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 0, 0]
        delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:, :, 1, 0]
        delta_x = tf.reshape(delta_x, [1, b, h, w, 1])
        delta_y = tf.reshape(delta_y, [1, b, h, w, 1])

        start_coords = tf.reshape(coords2d[:, :, :2, :] * f_vec[:, :, :2, :], [1, b, h, w, 2])
        proj_coords = tf.reshape(proj_coords[:, :, :2, :], [1, b, h, w, 2])

        # para to flow
        para = tf.reshape(para, [nbre_copies, b, h, w, 1])
        sqrt_value = tf.sqrt(delta_x ** 2 + delta_y ** 2)
        divider = sqrt_value / para  # is correct computation after simplification
        delta = tf.concat([delta_x / divider, delta_y / divider], axis=-1)
        flow = proj_coords + delta - start_coords
        flow = tf.reshape(tf.reverse(flow, axis=[-1]), [nbre_copies * b, h, w, 2])

        c1 = tile_in_batch(c1, nbre_copies)

        if additional_data is None:
            combined_data = tile_in_batch(c2, nbre_copies)
        else:
            combined_data = tile_in_batch(tf.concat([c2, additional_data], axis=-1), nbre_copies)

        combined_data_w = dense_image_warp(combined_data, flow)

        c2_w = combined_data_w[:, :, :, :c]

        # Compute costs (operations performed in float16 for speedup)
        sub_costs = tf.stack(
            tf.split(tf.cast(c1, tf.float16) * tf.cast(c2_w, tf.float16), num_or_size_splits=nbre_cuts, axis=-1), 0)
        cv = tf.reduce_mean(sub_costs, axis=-1)
        cv = tf.cast(tf.transpose(tf.reshape(cv, [(nbre_cuts) * nbre_copies, -1, h, w]), perm=[1, 2, 3, 0]), tf.float32)

        if not additional_data is None:
            additional_data_w = combined_data_w[:, :, :, -c_add:]
            additional_data_w = tf.transpose(tf.reshape(additional_data_w, [nbre_copies, -1, h, w, c_add]),
                                             perm=[1, 2, 3, 4, 0])
            return cv, additional_data_w
        else:
            return cv


@tf.function
def cost_volume(c1, c2, search_range, name="cost_volume", dilation_rate=1, nbre_cuts=1):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Feature map 1
        c2: Feature map 2
        search_range: Search range (maximum paralacement)
    """
    with tf.compat.v1.name_scope(name):
        c1 = tf.cast(c1, tf.float16)
        c2 = tf.cast(c2, tf.float16)
        strided_search_range = search_range * dilation_rate
        padded_lvl = tf.pad(c2, [[0, 0], [strided_search_range, strided_search_range],
                                 [strided_search_range, strided_search_range], [0, 0]])
        _, h, w, _ = c2.get_shape().as_list()
        max_offset = search_range * 2 + 1

        c1_nchw = tf.transpose(c1, perm=[0, 3, 1, 2])
        pl_nchw = tf.transpose(padded_lvl, perm=[0, 3, 1, 2])

        c1_nchw = tf.stack(tf.split(c1_nchw, num_or_size_splits=nbre_cuts, axis=1), axis=4)
        pl_nchw = tf.stack(tf.split(pl_nchw, num_or_size_splits=nbre_cuts, axis=1), axis=4)

        cost_vol = []
        for y in range(0, max_offset):
            for x in range(0, max_offset):
                slice = tf.slice(pl_nchw, [0, 0, y * dilation_rate, x * dilation_rate, 0], [-1, -1, h, w, -1])
                cost = tf.reduce_mean(c1_nchw * slice, axis=1)
                cost_vol.append(cost)
        cost_vol = tf.concat(cost_vol, axis=3)
        cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

        return tf.cast(cost_vol, tf.float32)
