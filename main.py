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

import os
import argparse
from m4depthu_options import M4DepthUOptions

cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
model_opts = M4DepthUOptions(cmdline)
cmd, test_args = cmdline.parse_known_args()
if cmd.mode == 'eval':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras
import dataloaders as dl
from callbacks import *
from m4depthu_network import *
from metrics import *

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

if __name__ == '__main__':

    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model_opts = M4DepthUOptions(cmdline)
    cmd, test_args = cmdline.parse_known_args()

    # configure tensorflow gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        print("GPU memory limit failed")
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    enable_validation = cmd.enable_validation

    working_dir = os.getcwd()
    print("The current working directory is : %s" % working_dir)

    chosen_dataloader = dl.get_loader(cmd.dataset)

    seq_len = cmd.seq_len
    nbre_levels = cmd.arch_depth
    ckpt_dir = cmd.ckpt_dir
    if "tartanair" in cmd.dataset or "midair-" in cmd.records_path:
        tmp = os.path.split(cmd.records_path)
        save_dir_dataset = os.path.basename(tmp[-2]) if tmp[-1] == "" else tmp[-1]
    else:
        save_dir_dataset = cmd.dataset

    if cmd.mode == 'train' or cmd.mode == 'finetune':

        print("Training on %s" % cmd.dataset)
        tf.random.set_seed(42)
        chosen_dataloader.get_dataset("train", model_opts.dataloader_settings, batch_size=cmd.batch_size)
        data = chosen_dataloader.dataset

        model = M4DepthU(depth_type=chosen_dataloader.depth_type,
                        nbre_levels=nbre_levels,
                        ablation_settings=model_opts.ablation_settings,
                        is_training=True,
                        loss_settings=model_opts.loss_settings)

        # Initialize callbacks
        tensorboard_cbk = keras.callbacks.TensorBoard(
            log_dir=cmd.log_dir, histogram_freq=1200, write_graph=True,
            write_images=False, update_freq=1200,
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None)
        model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir,"train"), resume_training=True)

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(optimizer=opt, metrics=[RootMeanSquaredLogError()])

        if enable_validation:
            val_cbk = [CustomKittiValidationCallback(cmd, args=test_args)]
        else:
            val_cbk = []

        # Adapt number of steps depending on desired usecase
        if cmd.mode == 'finetune':
            nbre_epochs = model_checkpoint_cbk.resume_epoch + (20000 // chosen_dataloader.length)
        else:
            nbre_epochs = (250000 // chosen_dataloader.length)

        model.fit(data, epochs= nbre_epochs + 1,
                  initial_epoch=model_checkpoint_cbk.resume_epoch,
                  callbacks=[tensorboard_cbk, model_checkpoint_cbk] + val_cbk)

    elif cmd.mode == 'validation':

        weights_dir = os.path.join(ckpt_dir,"train")

        print("Evaluating on %s" % cmd.dataset)
        chosen_dataloader.get_dataset("eval", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset

        tb_callback = ProfilePredictCallback(log_dir=cmd.log_dir, profile_batch='10, 25', write_images=False)

        model = M4DepthU(nbre_levels=nbre_levels, ablation_settings=model_opts.ablation_settings)


        model_checkpoint_cbk = CustomCheckpointCallback(weights_dir, resume_training=True)
        model.compile(metrics=[AbsRelError(),
                               SqRelError(),
                               RootMeanSquaredError(),
                               RootMeanSquaredLogError(),
                               ThresholdRelError(1), ThresholdRelError(2), ThresholdRelError(3)])

        metrics = model.evaluate(data, callbacks=[model_checkpoint_cbk])

        # Keep track of the computed performance
        manager = BestCheckpointManager(os.path.join(ckpt_dir,"train"), os.path.join(ckpt_dir,"best"), keep_top_n=5)
        perfs = {"abs_rel": [metrics[0]], "sq_rel": [metrics[1]], "rmse": [metrics[2]], "rmsel": [metrics[3]],
                 "a1": [metrics[4]], "a2": [metrics[5]], "a3": [metrics[6]]}
        manager.update_backup(perfs)
        string = ''
        for perf in metrics:
            string += format(perf, '.4f') + "\t\t"
        with open(os.path.join(*[ckpt_dir, "validation-perfs.txt"]), 'a') as file:
            file.write(string + '\n')

    elif cmd.mode == "eval":
        weights_dir = os.path.join(ckpt_dir,"best")

        print("Evaluating on %s" % cmd.dataset)
        chosen_dataloader.get_dataset("eval", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset

        tb_callback = ProfilePredictCallback(log_dir=cmd.log_dir, profile_batch='10, 25', write_images=False)

        model = M4DepthU(nbre_levels=nbre_levels, ablation_settings=model_opts.ablation_settings)

        model_checkpoint_cbk = CustomCheckpointCallback(weights_dir, resume_training=True)
        model.compile()

        first_sample = data.take(1)
        model.predict(first_sample, callbacks=[model_checkpoint_cbk])

        # Enable to get inference stats
        # tmp = model.predict(data.take(200), callbacks=[model_checkpoint_cbk,tb_callback])
        # model.summary()

        # Performance metrics to compute for evaluation
        error_metrics = [AbsRelError(), RootMeanSquaredLogError(), ThresholdRelError(1)]
        ause_metrics = [AuSEAbsRelError(), AuSERootMeanSquaredLogError(), AuSEThresholdRelError(1, name="AuSE_Delta1")]

        # Prevent excessive data processing time by downscaling and downsampling outputs of large datasets
        downscale = 2
        downsample = 4
        max_depth = 80.

        if "kitti" in cmd.dataset: # Don't downscale data registered from velodyne sensor
            h, w = [chosen_dataloader.out_size[0], chosen_dataloader.out_size[1]]
        else:
            h, w = [chosen_dataloader.out_size[0]//downscale, chosen_dataloader.out_size[1]//downscale]

        def get_perf_string(metrics):
            # Used to get the result of an array of performance metrics in a string
            out = ""
            for metric in metrics:
                out += metric.name + " :" + "{:9.4f}".format(metric.result()) + "\t"
            return out

        # Custom evaluation loop. This is required because of the difference in inputs between regular and AuSE performance metrics.
        for i, sample in enumerate(data):
            outputs = model.predict_step(sample)

            depth_validity = tf.cast(tf.greater(sample["depth"], 1e-3), tf.float32) * tf.cast(tf.less(sample["depth"], max_depth), tf.float32)
            d_gt = tf.clip_by_value(sample["depth"], 0.001, max_depth)
            d_est = tf.clip_by_value(outputs["depth"], 0.001, max_depth)
            uncert = tf.identity(outputs["uncertainty"])
            # tf.print(outputs["uncertainty"])

            if "kitti" in cmd.dataset and not (i+1)%4: # Only process the last element of a sequence on the KITTI dataset
                for metric in ause_metrics:
                    metric.update_state(d_gt, d_est, uncert, mask=depth_validity)

                for metric in error_metrics:
                    metric.update_state(d_gt*depth_validity, d_est)

            elif not "kitti" in cmd.dataset and not sample["new_traj"]:
                if not i%downsample:
                    depth_gt_resized = tf.image.resize(d_gt, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    depth_est_resized = tf.image.resize(d_est, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    uncertainty_resized = tf.image.resize(uncert, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    validity_mask = tf.image.resize(depth_validity, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                    for metric in ause_metrics:
                        metric.update_state(depth_gt_resized, depth_est_resized, uncertainty_resized, mask = validity_mask)

                for metric in error_metrics:
                    metric.update_state(d_gt*depth_validity, d_est)

            printProgressBar(i, chosen_dataloader.length, prefix='Progress:', suffix=get_perf_string(error_metrics), length=50)
        printProgressBar(chosen_dataloader.length, chosen_dataloader.length, prefix='Progress:', suffix=get_perf_string(error_metrics), length=50)

        readable_results_ause = get_perf_string(ause_metrics) # Warning, getting the results for ause metrics is a heavy operation
        readable_results_error = get_perf_string(error_metrics)

        # Print results in the console
        print("Averaged performance scores:")
        print(readable_results_error)
        print(readable_results_ause)

        file = open(os.path.join(*[ckpt_dir, "performance-" + save_dir_dataset + ".txt"]), "w")
        file.write(readable_results_error + "\n\r")
        file.write(readable_results_ause + "\n\r")

    elif cmd.mode == "predict":

        chosen_dataloader.get_dataset("predict", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset

        model = M4DepthU(nbre_levels=nbre_levels, ablation_settings=model_opts.ablation_settings)
        model.compile()
        model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir, "best"), resume_training=True)
        first_sample = data.take(1)
        model.predict(first_sample, callbacks=[model_checkpoint_cbk])

        is_first_run = True

        # Do what you want with the outputs
        for i, sample in enumerate(data):
            if not is_first_run and sample["new_traj"]:
                print("End of trajectory")

            is_first_run = False

            est = model([[sample], sample["camera"]])  # Run network to get estimates
            d_est = est["depth"][0, :, :, :]  # Estimate : [h,w,1] matrix with depth in meter
            uncertainty = est['uncertainty'][0, :, :, :]  # Uncertainty on the depth estimate : [h,w,1] matrix, unitless
            d_gt = sample['depth'][0, :, :, :]  # Ground truth : [h,w,1] matrix with depth in meter
            i_rgb = sample['RGB_im'][0, :, :, :]  # RGB image : [h,w,3] matrix with rgb channels ranging between 0 and 1
