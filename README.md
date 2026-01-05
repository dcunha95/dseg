# dseg

This repo contains the core toolkit for deep learning segmentation of plaque and lumen regions of Intravascular Ultrasound (IVUS) images used in[](), running on top of `tensorflow.keras`.

Please note that it is not currently in development, being originally developed for the now deprecated Python 3.7.

## Usage 

It provides a collection of classes (see `setup.py`) designed to allow running multiple experiments with various parameter settings being managed by Python.

You're likely here for the U-net and U-net++ constructors. If that's the case, please check the methods `dseg.nets.NetBuilder.unet` and `dseg.nets.NetBuilder.unet_pp`. The *Example script* section provides a standard use case.

## Example script

```python
import tensorflow as tf
from dseg.model import Model
from dseg.setup import NetConfig, PipelineConfig, FitConfig, Setup
from dseg.data_manipulator import DataUtils

if __name__ == "__main__":

    # %% Open mirrored scope and reduce memory footprint

    print("----> cleaning session")
    tf.keras.backend.clear_session()

    print('----> starting mirrored strategy')
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    print('----> opening strategy scope')
    with strategy.scope():
    # if True:

        print("----> setting flags")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        tf.config.optimizer.set_jit("autoclustering")

        # %% Training setup
        print("----> setting up net config")
        net_config = NetConfig(
            model_type="unet++",
            depth=5,
            pool_size=2,
            concat_all=True,
            node_type=4,
            image_size=(128, 128),
            base_filters=16,
            kernel_size=3,
            dropout_amount=0.3,
            use_bn=True,
            down_size=4,
            label_amount=3,
            multi_output=False,
            channels=3,
            channel_strides=2,
        )

        print("----> setting up fit config")
        fit_config = FitConfig(
            sample_weight=None,
            batch_size=64,
            epochs=30,
            optimizer="adam",
            learning_rate=0.001,
            lr_decay_after_epoch=100,
            lr_decay=0.05,
            loss="iou",
            # loss="binary_crossentropy",
        )

        print("----> setting up pipeline config")
        pipeline_config = PipelineConfig(
            split=[0.6, 0.2, 0.2],
            dataset_percent=1,
            # print_options=["raw", "output", "input", "input_original", "gt", "gt_original", "channels", "contour"],
            name_format=["Average", "Name"],
        )

        print("----> setting up setup")
        setup = Setup(
            pipeline_config=pipeline_config,
            net_config=net_config,
            fit_config=fit_config,
            # model_from_file="ivus_1/model.h5",
            # base_training_info=0,
            notes="",
        )
        # %% Instantiate model object

        print("----> instantiating model")
        model = Model(
            setup,
            model_name="ivus",
        )

        # %% Load Dataset

        print("----> retrieving dataset")
        trn = DataUtils.load_dataset_reference("dataset/dataset_1MC_train.csv")
        val = DataUtils.load_dataset_reference("dataset/dataset_1MC_val.csv")
        tst = DataUtils.load_dataset_reference("dataset/dataset_1MC_test.csv")
        stt = DataUtils.load_dataset_reference("dataset/dataset_1MC_stat.csv")

        # print("----> retrieving dataset")
        # dataset = DataUtils.load_dataset_reference("dataset/reduced.csv")
        # trn, val, tst = DataUtils.split_dataset(dataset)
        # stt = val.copy()

        if net_config.channels != 1:
            print("----> processing dataset")
            trn = DataUtils.process_dataset(trn)
            val = DataUtils.process_dataset(val)
            tst = DataUtils.process_dataset(tst)
            stt = DataUtils.process_dataset(stt)

            print("----> pruning dataset")
            trn = DataUtils.prune_dataset(trn, channels=net_config.channels, strides=net_config.channel_strides)
            val = DataUtils.prune_dataset(val, channels=net_config.channels, strides=net_config.channel_strides)
            tst = DataUtils.prune_dataset(tst, channels=net_config.channels, strides=net_config.channel_strides)
            stt = DataUtils.get_available_from_dataset(stt, val)
        
        print("----> passing dataset to model")
        model.get_dataset(trn=trn, val=stt, tst=tst, stt=stt)    

        print("----> compiling")
        model.compile()

    # %% Fit

    print("----> starting fit")

    model.fit()

    # %% Run analysis, save metrics and examples

    print("----> making predictions")
    model.predict(data=stt, save_folder="out")

    print("----> plot training")
    training = model.plot_training()

    print("----> running analysis")
    model.run_analysis()

    print("----> saving analysis")
    model.analysis['data_formatted'].to_csv(model.model_name + '/analysis.csv')
    model.analysis['data_info_formatted'].to_csv(model.model_name + '/analysis_summary.csv')
    DataUtils.make_tables(analysis=model.analysis, path=model.model_name)
    
    print("----> saving examples")
    model.predict(data=model.analysis['data_formatted'], save_folder=model.model_name + '/stat_gen', simple_print=False)
    # model.predict(data=model.analysis['data_sorted'].iloc[0:100], save_folder=model.model_name + '/worst_preds', simple_print=False)
    # model.predict(data=model.analysis['data_sorted'].iloc[-100:], save_folder=model.model_name + '/best_preds', simple_print=False)
```