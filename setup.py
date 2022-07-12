import json


class PipelineConfig:
    def __init__(
        self,
        split=[0.6, 0.2, 0.2],
        dataset_percent=0.1,
        print_options=[True, True, True, True, True, True],
        name_format=["Average", "Name"],
    ):
        # pipeline related:
        self.split = split
        self.dataset_percent = dataset_percent
        self.print_options = print_options  # print options: [raw, output, input, input_original, gt, gt_original]
        self.name_format = name_format


class NetConfig:
    def __init__(
        self,
        model_type="unet",
        depth=4,
        pool_size=2,
        concat_all=True,
        node_type=4,
        image_size=(16, 16),
        down_size=None,
        base_filters=2,
        kernel_size=3,
        dropout_amount=0.3,
        use_bn=True,
        label_amount=3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        multi_output=False,
        channels = 1,
        channel_strides = 1,
    ):
        # Net related:
        self.model_type = model_type
        self.depth = depth
        self.pool_size = pool_size
        self.concat_all = concat_all
        self.node_type = node_type
        self.image_size = image_size
        self.down_size = down_size
        self.input_shape = self.image_size + (channels,)
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.dropout_amount = dropout_amount
        self.use_bn = use_bn
        self.label_amount = label_amount
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.multi_output = multi_output
        self.channels = channels
        self.channel_strides = channel_strides

class FitConfig:
    def __init__(
        self,
        sample_weight=None,
        batch_size=40,
        epochs=20,
        optimizer="adam",
        learning_rate=0.001,
        lr_decay_after_epoch=None,
        lr_decay=0.05,
        loss="categorical_crossentropy",
        monitor="val_loss",  # "val_loss" or "val_mean_io_u"
    ):
        # fit related:
        self.sample_weight = sample_weight
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_decay_after_epoch = lr_decay_after_epoch
        self.lr_decay = lr_decay
        self.loss = loss
        self.monitor = monitor


class Setup:
    def __init__(
        self,
        pipeline_config=PipelineConfig(),
        net_config=NetConfig(),
        fit_config=FitConfig(),
        model_from_file=None,
    ):
        self.pipeline_config = pipeline_config
        self.net_config = net_config
        self.fit_config = fit_config
        self.model_from_file = model_from_file

    @property
    def dict(self):
        setup_dic = {}
        for i in self.__dict__:
            if i in ["pipeline_config", "net_config", "fit_config"]:
                dic = {}
                for j in self.__dict__[i].__dict__:
                    dic[j] = self.__dict__[i].__dict__[j]
                setup_dic[i] = dic
            else:
                setup_dic[i] = self.__dict__[i]

        return setup_dic

    