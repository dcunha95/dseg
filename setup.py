import json
import tensorflow as tf

class PipelineConfig:
    def __init__(
        self,
        split=[0.6, 0.2, 0.2],
        dataset_percent=0.1,
        print_options=["raw", "output", "input", "input-original", "gt", "gt-original", "channels", "contour", "3d"],
        name_format=["Average", "Name"],
        **kwargs,
    ):

        # pipeline related:
        self.split = split
        self.dataset_percent = dataset_percent
        self.print_options = print_options  # print options: [raw, output, input, input_original, gt, gt_original]
        self.name_format = name_format

    def __repr__(self) -> str:
        return str(self.__dict__)


class NetConfig:
    def __init__(
        self,
        model_type="unet",
        depth=4,
        pool_size=2,
        concat_all=True,
        node_type=4,
        image_size=(512, 512),
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
        **kwargs,
    ):
        # Net related:
        self.model_type = model_type
        self.depth = depth
        self.pool_size = pool_size
        self.concat_all = concat_all
        self.node_type = node_type
        self.image_size = tuple(image_size)
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

    def __repr__(self) -> str:
        return str(self.__dict__)

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
        **kwargs,
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

    def __repr__(self) -> str:
        return str(self.__dict__)


class TensorFlowConfig:
    def __init__(
        self,
        mirrored_strategy = None,
        mixed_precision = None,
        auto_clustering = None,
        **kwargs,
    ):

        # test_variable = tf.Variable(1.)
        # self.mirrored_strategy = str(type(test_variable)) == "<class 'tensorflow.python.distribute.values.MirroredVariable'>"
        # self.mixed_precision = tf.keras.mixed_precision.global_policy()._name
        # self.auto_clustering = tf.config.optimizer.get_jit()
        # del test_variable

        if mirrored_strategy is None:
            test_variable = tf.Variable(1.)
            mirrored_strategy = str(type(test_variable)) == "<class 'tensorflow.python.distribute.values.MirroredVariable'>"
            del test_variable

        if mixed_precision is None:
            mixed_precision = tf.keras.mixed_precision.global_policy()._name

        if auto_clustering is None:
            auto_clustering = tf.config.optimizer.get_jit()
            

        self.mirrored_strategy = mirrored_strategy
        self.mixed_precision = mixed_precision
        self.auto_clustering = auto_clustering

    def __repr__(self) -> str:
        return str(self.__dict__)


class Setup:
    def __init__(
        self,
        pipeline_config=None,
        net_config=None,
        fit_config=None,
        tf_config=None,
        model_from_file=None,
        base_training_info=None,
        notes="",
        **kwargs,
    ):

        if pipeline_config is None:
            pipeline_config=PipelineConfig()
        
        if net_config is None:
            net_config=NetConfig()

        if fit_config is None:
            fit_config=FitConfig()

        if tf_config is None:
            tf_config=TensorFlowConfig()

        self.pipeline_config = pipeline_config
        self.net_config = net_config
        self.fit_config = fit_config
        self.tf_config = tf_config
        
        self.model_properties = {}
        
        self.model_from_file = model_from_file
        self.base_training_info = base_training_info
        self.notes = notes


    @property
    def to_dict(self):
        setup_dic = {}
        for i in self.__dict__:
            if i in ["pipeline_config", "net_config", "fit_config", "tf_config"]:
                dic = {}
                for j in self.__dict__[i].__dict__:
                    dic[j] = self.__dict__[i].__dict__[j]
                setup_dic[i] = dic
            else:
                setup_dic[i] = self.__dict__[i]

        return setup_dic

    def __repr__(self):

        d = self.to_dict
        sl = []
        for i in ["pipeline_config", "net_config", "fit_config", "tf_config", "model_properties"]:
            sl.append("".join([i, "\n"]))
            
            for j in d[i]:
                sl.append("".join( ["\n\t", str(j), ": ", str(d[i][j])] ))

            sl.append("\n\n")
        
        sl.append("".join(["model_from_file: ", str(d["model_from_file"])]))
        sl.append("".join(["\nbase_training_info: ", str(d["base_training_info"])]))
        sl.append("".join(["\nnotes: ", str(d["notes"])]))
        
        return "".join(sl)

    def __eq__(self, other):

        if self.__class__ == other.__class__:
            return self.to_dict == other.to_dict
        else:
            return False

    @staticmethod
    def from_dict(d):
        
        if 'pipeline_config' in d:
            pipeline_config = PipelineConfig(**d["pipeline_config"]) 
        else:
            pipeline_config = None

        if 'net_config' in d:
            net_config = NetConfig(**d["net_config"]) 
        else:
            net_config = None

        if 'fit_config' in d:
            fit_config = FitConfig(**d["fit_config"]) 
        else:
            fit_config = None

        if 'tf_config' in d:
            tf_config = TensorFlowConfig(**d["tf_config"]) 
        else:
            tf_config = None

        if "model_from_file" in d:
            model_from_file = d["model_from_file"]
        else:
            model_from_file = None
        
        if "base_training_info" in d:
            base_training_info = d['base_training_info']
        else:
            base_training_info = None
        
        if "base_training_info" in d:
            notes = d['notes']
        else:
            notes = ""

        setup = Setup(
            pipeline_config=pipeline_config, 
            net_config=net_config, 
            fit_config=fit_config, 
            tf_config=tf_config,
            model_from_file=model_from_file,
            base_training_info=base_training_info,
            notes=notes,
        )
        
        if "model_properties" in d:
            setup.model_properties = d["model_properties"]
 
        return setup

    @staticmethod
    def from_json(path):

        with open(path) as d:
            loaded_dict = json.load(d)

        instance = Setup.from_dict(loaded_dict)

        return instance

    def to_json(self, path):
        with open(path, "w") as d:
            json.dump(self.to_dict, d, indent=4)    
