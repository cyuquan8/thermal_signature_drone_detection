# standard imports
import tensorflow as tf
import os

""" 
Convolutional Neural Network model classes for YOLO_V3 using TensorFlow 2.4.0
Purpose : efficiently generate model architecture by building from subclasses
"""

class darknet_conv2d_block(tf.keras.layers.Layer):
    
    def __init__(self, filters, kernel_size, strides, weight_decay, activation = True, batch_norm = True):
        
        """ class constructor that creates the layers attributes for darknet_conv2d_block """
        
        # inherit class constructor attributes from tf.keras.layers.Layer
        super(darknet_conv2d_block, self).__init__()
        
        # boolean for batch norm and activation 
        self.batch_norm = batch_norm
        self.activation = activation
        
        # store stride attributes
        self.strides = strides
        
        # determine padding given stride
        if strides == 1:
            
            # retain input size 
            padding = 'same'
        
        else:
            
            # top left padding for darknet
            self.padding_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
            
            # do not attempt to conserve input size
            padding = 'valid'
            
            
        # add conv2d layer attribute
        self.conv2d_block = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, 
                                                   padding = padding, 
                                                   kernel_regularizer = tf.keras.regularizers.l2(l = weight_decay))
        
        # add batch norm layer attribute
        self.batch_norm_block = tf.keras.layers.BatchNormalization()
    
    def call(self, inputs, training = False):
        
        """ function for forward pass of model """
        """ includes training argument as batch_norm functions differently during training and during inference """
        
        # pad if stride is not 1
        if self.strides != 1:
            
            # pad inputs
            x = self.padding_layer(inputs)
        
            # inputs --> conv2d_block
            x = self.conv2d_block(x)
        
        else:
            
            # inputs --> conv2d_block
            x = self.conv2d_block(inputs)
        
        # check boolean for batch_norm
        if self.batch_norm == True:
            
            # darknet_conv2d_block --> batch_norm
            x = self.batch_norm_block(x, training = training)
        
        # check boolean for activation
        if self.activation == True:
        
            # batch_norm --> leaky relu
            x = tf.nn.leaky_relu(x)
        
        return x
    
class darknet_res_block(tf.keras.layers.Layer):
    
    def __init__(self, filters, kernel_size, strides, weight_decay):
        
        """ class constructor that creates the layers attributes for darknet_res_block """
        
        # inherit class constructor attributes from tf.keras.layers.Layer
        super(darknet_res_block, self).__init__()
        
        # add darknet conv2d block attributes 
        self.darknet_conv2d_block_1 = darknet_conv2d_block(filters = filters[0], kernel_size = kernel_size[0], 
                                                           strides = strides[0], weight_decay = weight_decay[0])
        self.darknet_conv2d_block_2 = darknet_conv2d_block(filters = filters[1], kernel_size = kernel_size[1], 
                                                           strides = strides[1], weight_decay = weight_decay[1])
        
    def call(self, inputs, training = False):
        
        """ function for forward pass of model """
        """ includes training argument as batch_norm functions differently during training and duing inference """
        
        # inputs --> darknet_conv2d_block_1 --> darknet_conv2d_block_2 --> output + residual (input)
        residual = inputs
        x = self.darknet_conv2d_block_1(inputs)
        x = self.darknet_conv2d_block_2(x)
        x = tf.add(x, residual)
        
        return x 
    
class darknet_block(tf.keras.layers.Layer):
    
    def __init__(self, num_of_blocks, filters, kernel_size, conv_strides, conv_weight_decay, res_strides, 
                 res_weight_decay):
        
        """ class constructor that creates the layers attributes for darknet_block """
        
        # inherit class constructor attributes from tf.keras.layers.Layer
        super(darknet_block, self).__init__()
        
        # single conv block to half input size
        self.darknet_conv2d_block = darknet_conv2d_block(filters = filters[1], kernel_size = kernel_size[1], 
                                                         strides = conv_strides, weight_decay = conv_weight_decay)
        
        # list of darknet_res_blocks
        self.darknet_res_block_list = []
        
        # append darknet_res_block to darknet_res_block_list for stated number of blocks
        for _ in range(num_of_blocks):
            
            # append darknet_res_block
            self.darknet_res_block_list.append(darknet_res_block(filters = filters, kernel_size = kernel_size, 
                                                                 strides = res_strides, 
                                                                 weight_decay = res_weight_decay))
        
    def call(self, inputs, training = False):
        
        """ function for forward pass of model """
        """ includes training argument as batch_norm functions differently during training and duing inference """
        
        # inputs --> darknet_conv2d_block 
        x = self.darknet_conv2d_block(inputs)
        
        # conv2d_block_1 --> darknet_res_block 
        for i in range(len(self.darknet_res_block_list)):
            
            # pass input through respective darknet_res_block
            x = self.darknet_res_block_list[i](x)
        
        return x 

class darknet(tf.keras.layers.Layer):
    
    def __init__(self):
        
        """ class constructor that creates the layers attributes for darknet """
        
        # inherit class constructor attributes from tf.keras.layers.Layer
        super(darknet, self).__init__()
        
        # single darknet conv block
        self.darknet_conv2d_block = darknet_conv2d_block(filters = 32, kernel_size = 3, strides = 1, 
                                                         weight_decay = 0)
        
        # darknet_blocks in darknet 53
        self.darknet_block_1 = darknet_block(num_of_blocks = 1, filters = [32, 64], kernel_size = [1, 3], 
                                             conv_strides = 2, conv_weight_decay = 0, res_strides = [1, 1], 
                                             res_weight_decay = [0, 0])
        self.darknet_block_2 = darknet_block(num_of_blocks = 2, filters = [64, 128], kernel_size = [1, 3], 
                                             conv_strides = 2, conv_weight_decay = 0, res_strides = [1, 1], 
                                             res_weight_decay = [0, 0])
        self.darknet_block_3 = darknet_block(num_of_blocks = 8, filters = [128, 256], kernel_size = [1, 3], 
                                             conv_strides = 2, conv_weight_decay = 0, res_strides = [1, 1], 
                                             res_weight_decay = [0, 0])
        self.darknet_block_4 = darknet_block(num_of_blocks = 8, filters = [256, 512], kernel_size = [1, 3], 
                                             conv_strides = 2, conv_weight_decay = 0, res_strides = [1, 1], 
                                             res_weight_decay = [0, 0])
        self.darknet_block_5 = darknet_block(num_of_blocks = 4, filters = [512, 1024], kernel_size = [1, 3], 
                                             conv_strides = 2, conv_weight_decay = 0, res_strides = [1, 1], 
                                             res_weight_decay = [0, 0])
    
    def call(self, inputs, training = False):
        
        """ function for forward pass of model """
        """ includes training argument as batch_norm functions differently during training and duing inference """
        
        # inputs --> darknet_conv2d_block
        x = self.darknet_conv2d_block(inputs)
        
        # darknet_conv2d_block --> darknet_block_1
        x = self.darknet_block_1(x)
        
        # darknet_block_1 --> darknet_block_2
        x = self.darknet_block_2(x)
        
        # darknet_block_2 --> darknet_block_3
        x = self.darknet_block_3(x)
        
        # downsample output at stride 8
        stride_8 = x 
        
        # darknet_block_3 --> darknet_block_4
        x = self.darknet_block_4(x)
        
        # downsample output at stride 16
        stride_16 = x 
        
        # darknet_block_4 --> darknet_block_5
        stride_32 = self.darknet_block_5(x)
        
        return stride_8, stride_16, stride_32

class yolo_conv_block(tf.keras.layers.Layer): 
    
    def __init__(self, filters, kernel_size, strides, weight_decay):
        
        """ class constructor that creates the layers attributes for yolo_conv_block """

        # inherit class constructor attributes from tf.keras.layers.Layer
        super(yolo_conv_block, self).__init__()

        # upsampling attributes
        self.darknet_conv2d_block_1 = darknet_conv2d_block(filters = filters[0], kernel_size = kernel_size[0], 
                                                           strides = strides, weight_decay = weight_decay[0])
        self.upsampling = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.concatenate = tf.keras.layers.Concatenate()

        # add darknet conv2d block attributes 
        self.darknet_conv2d_block_2 = darknet_conv2d_block(filters = filters[0], kernel_size = kernel_size[0], 
                                                           strides = strides, weight_decay = weight_decay[0])
        self.darknet_conv2d_block_3 = darknet_conv2d_block(filters = filters[1], kernel_size = kernel_size[1], 
                                                           strides = strides, weight_decay = weight_decay[1])
        self.darknet_conv2d_block_4 = darknet_conv2d_block(filters = filters[0], kernel_size = kernel_size[0], 
                                                           strides = strides, weight_decay = weight_decay[0])
        self.darknet_conv2d_block_5 = darknet_conv2d_block(filters = filters[1], kernel_size = kernel_size[1], 
                                                           strides = strides, weight_decay = weight_decay[1])
        self.darknet_conv2d_block_6 = darknet_conv2d_block(filters = filters[0], kernel_size = kernel_size[0], 
                                                           strides = strides, weight_decay = weight_decay[0])
    
    def call(self, inputs, training = False):
        
        """ function for forward pass of model """
        """ includes training argument as batch_norm functions differently during training and duing inference """

        # check if input is a tuple
        if isinstance(inputs, tuple):

            # inputs comprises of x and prior downsamped input
            x, x_skip = inputs

            # x --> darknet_conv2d_block_1
            x = self.darknet_conv2d_block_1(x)

            # darknet_conv2d_block_1 --> upsample
            x = self.upsampling(x)

            # upsample --> concatenate
            x = self.concatenate([x, x_skip])

        else:

            x = inputs

        # x --> darknet_conv2d_block_2 --> darknet_conv2d_block_3 --> darknet_conv2d_block_4 --> darknet_conv2d_block_5
        # --> darknet_conv2d_block_6
        x = self.darknet_conv2d_block_2(x)
        x = self.darknet_conv2d_block_3(x)
        x = self.darknet_conv2d_block_4(x)
        x = self.darknet_conv2d_block_5(x)
        x = self.darknet_conv2d_block_6(x)

        return x 
    
class yolo_output_block(tf.keras.layers.Layer):
    
    def __init__(self, filters, kernel_size, strides, weight_decay, num_of_anchor_bbox, classes):
        
        """ class constructor that creates the layers attributes for yolo_output_block """

        # inherit class constructor attributes from tf.keras.layers.Layer
        super(yolo_output_block, self).__init__()

        # single darknet conv2d block
        self.darknet_conv2d_block_1 = darknet_conv2d_block(filters = filters, kernel_size = kernel_size[0], 
                                                           strides = strides[0], weight_decay = weight_decay[0])

        # output block
        self.darknet_conv2d_block_2 = darknet_conv2d_block(filters = num_of_anchor_bbox * (classes + 5), 
                                                           kernel_size = kernel_size[0], strides = strides[1], 
                                                           weight_decay = weight_decay[1], activation = False, 
                                                           batch_norm = False)
    
    def call(self, inputs, training = False):
        
        """ function for forward pass of model """
        """ includes training argument as batch_norm functions differently during training and duing inference """

        # x --> darknet_conv2d_block_1
        x = self.darknet_conv2d_block_1(inputs)

        # darknet_conv2d_block_1 --> darknet_conv2d_block_2
        x = self.darknet_conv2d_block_2(x)

        return x
    
class yolo_v3(tf.keras.Model):

    def __init__(self, num_of_anchor_bbox, classes, checkpoint_dir, model_name):
        
        """ class constructor that creates the layers attributes for yolo_v3 """
        
        # inherit class constructor attributes from tf.keras.Model
        super(yolo_v3, self).__init__()
        
        # checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        
        # checkpoint filepath 
        self.checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        
        # initialise darknet 53 attribute
        self.darknet_53 = darknet()
        
        # intialise yolo_conv_blocks
        self.yolo_conv_block_1 = yolo_conv_block(filters = [512, 1024], kernel_size = [1, 3], strides = 1, 
                                                 weight_decay = [0, 0])
        self.yolo_conv_block_2 = yolo_conv_block(filters = [256, 512], kernel_size = [1, 3], strides = 1, 
                                                 weight_decay = [0, 0])
        self.yolo_conv_block_3 = yolo_conv_block(filters = [128, 256], kernel_size = [1, 3], strides = 1, 
                                                 weight_decay = [0, 0])
        
        # intialise yolo_output_blocks
        self.yolo_output_block_1 = yolo_output_block(filters = 1024, kernel_size = [3, 1], strides = [1, 1], 
                                                     weight_decay = [0, 0], num_of_anchor_bbox = num_of_anchor_bbox, 
                                                     classes = classes)
        self.yolo_output_block_2 = yolo_output_block(filters = 512, kernel_size = [3, 1], strides = [1, 1], 
                                                     weight_decay = [0, 0], num_of_anchor_bbox = num_of_anchor_bbox, 
                                                     classes = classes)
        self.yolo_output_block_3 = yolo_output_block(filters = 256, kernel_size = [3, 1], strides = [1, 1], 
                                                     weight_decay = [0, 0], num_of_anchor_bbox = num_of_anchor_bbox, 
                                                     classes = classes)
        
    def call(self, inputs, training = False):
        
        """ function for forward pass of model """
        """ includes training argument as batch_norm functions differently during training and duing inference """
        
        # inputs --> darknet_53
        stride_8, stride_16, stride_32 = self.darknet_53(inputs)
        
        # stride_32 --> self.yolo_conv_block_1
        stride_32 = self.yolo_conv_block_1(stride_32)
        
        # yolo_conv_block_1 --> yolo_output_block_1
        output_1 = self.yolo_output_block_1(stride_32)
        
        # (self.yolo_conv_block_1 + stride_16) --> self.yolo_conv_block_2
        stride_32 = self.yolo_conv_block_2((stride_32, stride_16))
        
        # yolo_conv_block_2 --> yolo_output_block_2
        output_2 = self.yolo_output_block_2(stride_32)
        
        # (self.yolo_conv_block_2 + stride_8) --> self.yolo_conv_block_3
        stride_32 = self.yolo_conv_block_3((stride_32, stride_8))
        
        # yolo_conv_block_3 --> yolo_output_block_3
        output_3 = self.yolo_output_block_3(stride_32)
        
        return [output_3, output_2, output_1]
