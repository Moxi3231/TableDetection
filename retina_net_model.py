import os
import box_util
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import encoder_decoder

from importlib import reload
import losses_for_model

reload(losses_for_model)
reload(box_util)
reload(encoder_decoder)


CombinedLoss = losses_for_model.CombinedLoss

compute_intersection_over_union, convert_to_min_max_corner = box_util.compute_intersection_over_union,box_util.convert_to_min_max_corner
AnchorBox = encoder_decoder.AnchorBox

BoxDecoder = encoder_decoder.BoxDecoder
LabelEncoder = encoder_decoder.LabelEncoder

# Retina Net :: Implementation
# From Keras
# https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/retinanet.ipynb


def get_backbone():
    """
    Returns ResNetX
    """
    backbone = keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3])

    #Original: Conv3_block4_out
    #For other backbone: Conv3_block3_out
    c3_out, c4_out, c5_out = [backbone.get_layer(lname).output for lname in [
        "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]]
    return keras.Model(inputs=[backbone.inputs], outputs=[c3_out, c4_out, c5_out])


class FeaturePyramid(keras.layers.Layer):
    """
    Feature Pyramid

    Input:  Backbone from which this layer will take input
    """

    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        ###                                 Filters,KernelSize,Stride,Padding
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output


def build_head(output_filters, bias_init):
    """
        Return Head
    """
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(keras.layers.Conv2D(
            256, 3, padding="same", kernel_initializer=kernel_init))
        head.add(keras.layers.ReLU())

    head.add(keras.layers.Conv2D(output_filters, 3, 1, padding="same",
             kernel_initializer=kernel_init, bias_initializer=bias_init))
    return head


class RetNet(keras.Model):
    """

    Initial Model
                    Image
                      |
       ----------------------------------
       |              |                 |
    Original        Processed       Processed
      Image          Image1           Image2
       |              |                 |
     ---------------------------------------
                   backbone       
   -------------------------------------------
       |              |                 |
      FPN1           FPN2             FPN3
       \              |               /
        \             |              /
         \            |             /
          \           |            /
           \          |           /
           ------------------------
             Class      |   Box
             Classifier | Regressor
           ------------------------


    Current Model: Retina Net With Only One Class
        Image --> Processed Image -> Backbone -> Feature Pyramid Network -> Class | Box
    
    """
    def __init__(self, num_classes=3,num_box_per_pixel = 9.0, backbone=None,decoder:BoxDecoder = None, *args, **kwargs):
        super(RetNet,self).__init__(name="RetNet", *args, **kwargs)
        self.feature_pyramid_net = FeaturePyramid(backbone=backbone)
        #self.feature_pyramid_net2 = FeaturePyramid(backbone=backbone)
        #self.feature_pyramid_net3 = FeaturePyramid(backbone=backbone)

        self.num_cls = num_classes
        #Classifier: Class to which particular anchor belong is to be predicted by cls_head
        self.cls_head = build_head(num_box_per_pixel*num_classes, tf.constant_initializer(-np.log((1-0.01)/0.01)))
        #Box Regressor: Predict the offset of the anchor
        self.box_head = build_head(num_box_per_pixel*4, "zeros")

        self._box_decoder = decoder if decoder is not None else BoxDecoder()
        self._anchor_box = AnchorBox()
       

    def call(self, images, training=False):
        
        features1 = self.feature_pyramid_net(images, training=training)
        #features2 = self.feature_pyramid_net2(image[:,1,:,:,:], training=training)
        #features3 = self.feature_pyramid_net3(image[:,2,:,:,:], training=training)

        image_shape = tf.cast(tf.shape(images),dtype=tf.float32)
        N = tf.cast(image_shape[0],dtype= tf.int32)
        cls_out, box_out = [], []

        #for f1, f2, f3 in zip(features1, features2, features3):
        #    feature = keras.layers.Average()([f1, f2, f3])
        #    box_out.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
        #    cls_out.append(tf.reshape(self.cls_head(
        #        feature), [N, -1, self.num_cls]))
        for f1 in features1:
            box_out.append(tf.reshape(self.box_head(f1),[N,-1,4]))
            cls_out.append(tf.reshape(self.cls_head(f1),[N,-1,self.num_cls]))

        cls_out = tf.concat(cls_out, axis=1)
        box_out = tf.concat(box_out, axis=1)

        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_out = self._box_decoder.decode_box_predictions_util(anchor_boxes[None, ...], box_out)

        return tf.concat([box_out, cls_out], axis=-1)


class PredictionDecoder(tf.keras.layers.Layer):


    """
    Decode the prediction generated by the model.
    After the model predicts the offset.
    This layer will further calculate the change required in anchor's position 
    and then will decode the anchor, as anchor were encoded.
    """
    def __init__(self, 
                confidence_threshold=0.05, 
                nms_iou_threshold=0.15, 
                max_detection_per_class=550, 
                max_dectection=1100,
                decoder:BoxDecoder = None, 
                **kwargs):

        super(PredictionDecoder, self).__init__(**kwargs)
        
        self._box_decoder = decoder if decoder is not None else BoxDecoder()

        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_per_class = max_detection_per_class
        self.max_detection = max_dectection
        self._anchor_box = AnchorBox()
       
    
    def call(self, images, predictions):
        #image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        #anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = convert_to_min_max_corner(predictions[:, :, :4])
        
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        #tf.print(tf.reduce_sum(cls_predictions),tf.shape(cls_predictions))
       
        #boxes = self._box_decoder.decode_box_predictions_util(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
                                    tf.expand_dims(box_predictions, axis=2), 
                                    cls_predictions, 
                                    self.max_detection_per_class, 
                                    self.max_detection, 
                                    self.nms_iou_threshold, 
                                    self.confidence_threshold, 
                                    clip_boxes=False)


def get_model():
    num_classes = 1
    #num_classes += 1
    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025,0.00005,0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500,2400,240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=learning_rate_boundaries, values=learning_rates)

    #learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(0.0009, 500000, 1e-06, power=0.90)
    resnetX_backbone = get_backbone()
    loss_fn = CombinedLoss(num_classes=num_classes)
    model = RetNet(num_classes,num_box_per_pixel=4, backbone=resnetX_backbone)

    #optimizer = tf.optimizers.Adam(learning_rate=learning_rate_fn, clipnorm=1.0, epsilon=1e-08, amsgrad=True)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)

    return model
