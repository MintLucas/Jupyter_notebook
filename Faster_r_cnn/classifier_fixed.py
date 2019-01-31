#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 17:54:02 2018

@author: jon-liu
"""

import tensorflow as tf
import numpy as np
import keras.engine as KE
import utils
import keras.backend as K

import keras.layers as KL


def roi_pooling(feature_map, rois, batch_size, num_rois, pool_size):

    def crop_graph(feature_map, box):
        y1,x1,y2,x2 = box[0], box[1], box[2], box[3]
        y1, x1, y2, x2 = tf.cast(y1, tf.int32), tf.cast(x1, tf.int32), tf.cast(y2, tf.int32), tf.cast(x2, tf.int32)
        croped_map = feature_map[:, y1:y2, x1:x2, :]
        croped_map = KL.Lambda(lambda x: tf.image.resize_images(x, [pool_size, pool_size], method=1), name="crop_resize")(croped_map)
        return croped_map

    def crop_graph_oneBatch(feature_map, boxes, batch_size):
        croped_map = utils.batch_slice(boxes, lambda x: crop_graph(feature_map,x), batch_size=batch_size)
        croped_map = K.squeeze(croped_map, 1)
        return croped_map

    def crop_graph_Batches(feature_map, boxes, batch_size, num_rois):
        croped_map = utils.batch_slice(boxes, lambda x: crop_graph_oneBatch(feature_map,x, num_rois), \
                                       batch_size=batch_size)
        return croped_map
    croped_map = crop_graph_Batches(feature_map, rois, batch_size, num_rois)
    croped_map = KL.Lambda(lambda x: 1*x)(croped_map)
    return croped_map

def roi_poolingV2(feature_map, rois, batch_size, num_rois, pool_size):
    

    def crop_graph(feature_map, box):
        y1,x1,y2,x2 = box[0], box[1], box[2], box[3]
        #y1, x1, y2, x2 = tf.cast(y1, tf.int32), tf.cast(x1, tf.int32), tf.cast(y2, tf.int32), tf.cast(x2, tf.int32)
        y1, x1, y2, x2 = K.cast(y1, "int32"), K.cast(x1, "int32"), K.cast(y2, "int32"), K.cast(x2, "int32")
        croped_map = feature_map[:, y1:y2, x1:x2, :]
        croped_map = KL.Lambda(lambda x: tf.image.resize_images(x, [pool_size, pool_size], method=1), name="crop_resize")(croped_map)
        #croped_map = tf.image.resize_images(croped_map, pool_size, method=1)
        return croped_map

    def crop_graph_oneBatch(feature_map, boxes, batch_size):
        croped_map = utils.batch_slice(boxes, lambda x: crop_graph(feature_map,x), batch_size=batch_size)
        croped_map = K.squeeze(croped_map, 1)
        return croped_map

    def crop_graph_Batches(feature_map, boxes, batch_size, num_rois):
        croped_map = utils.batch_slice(boxes, lambda x: crop_graph_oneBatch(feature_map,x, num_rois), \
                                       batch_size=batch_size)
        return croped_map
    croped_map = crop_graph_Batches(feature_map, rois, batch_size, num_rois)
    croped_map = KL.Lambda(lambda x: tf.reshape(x, [batch_size, num_rois, pool_size, pool_size, -1]))(croped_map)
    #croped_map = KL.Lambda(lambda x: 1*x)(croped_map)
    return croped_map

class roi_pooling_graph(KE.Layer):
    def __init__(self, batch_size, num_rois, pool_size, **kwargs):
        super(roi_pooling_graph, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_rois = num_rois
        #self.pool_size = pool_size
        self.pool_size = pool_size
    
    def call(self, inputs):
        feature_map = inputs[0]
        rois = inputs[1]
        out = roi_poolingV2(feature_map, rois, self.batch_size, self.num_rois, self.pool_size)
        out = K.permute_dimensions(out, (0, 1, 2, 3, 4))
        return out
    
    def compute_out_shape(self, input_shape):
        #return [None, None, self.pool_size, self.pool_size, input_shape[0][-1]]
        return None, self.num_rois, self.pool_size, self.pool_size, input_shape[0][-1]
    
class BatchNorm(KL.BatchNormalization):

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)

import tensorflow as tf
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

class RoiPoolingConv(KE.Layer):

    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            y1 = rois[0, roi_idx, 0]
            x1 = rois[0, roi_idx, 1]
            y2 = rois[0, roi_idx, 2]
            x2 = rois[0, roi_idx, 3]
          
            #row_length = w / float(self.pool_size)
            #col_length = h / float(self.pool_size)
            x2 = x1 + K.maximum(1.0,x2-x1)
            y2 = y1 + K.maximum(1.0,y2-y1)
            
            num_pool_regions = self.pool_size

         
            y1 = K.cast(y1, 'int32')
            x1 = K.cast(x1, 'int32')
            y2 = K.cast(y2, 'int32')
            x2 = K.cast(x2, 'int32')

            rs = tf.image.resize_images(img[:, y1:y2, x1:x2, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return final_output
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   
        
#########################################V2

def roi_pooling_onebacth(img, rois, num_rois, pool_size, nb_channels):
    img = K.expand_dims(img, 0)
    outputs = []
    for roi_idx in range(num_rois):

        y1 = rois[roi_idx, 0]
        x1 = rois[roi_idx, 1]
        y2 = rois[roi_idx, 2]
        x2 = rois[roi_idx, 3]

        x2 = x1 + K.maximum(1.0,x2-x1)
        y2 = y1 + K.maximum(1.0,y2-y1)

        y1 = K.cast(y1, 'int32')
        x1 = K.cast(x1, 'int32')
        y2 = K.cast(y2, 'int32')
        x2 = K.cast(x2, 'int32')

        rs = tf.image.resize_images(img[:, y1:y2, x1:x2, :], (pool_size, pool_size))
        outputs.append(rs)

    final_output = K.concatenate(outputs, axis=0)
    final_output = K.reshape(final_output, (-1, num_rois, pool_size, pool_size, nb_channels))
    return final_output

class RoiPoolingConvV2(KE.Layer):

    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConvV2, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        out = utils.batch_slice([img, rois], \
                                lambda x,y: roi_pooling_onebacth(x,y,self.num_rois, self.pool_size, self.nb_channels), \
                                batch_size=20)
        out = K.reshape(out, (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return out

    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   

    
#经过roipooling增加了一维 batch_size,roi_nums,w,h,feature_map_depth,将roi_nums变为时间步修正
def fpn_classifiler(feature_map, rois, batch_size, num_rois, pool_size, num_classes):
    #x = roi_pooling(feature_map, rois, batch_size, num_rois, pool_size)
    #x = roi_pooling_graph(batch_size, num_rois, pool_size)([feature_map, rois])
    x = RoiPoolingConvV2(7, num_rois)([feature_map, rois])
    x = KL.TimeDistributed(KL.Conv2D(512, pool_size, padding="valid"),
                           name="mrcnn_class_conv1")(x)#pool_size最后后变为一个像素
    
    x = KL.TimeDistributed(BatchNorm(axis=3), name="fpn_classifier_bn0")(x)
    x = KL.Activation("relu")(x)
    
    x = KL.TimeDistributed(KL.Conv2D(512, (1, 1), padding="valid"), name="fpn_classifier_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3), name="fpn_classifier_bn1")(x)
    x = KL.Activation("relu")(x)
    
    
    base = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),name="fpn_classifier_squeeze")(x)#宽和高压缩为1纬
    #base = KL.Lambda(lambda x: 1*x, name="convert_to_keras")(base_)
    
    class_logits = KL.TimeDistributed(KL.Dense(num_classes), name="fpn_classifier_logits")(base)
    class_prob = KL.TimeDistributed(KL.Activation("softmax"), name="fpn_classifier_prob")(class_logits)
    
    class_fc = KL.TimeDistributed(KL.Dense(4*num_classes, activation='linear'), name="fpn_classifier_fc")(base)
    s = K.int_shape(class_fc)
    class_bbox = KL.Reshape((s[1], num_classes, 4), name="fpn_class_deltas")(class_fc)

    return class_logits, class_prob, class_bbox
    
    

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    