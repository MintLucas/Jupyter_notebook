#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:37:14 2018

@author: jon-liu
"""


import tensorflow as tf
import numpy as np
import keras.backend as K
import keras.engine as KE
import keras.layers as KL



def anchor_refinement(boxes, deltas):
    boxes = tf.cast(boxes, tf.float32)
    h = boxes[:, 2] - boxes[:, 0]
    w = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + h / 2
    center_x = boxes[:, 1] + w / 2

    center_y += deltas[:, 0] * h
    center_x += deltas[:, 1] * w
    h *= tf.exp(deltas[:, 2])
    w *= tf.exp(deltas[:, 3])
    
    y1 = center_y - h / 2
    x1 = center_x - w / 2
    y2 = center_y + h / 2
    x2 = center_x + w / 2
    boxes = tf.stack([y1, x1, y2, x2], axis=1)
    return boxes
    
def boxes_clip(boxes, window):
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    cliped = tf.concat([y1, x1, y2, x2], axis=1)
    cliped.set_shape((cliped.shape[0], 4))
    return cliped
    
def batch_slice(inputs, graph_fn, batch_size):
    if not isinstance(inputs, list):
        inputs = [inputs]
    output = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (list, tuple)):
            output_slice = [output_slice]
        output.append(output_slice)
    output = list(zip(*output))
    result = [tf.stack(o, axis=0) for o in output]
    if len(result)==1:
        result = result[0]
    return result
    

class proposal(KE.Layer):
    def __init__(self, proposal_count, nms_thresh, anchors, batch_size, config=None, **kwargs):
        super(proposal, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.anchors = anchors
        self.batch_size = batch_size
        self.config = config
        self.nms_thresh = nms_thresh
    
    def call(self, inputs):
        probs = inputs[0][:, :, 1]
        deltas = inputs[1]
        deltas = deltas*np.reshape(self.config.RPN_BBOX_STD_DEV, (1, 1, 4))
        prenms_num = min(self.anchors.shape[0], 100)
        idxs = tf.nn.top_k(probs, prenms_num).indices

        probs = batch_slice([probs, idxs], lambda x,y:tf.gather(x, y), self.batch_size)
        deltas = batch_slice([deltas, idxs], lambda x,y:tf.gather(x, y), self.batch_size)
        anchors = batch_slice([idxs], lambda x:tf.gather(self.anchors, x), self.batch_size)
        refined_boxes = batch_slice([anchors, deltas], lambda x,y:anchor_refinement(x,y), self.batch_size)
        H,W = self.config.image_size[:2]
        windows = np.array([0, 0, H, W]).astype(np.float32)
        cliped_boxes = batch_slice([refined_boxes], lambda x:boxes_clip(x, windows), self.batch_size)
        normalized_boxes = cliped_boxes / np.array([H, W, H, W])
        def nms(normalized_boxes, scores):
            idxs_ = tf.image.non_max_suppression(normalized_boxes, scores, self.proposal_count, self.nms_thresh)
            box = tf.gather(normalized_boxes, idxs_)
            pad_num = tf.maximum(self.proposal_count - tf.shape(normalized_boxes)[0],0)
            box = tf.pad(box, [(0, pad_num), (0,0)])
            return box
        proposals_ = batch_slice([normalized_boxes, probs], nms, self.batch_size)
        return proposals_
    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)
   
     
##########detection_targe      

def box_refinement_graph(boxes, gt_box):
    boxes = tf.cast(boxes, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)
    
    heght = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * heght 
    center_x = boxes[:, 1] + 0.5 * width 
    
    gt_h = gt_box[:, 2] - gt_box[:, 0]
    gt_w = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_h 
    gt_center_x = gt_box[:, 1] + 0.5 * gt_w 
    
    dy = (gt_center_y - center_y) / heght
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_h / heght)
    dw = tf.log(gt_w / width)
    deltas = tf.stack([dy, dx, dh, dw], axis=1)
    return deltas
    

def overlaps_graph(boxes1, boxes2):
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),[1,1,tf.shape(boxes2)[0]]),[-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    
    intersection = tf.maximum((y2 - y1),0) * tf.maximum((x2 - x1),0)
    union = (b1_y2 - b1_y1) * (b1_x2 - b1_x1) + (b2_y2 - b2_y1) * (b2_x2 - b2_x1) - intersection
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps
    
def trim_zero_graph(x, name=None):
    none_zeros = tf.cast(tf.reduce_sum(tf.abs(x), axis=1), tf.bool)
    result = tf.boolean_mask(x, none_zeros, name=name)
    return result, none_zeros


def detection_target_graph(proposals, gt_boxes, gt_class_id, config=None):
    proposals, _ = trim_zero_graph(proposals, name="trim_proposals")
    gt_boxes, none_zero = trim_zero_graph(gt_boxes, name="trim_boxes")
    gt_class_id = tf.boolean_mask(gt_class_id, none_zero, name="trim_class_ids")
    
    overlap = overlaps_graph(proposals, gt_boxes)
    iou_max = tf.reduce_max(overlap, axis=1)
    iou_gt_max = tf.argmax(overlap, axis=0)
    positive_mask = (iou_max > 0.5)
    negative_mask = (iou_max < 0.5)
    positive_idxs = tf.where(positive_mask)[:, 0]
    negative_idxs = tf.where(negative_mask)[:, 0]
    
    num_positive = int(config.num_proposals_train *  config.num_proposals_ratio)
    positive_idxs = tf.random_shuffle(positive_idxs)[:num_positive]
    positive_idxs = tf.concat([positive_idxs, iou_gt_max], axis=0)
    positive_idxs = tf.unique(positive_idxs)[0]
    num_positive = tf.shape(positive_idxs)[0]
    
    r = 1 / config.num_proposals_ratio
    num_negative = tf.cast(r * tf.cast(num_positive, tf.float32), tf.int32) - num_positive
    negative_idxs = tf.random_shuffle(negative_idxs)[:num_negative]
        
    positive_roi = tf.gather(proposals, positive_idxs)
    negative_roi = tf.gather(proposals, negative_idxs)
    
    postive_overlap = tf.gather(overlap, positive_idxs)
    gt_assignment = tf.argmax(postive_overlap, axis=1)
    gt_boxes = tf.gather(gt_boxes, gt_assignment)
    gt_class_id = tf.gather(gt_class_id, gt_assignment)
    
    deltas = box_refinement_graph(positive_roi, gt_boxes)
    deltas /= config.RPN_BBOX_STD_DEV
    
    rois = tf.concat([positive_roi, negative_roi], axis=0)
    N = tf.shape(negative_roi)[0]
    P = config.num_proposals_train - tf.shape(rois)[0]
    deltas = tf.pad(deltas, [(0, P+N), (0, 0)])
    gt_boxes = tf.pad(gt_boxes, [(0, P+N),(0, 0)])
    gt_class_id = tf.pad(gt_class_id, [(0, P+N)])
    rois = tf.pad(rois, [(0, P),(0, 0)])
    return rois, deltas, gt_boxes, gt_class_id
    
    
class detection_target(KE.Layer):
    def __init__(self, config, batch_size, **kwargs):
        super(detection_target, self).__init__(**kwargs)
        self.config = config
        
    def call(self, inputs):
        proposals = inputs[0]
        gt_bboxes = inputs[1]
        gt_class_ids = inputs[2]

        outs = batch_slice([proposals, gt_bboxes, gt_class_ids], 
                                                                 lambda x, y, z:detection_target_graph(x, y, z, self.config), self.config.batch_size)   
        return outs
        
    def compute_output_shape(self, input_shape):
        return [(None, self.config.num_proposals_train, 4),
                (None, self.config.num_proposals_train, 4),
                (None, self.config.num_proposals_train, 4),
                (None, 1)]
                
    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]




    

##### ########### feature map classifier

def roi_pooling_cell(feature_map, rois, num_rois, pool_size, num_channels):
    feature_map = K.expand_dims(feature_map, 0)
    out_rois = []
    for i in range(num_rois):
        y1 = rois[i, 0]
        x1 = rois[i, 1]
        y2 = rois[i, 2]
        x2 = rois[i, 3]
        
        y2 = y1 + tf.maximum(1.0, y2-y1)
        x2 = x1 + tf.maximum(1.0, x2-x1)
        y1 = K.cast(y1, 'int32')
        x1 = K.cast(x1, 'int32')
        y2 = K.cast(y2, 'int32')
        x2 = K.cast(x2, 'int32')
        out_roi = tf.image.resize_images(feature_map[:, y1:y2, x1:x2, :], [pool_size, pool_size])
        out_rois.append(out_roi)
    final_out = K.concatenate(out_rois, axis=0)
    final_out = K.reshape(final_out, [-1, num_rois, pool_size, pool_size, num_channels])
    return final_out
    

class roi_pooling(KE.Layer):
    def __init__(self, num_rois, pool_size, config, **kwargs):
        self.num_rois = num_rois
        self.pool_size = pool_size
        self.config = config
        super(roi_pooling, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.num_channel = input_shape[0][3]
    
    def call(self, inputs):
        assert len(inputs) == 2
        feature_map = inputs[0]
        rois = inputs[1]
        out = batch_slice([feature_map, rois], lambda x,y:roi_pooling_cell(x, y, self.num_rois, self.pool_size, self.num_channel), self.config.batch_size)
        out = K.reshape(out,(-1, self.num_rois, self.pool_size, self.pool_size, self.num_channel))
        return out
    
    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.num_channel
    
    def get_config(self):
        config = {"num_rois", self.num_rois,
                  "pool_size",self.pool_size}
        base_config = super(roi_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   
    

class BatchNorm(KL.BatchNormalization):

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)
        
def header_classifier(feature_map, rois, num_rois, pool_size, num_classes, config):
    x = roi_pooling(num_rois, pool_size, config)([feature_map, rois])
    x = KL.TimeDistributed(KL.Conv2D(512, (pool_size, pool_size), padding="valid"), name="header_classifier_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3), name="header_classifier_bn1")(x)  
    x = KL.Activation("relu")(x)
    
    x = KL.TimeDistributed(KL.Conv2D(1024, (1,1), padding="valid"), name="header_classifier_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3), name="header_classifier_bn2")(x)  
    x = KL.Activation("relu")(x)
    
    base = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="head_classifier_squeeze")(x)
    frcnn_class = KL.TimeDistributed(KL.Dense(num_classes), name="head_classifier_class")(base)
    frcnn_prob = KL.TimeDistributed(KL.Activation("softmax"), name="head_classifier_prob")(frcnn_class)
    
    frcnn_fc = KL.TimeDistributed(KL.Dense(4*num_classes, activation="linear"), name="head_classifier_fc")(base)
    s = K.int_shape(frcnn_fc)
    frcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="fpn_class_deltas")(frcnn_fc)
    return frcnn_class, frcnn_prob, frcnn_bbox


    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    