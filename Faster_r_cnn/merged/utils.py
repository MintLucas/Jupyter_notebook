#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 10:36:23 2018

@author: jon-liu
"""

import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random
import cv2
import keras.engine as KE

       
def anchor_gen(featureMap_size, ratios, scales, rpn_stride, anchor_stride):
    ratios, scales = np.meshgrid(ratios, scales)
    ratios, scales = ratios.flatten(), scales.flatten()
    
    width = scales / np.sqrt(ratios)
    height = scales * np.sqrt(ratios)
    
    shift_x = np.arange(0, featureMap_size[0], anchor_stride) * rpn_stride
    shift_y = np.arange(0, featureMap_size[1], anchor_stride) * rpn_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    centerX, anchorX = np.meshgrid(shift_x, width)
    centerY, anchorY = np.meshgrid(shift_y, height)
    boxCenter = np.stack([centerY, centerX], axis=2).reshape(-1, 2)
    boxSize = np.stack([anchorX, anchorY], axis=2).reshape(-1, 2)
    
    boxes = np.concatenate([boxCenter - 0.5 * boxSize, boxCenter + 0.5 * boxSize], axis=1)
    return boxes

def compute_iou(box, boxes, area, areas):
    y1 = np.maximum(box[0], boxes[:, 0])
    x1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[2], boxes[:, 2])
    x2 = np.minimum(box[3], boxes[:, 3])
    interSec = np.maximum(y2-y1, 0) * np.maximum(x2-x1, 0)
    union = areas[:] + area - interSec 
    iou = interSec / union
    return iou


def compute_overlap(boxes1, boxes2):
    areas1 = (boxes1[:,3] - boxes1[:,1]) * (boxes1[:,2] - boxes1[:,0])
    areas2 = (boxes2[:,3] - boxes2[:,1]) * (boxes2[:,2] - boxes2[:,0])
    overlap = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(boxes2.shape[0]):
        box = boxes2[i]
        overlap[:,i] = compute_iou(box, boxes1, areas2[i], areas1)
    return overlap
    

def anchors_refinement(boxes, deltas):
    #boxes = boxes.astype(np.float32)
    boxes = tf.cast(boxes, tf.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return tf.stack([y1, x1, y2, x2], axis=1)
    
    
def clip_boxes_graph(boxes, window):

    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


def non_max_suppression(boxes, scores, nms_threshold):
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    areas = (y2 - y1) * (x2 - x1)
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        ix = idxs[0]
        ious = compute_iou(boxes[ix], boxes[idxs[1:]], areas[ix], areas[idxs[1:]])
        keep.append(ix)
        remove_idxs = np.where(ious > nms_threshold)[0] + 1
        idxs = np.delete(idxs, remove_idxs)
        idxs = np.delete(idxs, 0)
    return np.array(keep, dtype=np.int32)
    
#真实的boxes,和预测anchors之间计算IOU    
def build_rpnTarget(boxes, anchors, config):
    rpn_match = np.zeros(anchors.shape[0],dtype=np.int32)
    rpn_bboxes = np.zeros((config.train_rois_num, 4))
    
    iou = compute_overlap(anchors, boxes)
    maxArg_iou = np.argmax(iou, axis=1)
    max_iou = iou[np.arange(iou.shape[0]), maxArg_iou]
    postive_anchor_idxs = np.where(max_iou > 0.4)[0]
    negative_anchor_idxs = np.where(max_iou < 0.1)[0]
    
    rpn_match[postive_anchor_idxs]=1
    rpn_match[negative_anchor_idxs]=-1
    maxIou_anchors = np.argmax(iou, axis=0)
    rpn_match[maxIou_anchors] =1
    
    #每个anchor设置三种label
    ids = np.where(rpn_match==1)[0]
    extral = len(ids) - config.train_rois_num // 2
    if extral > 0:
        ids_ = np.random.choice(ids, extral, replace=False)
        rpn_match[ids_] = 0
   
    ids = np.where(rpn_match==-1)[0]
    extral = len(ids) - ( config.train_rois_num - np.where(rpn_match==1)[0].shape[0])
    if extral > 0:
        ids_ = np.random.choice(ids, extral, replace=False)
        rpn_match[ids_] = 0
    
    #对label为1的计算偏差delta
    idxs = np.where(rpn_match==1)[0]
    ix = 0
    for i, a in zip(idxs, anchors[idxs]):
        gt = boxes[maxArg_iou[i]]
        
        gt_h = gt[2] - gt[0]#gt真实的box
        gt_w = gt[3] - gt[1]
        gt_centy = gt[0] + 0.5 * gt_h
        gt_centx = gt[1] + 0.5 * gt_w

        a_h = a[2] - a[0]#a anchors
        a_w = a[3] - a[1]
        a_centy = a[0] + 0.5 * a_h
        a_centx = a[1] + 0.5 * a_w
        
        #中心点相差除以长度算出小数,log计算两个边的偏差
        rpn_bboxes[ix] = [(gt_centy - a_centy)/a_h, (gt_centx - a_centx)/a_w,
                         np.log(gt_h / a_h), np.log(gt_w / a_w)]
        #归一化,不然算出的loss太小deata百分数对其影响不大
        rpn_bboxes[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bboxes
 
def batch_slice(inputs, graph_fn, batch_size, names=None):
    
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
        
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]
    return result

class proposalLayer(KE.Layer):
    def __init__(self, anchors, proposal_count, nms_thrshold, batch_size, config=None,  **kwargs):
        super(proposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.anchors = anchors.astype(np.float32)
        self.nms_thrshold = nms_thrshold
        self.batch_size = batch_size
    
    def call(self, inputs):
        scores = inputs[0][:,:,1]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        pre_nms_count = min(self.anchors.shape[0], 500)
        idxs = tf.nn.top_k(scores, pre_nms_count).indices

        scores = batch_slice([scores, idxs],lambda x,y: tf.gather(x,y),self.batch_size)
        anchors = batch_slice([idxs],lambda x: tf.gather(self.anchors ,x),self.batch_size)     
        deltas = batch_slice([deltas, idxs],lambda x,y: tf.gather(x,y),self.batch_size)
        

        box_refined = batch_slice([anchors, deltas],lambda x, y: anchors_refinement(x, y),self.batch_size)
        
        H, W = self.config.image_size[:2]
        window = np.array([0, 0, H, W]).astype(np.float32)
        
        box_refined = batch_slice([box_refined],lambda x: clip_boxes_graph(x, window),self.batch_size)
        
        normalize_box = box_refined / np.array([H, W, H, W])
        
        def nms(normalize_box, scores):
            indices = tf.image.non_max_suppression(normalize_box, scores, self.proposal_count, self.nms_thrshold)
            box = tf.gather(normalize_box, indices)
            pad_count = tf.maximum(self.proposal_count - tf.shape(box)[0],0)
            box = tf.pad(box,[(0, pad_count),(0,0)])
            return box
        box = batch_slice([normalize_box, scores],nms,self.batch_size)
        return box
    
    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)

    

class shapeData():
    def __init__(self, image_size, config):
        self.image_size = image_size
#        self.num_image = num_image
        self.config = config
        
    def load_data(self):
        images = np.zeros((self.image_size[0], self.image_size[1], 3))
#        bboxs = []
#        ids = []
#        rpn_match = []
#        rpn_bboxes = []
        anchors = anchor_gen(self.config.featureMap_size, self.config.ratios, self.config.scales, self.config.rpn_stride, self.config.anchor_stride)

        images, bboxs, ids = self.random_image(self.image_size)
        rpn_match, rpn_bboxes = build_rpnTarget(bboxs, anchors, self.config)
        return images, bboxs, ids, rpn_match, rpn_bboxes, anchors
        
    def random_image(self, image_size):
        typeDict = {'square':1, 'circle':2, 'triangle':3}
        H,W = image_size[0], image_size[1]
        #image = np.random.randn(H, W, 3)
        red = np.ones((64,64,1))*30
        green = np.ones((64,64,1))*60
        blue = np.ones((64,64,1))*90
        image = np.concatenate([red, green, blue], axis=2)
        num_obj = random.sample([1,2,3,4], 1)[0]
        #num_obj = 1                     
        bboxs = np.zeros((num_obj, 4))
        Ids = np.zeros((num_obj, 1))
        shapes = []
        dims = np.zeros((num_obj, 3))
        for i in range(num_obj):
            shape = random.sample(list(typeDict), 1)[0]
            shapes.append(shape)
            
            Ids[i] = typeDict[shape]
            x, y = np.random.randint(H//4, W//4 + W//2, 1)[0], np.random.randint(H//4, W//4 + W//2, 1)[0]
            #x, y = 32, 64
            s = np.random.randint(H//16, W//8, 1)[0]
            #s = 12
            dim = x, y, s
            dims[i]=dim
            #color = random.randint(1,255)
            #image = self.draw_shape(image, shape, dims, color)
            bboxs[i] = self.draw_boxes(dim)
        keep_idxs = non_max_suppression(bboxs, np.arange(num_obj), 0.01)
        bboxs = bboxs[keep_idxs]
        Ids = Ids[keep_idxs]
        shapes = [shapes[i] for i in keep_idxs]
        dims = dims[keep_idxs]
        for j in range(bboxs.shape[0]):
            color = random.randint(1,255)
            shape = shapes[j]
            dim = dims[j]
            image = self.draw_shape(image, shape, dim, color)
        return image, bboxs, Ids
    
    def draw_shape(self, image, shape, dims, color):
        x, y, s = dims.astype(np.int32)
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                    ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def draw_boxes(self, dims):
        x, y, s = dims
        bbox = [x-s, y-s, x+s, y+s]
        bbox = np.array(bbox)
        return bbox
        
################################################################  DetectionTargetLayer

#def box_refinement_graph(boxes, gt_box):
#    boxes = tf.cast(boxes, tf.float32)
#    gt_box = tf.cast(gt_box, tf.float32)
#    
#    height = boxes[:, 2] - boxes[:, 0]
#    width = boxes[:, 3] - boxes[:, 1]
#    center_y = boxes[:, 0] + 0.5 * height 
#    cneter_x = boxes[:, 1] + 0.5 * width
#    
#    gt_height = boxes[:, 2] - boxes[:, 0]
#    gt_width = boxes[:, 3] - boxes[:, 1]
#    gt_center_y = boxes[:, 0] + 0.5 * height 
#    gt_cneter_x = boxes[:, 1] + 0.5 * width
#    
#    dy = (gt_center_y - center_y) / height
#    dx = (gt_cneter_x - cneter_x) / width
#    dh = tf.log(gt_height / height)
#    dw = tf.log(gt_width / width)
#    result = tf.stack([dy, dx, dh, dw], axis=1)
#    return result
#
#def overlaps_graph(boxes1, boxes2):
#    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),[1,1,tf.shape(boxes2)[0]]),[-1, 4])
#    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
#    
#    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
#    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
#    
#    y1 = tf.maximum(b1_y1, b2_y1)
#    x1 = tf.maximum(b1_x1, b2_x1)
#    y2 = tf.minimum(b1_y2, b2_y2)
#    x2 = tf.minimum(b1_x2, b2_x2)
#    
#    intersection = tf.maximum((y2 - y1),0) * tf.maximum((x2 - x1),0)
#    union = (b1_y2 - b1_y1) * (b1_x2 - b1_x1) + (b2_y2 - b2_y1) * (b2_x2 - b2_x1) - intersection
#    iou = intersection / union
#    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
#    return overlaps
#
#
#def trim_zeros_graph(boxes, name=None):
#    none_zero = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
#    boxes = tf.boolean_mask(boxes,none_zero, name=name)
#    return boxes, none_zero
#
#def detection_target_graph(proposals, gt_class_ids, gt_bboxes, config):
#    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
#    gt_bboxes, none_zeros = trim_zeros_graph(gt_bboxes, name="trim_bboxes")
#    gt_class_ids = tf.boolean_mask(gt_class_ids, none_zeros)
#    
#    overlaps = overlaps_graph(proposals, gt_bboxes)
#    max_iouArg = tf.reduce_max(overlaps, axis=1)
#    max_iouGT = tf.argmax(overlaps, axis=0)
#    
#    positive_mask = (max_iouArg > 0.5)
#    positive_idxs = tf.where(positive_mask)[:,0]
#    negative_idxs = tf.where(max_iouArg < 0.5)[:,0]
#    
#    num_positive = int(config.num_proposals_train *  config.num_proposals_ratio)
#    positive_idxs = tf.random_shuffle(positive_idxs)[:num_positive]
#    positive_idxs = tf.concat([positive_idxs, max_iouGT], axis=0)
#    positive_idxs = tf.unique(positive_idxs)[0]
#    
#    num_positive = tf.shape(positive_idxs)[0]
#    
#    r = 1 / config.num_proposals_ratio
#    num_negative = tf.cast(r * tf.cast(num_positive, tf.float32), tf.int32) - num_positive
#    negative_idxs = tf.random_shuffle(negative_idxs)[:num_negative]
#    
#    positive_rois = tf.gather(proposals, positive_idxs)
#    negative_rois = tf.gather(proposals, negative_idxs)
#
#    positive_overlap = tf.gather(overlaps, positive_idxs)
#    
#    gt_assignment = tf.argmax(positive_overlap, axis=1)
#    gt_bboxes = tf.gather(gt_bboxes, gt_assignment)
#    gt_class_ids = tf.gather(gt_class_ids, gt_assignment)
#    
#    deltas = box_refinement_graph(positive_rois, gt_bboxes)
#    #deltas = utils.anchor_deltas(positive_rois, gt_bboxes)
#    rois = tf.concat([positive_rois, negative_rois], axis=0)
#    
#    N = tf.shape(negative_rois)[0]
#    P = config.num_proposals_train - tf.shape(rois)[0]
#    
#    rois = tf.pad(rois,[(0,P),(0,0)])
#    gt_class_ids = tf.pad(gt_class_ids, [(0, N+P)])
#    deltas = tf.pad(deltas,[(0,N+P),(0,0)])
#    gt_bboxes = tf.pad(gt_bboxes,[(0,N+P),(0,0)])
#    
#    return rois, gt_class_ids, deltas, gt_bboxes 
#    
#    
#class DetectionTarget(KE.Layer):
#    
#    def __init__(self, config, **kwargs):
#        super(DetectionTarget, self).__init__(**kwargs)
#        self.config = config
#        
#    def call(self, inputs):
#        proposals = inputs[0]
#        gt_class_ids = inputs[1]
#        gt_bboxes = inputs[2]
#        
#        names = ["rois", "target_class_ids", "target_deltas","target_bbox"]
#        outputs = batch_slice([proposals, gt_class_ids, gt_bboxes],
#                            lambda x,y,z: detection_target_graph(x, y, z, self.config), self.config.batch_size, names=names)
#        return outputs
#    
#    def compute_output_shape(self, input_shape):
#        return [(None, self.config.num_proposals_train, 4),
#                (None, 1),
#                (None, self.config.num_proposals_train, 4),
#                (None, self.config.num_proposals_train, 4)]
#                
#    def compute_mask(self, inputs, mask=None):
#        return [None, None, None, None]


def box_refinement_graph(boxes, gt_box):
    boxes = tf.cast(boxes, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)
    
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height 
    cneter_x = boxes[:, 1] + 0.5 * width
    
    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height 
    gt_cneter_x = gt_box[:, 1] + 0.5 * gt_width
    
    dy = (gt_center_y - center_y) / height
    dx = (gt_cneter_x - cneter_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)
    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result

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


def trim_zeros_graph(boxes, name=None):
    none_zero = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes,none_zero, name=name)
    return boxes, none_zero

def detection_target_graph(proposals, gt_class_ids, gt_bboxes, config):
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_bboxes, none_zeros = trim_zeros_graph(gt_bboxes, name="trim_bboxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, none_zeros)
    
    overlaps = overlaps_graph(proposals, gt_bboxes)
    max_iouArg = tf.reduce_max(overlaps, axis=1)
    max_iouGT = tf.argmax(overlaps, axis=0)
    
    positive_mask = (max_iouArg > 0.5)
    positive_idxs = tf.where(positive_mask)[:,0]
    negative_idxs = tf.where(max_iouArg < 0.5)[:,0]
    
    num_positive = int(config.num_proposals_train *  config.num_proposals_ratio)
    positive_idxs = tf.random_shuffle(positive_idxs)[:num_positive]
    positive_idxs = tf.concat([positive_idxs, max_iouGT], axis=0)
    positive_idxs = tf.unique(positive_idxs)[0]
    
    num_positive = tf.shape(positive_idxs)[0]
    
    r = 1 / config.num_proposals_ratio
    num_negative = tf.cast(r * tf.cast(num_positive, tf.float32), tf.int32) - num_positive
    negative_idxs = tf.random_shuffle(negative_idxs)[:num_negative]
    
    positive_rois = tf.gather(proposals, positive_idxs)
    negative_rois = tf.gather(proposals, negative_idxs)

    positive_overlap = tf.gather(overlaps, positive_idxs)
    
    gt_assignment = tf.argmax(positive_overlap, axis=1)
    gt_bboxes = tf.gather(gt_bboxes, gt_assignment)
    gt_class_ids = tf.gather(gt_class_ids, gt_assignment)
    
    deltas = box_refinement_graph(positive_rois, gt_bboxes)
    deltas /= config.RPN_BBOX_STD_DEV
    #deltas = utils.anchor_deltas(positive_rois, gt_bboxes)
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    
    N = tf.shape(negative_rois)[0]
    P = config.num_proposals_train - tf.shape(rois)[0]
    
    rois = tf.pad(rois,[(0,P),(0,0)])
    gt_class_ids = tf.pad(gt_class_ids, [(0, N+P)])
    deltas = tf.pad(deltas,[(0,N+P),(0,0)])
    gt_bboxes = tf.pad(gt_bboxes,[(0,N+P),(0,0)])
    
    return rois, gt_class_ids, deltas, gt_bboxes 
    
    
class DetectionTarget(KE.Layer):
    
    def __init__(self, config, **kwargs):
        super(DetectionTarget, self).__init__(**kwargs)
        self.config = config
        
    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_bboxes = inputs[2]
        
        names = ["rois", "target_class_ids", "target_deltas","target_bbox"]
        outputs = batch_slice([proposals, gt_class_ids, gt_bboxes],
                            lambda x,y,z: detection_target_graph(x, y, z, self.config), self.config.batch_size, names=names)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return [(None, self.config.num_proposals_train, 4),
                (None, 1),
                (None, self.config.num_proposals_train, 4),
                (None, self.config.num_proposals_train, 4)]
                
    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]














     