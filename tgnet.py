import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, './utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module_spider, pointnet_fp_module,spiderConv
from crfrnn_layer import CrfRnnLayer


def get_model(point_cloud, is_training, bn_decay, part_num, batch_size, num_point, weight_decay):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz
    data_format = 'NHWC'
    num_class = 10


    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module_spider(l0_xyz, l0_points, npoint=2048, radius=0.1, nsample=12, mlp=[32,32,64], mlp2=[32,32,64], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', pooling ='max')
    
    #layer 2
    l2_xyz, l2_points, l2_indices = pointnet_sa_module_spider(l1_xyz, l1_points, npoint=1024, radius=0.2, nsample=12, mlp=[64,64,128], mlp2=[64,64,128], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2', pooling ='max')
    
    #layer 3
    l3_xyz, l3_points, l3_indices = pointnet_sa_module_spider(l2_xyz, l2_points, npoint=512, radius=0.4, nsample=12, mlp=[128,128,128], mlp2=[128,128,256], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3', pooling ='max')

    #layer 
    l4_xyz, l4_points, l4_indices = pointnet_sa_module_spider(l3_xyz, l3_points, npoint=256, radius=0.8, nsample=12, mlp=[128,256,512], mlp2=[256,256,512], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4', pooling ='max')


    # Feature Propagation layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,128], is_training, bn_decay, scope='fa_layer2')   
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [128,128], is_training = is_training, bn_decay = bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training = is_training, bn_decay = bn_decay, scope='fa_layer4')



    #l0 points
    offset = l0_xyz
    units = [32, 64, 64]
    for i, num_out_channel in enumerate(units):
        offset = tf_util.conv1d(offset, num_out_channel, 1, padding='VALID', bn=True, is_training=is_training, scope='convf%d'%(i), bn_decay=bn_decay) 
    l0_points = tf.concat([l0_points,offset], axis=2)
    out_max = tf_util.max_pool2d(tf.expand_dims(l0_points,2), [num_point, 1], padding='VALID', scope='maxpool')
    out_max = tf.tile(out_max, [1, num_point, 1, 1])
    l0_points = tf.concat([l0_points,tf.squeeze(out_max)], axis=2)

    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc11', bn_decay=bn_decay)
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')


    net = CrfRnnLayer(image_dims=(batch_size, num_point),
                         num_classes=num_class,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=2,
                         name='crfrnn')([net, point_cloud])

    return tf.squeeze(net)




    



def get_loss(pred, label):
    """ pred: B,N,13
        label: B,N """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    return tf.reduce_mean(loss)



if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
