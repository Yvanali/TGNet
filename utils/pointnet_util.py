""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, './utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
import math

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    N = idx.shape[1]
    view_shape = list(idx.shape)
    #print('---------------------------------', view_shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = tf.reshape(tf.range(B),[B,1])#[24,1]
    batch_indices = tf.tile(batch_indices,repeat_shape)#[24,1024]
    Indices = tf.stack((batch_indices, idx),axis=2)

    #new_points = points[batch_indices, idx, :]
    #zz = np.zeros(shape=(idx.shape[0], 1), dtype=np.int32)
    #Indices = np.hstack((zz, idx))

    new_points = tf.gather_nd(points, Indices)
    return new_points

def pointSIFT_group(radius, xyz, points, use_xyz=True):
    idx = pointSIFT_select(xyz, radius)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 8, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, 8, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 8/32, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return xyz, new_points, idx, grouped_xyz

def pairwise_distance(point_cloud):
  """Compute pairwise distance of a point cloud.

  Args:
    point_cloud: tensor (batch_size, num_points, num_dims)

  Returns:
    pairwise distance: (batch_size, num_points, num_points)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)
    
  point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
  point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
  point_cloud_inner = -2*point_cloud_inner
  point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
  point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
  return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knns(adj_matrix, k=20):
  """Get KNN based on the pairwise distance.
  Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int

  Returns:
    nearest neighbors: (batch_size, num_points, k)
  """
  neg_adj = -adj_matrix
  _, nn_idx = tf.nn.top_k(neg_adj, k=k)
  return nn_idx

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    idxf = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, idxf) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    point_cloud_xyz = tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    grouped_xyz = grouped_xyz-point_cloud_xyz
    #idxf = tf.expand_dims(idxf, 2)
    if points is not None:
        #new_point = tf.tile(group_point(points, idxf), [1,1,nsample,1])
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        #grouped_points = tf.concat([new_point, grouped_points-new_point], axis=-1)
        #=================
        #_, neigh_points = tf.split(grouped_points, [nsample-1,1], 2)
        #point_cloud_central = tf.tile(tf.expand_dims(new_point, axis=-2), [1, 1, nsample, 1])  
        #point_cloud_neigh_points = tf.tile(neigh_points, [1, 1, nsample, 1])   
        #grouped_points = tf.concat([point_cloud_central, grouped_points-point_cloud_central], axis=-1)
        #==============
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,2,3,1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat

 
def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1




def pointnet_sa_module_spider(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope,  pooling, bn=True, knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NHWC' 
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)


        #------------------------------------------------------------------------
        #print('-----------------', edge_feature)
        batch_size = grouped_xyz.get_shape()[0].value
        num_point = grouped_xyz.get_shape()[1].value
        K_knn = grouped_xyz.get_shape()[2].value
        in_channels = new_points.get_shape()[3].value
        shape = [1, 1, 1, 3]
        shape1 = [1, 1, 1, 1,1]
        num_gau = 10

        X = grouped_xyz[:, :, :, 0]
        Y = grouped_xyz[:, :, :, 1]
        Z = grouped_xyz[:, :, :, 2]



        X = tf.expand_dims(X, -1)#[x, 1]
        Y = tf.expand_dims(Y, -1)
        Z = tf.expand_dims(Z, -1)
       
        #var = grouped_xyz*grouped_xyz


        initializer = tf.contrib.layers.xavier_initializer()
      
        w_x = tf.tile(tf_util._variable_on_cpu('weight_x', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_y = tf.tile(tf_util._variable_on_cpu('weight_y', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_z = tf.tile(tf_util._variable_on_cpu('weight_z', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_xyz = tf.tile(tf_util._variable_on_cpu('weight_xyz', shape, initializer), [batch_size, num_point, K_knn, 1])
      
        w_xy = tf.tile(tf_util._variable_on_cpu('weight_xy', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_yz = tf.tile(tf_util._variable_on_cpu('weight_yz', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_xz = tf.tile(tf_util._variable_on_cpu('weight_xz', shape, initializer), [batch_size, num_point, K_knn, 1])
        biases = tf.tile(tf_util._variable_on_cpu('biases', shape, 
                        tf.constant_initializer(0.0)), [batch_size, num_point, K_knn, 1])
      
        w_xx = tf.tile(tf_util._variable_on_cpu('weight_xx', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_yy = tf.tile(tf_util._variable_on_cpu('weight_yy', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_zz = tf.tile(tf_util._variable_on_cpu('weight_zz', shape, initializer), [batch_size, num_point, K_knn, 1])

        w_xxy = tf.tile(tf_util._variable_on_cpu('weight_xxy', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_xyy = tf.tile(tf_util._variable_on_cpu('weight_xyy', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_xxz = tf.tile(tf_util._variable_on_cpu('weight_xxz', shape, initializer), [batch_size, num_point, K_knn, 1])

        w_xzz = tf.tile(tf_util._variable_on_cpu('weight_xzz', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_yyz = tf.tile(tf_util._variable_on_cpu('weight_yyz', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_yzz = tf.tile(tf_util._variable_on_cpu('weight_yzz', shape, initializer), [batch_size, num_point, K_knn, 1])

      
        w_xxx = tf.tile(tf_util._variable_on_cpu('weight_xxx', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_yyy = tf.tile(tf_util._variable_on_cpu('weight_yyy', shape, initializer), [batch_size, num_point, K_knn, 1])
        w_zzz = tf.tile(tf_util._variable_on_cpu('weight_zzz', shape, initializer), [batch_size, num_point, K_knn, 1])

        biases1 = tf.tile(tf_util._variable_on_cpu('biases1', shape,tf.constant_initializer(0.0)), [batch_size, num_point, K_knn,1])
        biases2 = tf.tile(tf_util._variable_on_cpu('biases2', shape,tf.constant_initializer(0.0)), [batch_size, num_point, K_knn, 1])
        biases3 = tf.tile(tf_util._variable_on_cpu('biases3', shape,tf.constant_initializer(0.0)), [batch_size, num_point, K_knn, 1])
        biases4 = tf.tile(tf_util._variable_on_cpu('biases4', shape,tf.constant_initializer(0.0)), [batch_size, num_point, K_knn, 1])
        biases5 = tf.tile(tf_util._variable_on_cpu('biases5', shape,tf.constant_initializer(0.0)), [batch_size, num_point, K_knn, 1])
        biases6 = tf.tile(tf_util._variable_on_cpu('biases6', shape,tf.constant_initializer(0.0)), [batch_size, num_point, K_knn,1])
        biases7 = tf.tile(tf_util._variable_on_cpu('biases7', shape,tf.constant_initializer(0.0)), [batch_size, num_point, K_knn, 1])
        biases8 = tf.tile(tf_util._variable_on_cpu('biases8', shape,tf.constant_initializer(0.0)), [batch_size, num_point, K_knn, 1])
        biases9 = tf.tile(tf_util._variable_on_cpu('biases9', shape,tf.constant_initializer(0.0)), [batch_size, num_point, K_knn, 1])
        biases10 = tf.tile(tf_util._variable_on_cpu('biases10', shape,initializer), [batch_size, num_point, K_knn, 1])
        biases11 = tf.tile(tf_util._variable_on_cpu('biases11', shape,initializer), [batch_size, num_point, K_knn,1])
        biases12 = tf.tile(tf_util._variable_on_cpu('biases12', shape,initializer), [batch_size, num_point, K_knn, 1])
        biases13 = tf.tile(tf_util._variable_on_cpu('biases13', shape,initializer), [batch_size, num_point, K_knn, 1])
        biases14 = tf.tile(tf_util._variable_on_cpu('biases14', shape,initializer), [batch_size, num_point, K_knn, 1])
        biases15 = tf.tile(tf_util._variable_on_cpu('biases15', shape,initializer), [batch_size, num_point, K_knn, 1])
        biases16 = tf.tile(tf_util._variable_on_cpu('biases16', shape,initializer), [batch_size, num_point, K_knn,1])
        biases17 = tf.tile(tf_util._variable_on_cpu('biases17', shape,initializer), [batch_size, num_point, K_knn, 1])
        biases18 = tf.tile(tf_util._variable_on_cpu('biases18', shape,initializer), [batch_size, num_point, K_knn, 1])
        biases19 = tf.tile(tf_util._variable_on_cpu('biases19', shape,initializer), [batch_size, num_point, K_knn, 1])
        biases20 = tf.tile(tf_util._variable_on_cpu('biases20', shape,initializer), [batch_size, num_point, K_knn, 1])


        g1 = w_x * X + w_y * Y + w_z * Z + w_xyz * X * Y * Z+ biases
        g2 = w_xy * X * Y + w_yz * Y * Z + w_xz * X * Z 
        g3 = w_xx * X * X + w_yy * Y * Y + w_zz * Z * Z
        g4 = w_xxy * X * X * Y + w_xyy * X * Y * Y + w_xxz * X * X * Z
        g5 = w_xzz * X * Z * Z + w_yyz * Y * Y * Z + w_yzz * Y * Z * Z
        g6 = w_xxx * X * X * X + w_yyy * Y * Y * Y + w_zzz * Z * Z * Z
        g_d = g1 + g2 + g3 + g4 + g5 + g6

        #paris_Lille
        #g_d = g1

        g_d1 = tf.exp(-0.5*(g_d-biases1)*(g_d-biases1)/(biases11*biases11))
        g_d2 = tf.exp(-0.5*(g_d-biases2)*(g_d-biases2)/(biases12*biases12))
        g_d3 = tf.exp(-0.5*(g_d-biases3)*(g_d-biases3)/(biases13*biases13))
        g_d4 = tf.exp(-0.5*(g_d-biases4)*(g_d-biases4)/(biases14*biases14))
        g_d5 = tf.exp(-0.5*(g_d-biases5)*(g_d-biases5)/(biases15*biases15))
        g_d6 = tf.exp(-0.5*(g_d-biases6)*(g_d-biases6)/(biases16*biases16))
        g_d7 = tf.exp(-0.5*(g_d-biases7)*(g_d-biases7)/(biases17*biases17))
        g_d8 = tf.exp(-0.5*(g_d-biases8)*(g_d-biases8)/(biases18*biases18))
        g_d9 = tf.exp(-0.5*(g_d-biases9)*(g_d-biases9)/(biases19*biases19))
        g_d10 = tf.exp(-0.5*(g_d-biases10)*(g_d-biases10)/(biases20*biases20))

        '''g_d1 = tf.exp((g_d-biases1))
        g_d2 = tf.exp((g_d-biases2))
        g_d3 = tf.exp((g_d-biases3))
        g_d4 = tf.exp((g_d-biases4))
        g_d5 = tf.exp((g_d-biases5))
        g_d6 = tf.exp((g_d-biases6))
        g_d7 = tf.exp((g_d-biases7))
        g_d8 = tf.exp((g_d-biases8))
        g_d9 = tf.exp((g_d-biases9))
        g_d10 = tf.exp((g_d-biases10))'''

        
        g_d1 = tf.expand_dims(g_d1, 3)#[batch_size, num_point, K_knn, 1, 1]
        g_d2 = tf.expand_dims(g_d2, 3)#[batch_size, num_point, K_knn, 1, 1]
        g_d3 = tf.expand_dims(g_d3, 3)#[batch_size, num_point, K_knn, 1, 1]
        g_d4 = tf.expand_dims(g_d4, 3)#[batch_size, num_point, K_knn, 1, 1]
        g_d5 = tf.expand_dims(g_d5, 3)#[batch_size, num_point, K_knn, 1, 1]
        g_d6 = tf.expand_dims(g_d6, 3)#[batch_size, num_point, K_knn, 1, 1]
        g_d7 = tf.expand_dims(g_d7, 3)#[batch_size, num_point, K_knn, 1, 1]
        g_d8 = tf.expand_dims(g_d8, 3)#[batch_size, num_point, K_knn, 1, 1]
        g_d9 = tf.expand_dims(g_d9, 3)#[batch_size, num_point, K_knn, 1, 1]
        g_d10 = tf.expand_dims(g_d10, 3)#[batch_size, num_point, K_knn, 1, 1]
        g_d1 = tf.tile(g_d1, [1, 1, 1, in_channels, 1])#[batch_size, num_point, K_knn, in_channels, 1]
        g_d2 = tf.tile(g_d2, [1, 1, 1, in_channels, 1])
        g_d3 = tf.tile(g_d3, [1, 1, 1, in_channels, 1])
        g_d4 = tf.tile(g_d4, [1, 1, 1, in_channels, 1])
        g_d5 = tf.tile(g_d5, [1, 1, 1, in_channels, 1])
        g_d6 = tf.tile(g_d6, [1, 1, 1, in_channels, 1])
        g_d7 = tf.tile(g_d7, [1, 1, 1, in_channels, 1])
        g_d8 = tf.tile(g_d8, [1, 1, 1, in_channels, 1])
        g_d9 = tf.tile(g_d9, [1, 1, 1, in_channels, 1])
        g_d10 = tf.tile(g_d10, [1, 1, 1, in_channels, 1])
        new_points = tf.expand_dims(new_points, -1)
        new_points = new_points*g_d1+new_points*g_d2+new_points*g_d3+new_points*g_d4+new_points*g_d5+new_points*g_d6+new_points*g_d7+new_points*g_d8+new_points*g_d9+new_points*g_d10
        new_points = tf.reshape(new_points, [batch_size, num_point, K_knn, in_channels*3])

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                exp_dists = tf_util.conv2d(tf.transpose(exp_dists, [0,1,3,2]), K_knn, [1,1], padding='VALID', bn=True, is_training=is_training, scope='weighted', bn_decay=bn_decay) 
                exp_dists = tf.transpose(exp_dists, [0,1,3,2])
                weights = exp_dists/(tf.reduce_sum(exp_dists,axis=2,keep_dims=True)+1e-8) # (batch_size, npoint, nsample, 1)
                new_points1 = new_points
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
                avg_points_max = tf.reduce_max(new_points1, axis=[2], keep_dims=True, name='avgpool')
                new_points = tf.concat([new_points, avg_points_max], axis=-1)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)


        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1]) 
        '''_, new_points1, _, _ = pointSIFT_group(radius, new_xyz, new_points, use_xyz=False)
        new_points1 = tf.concat([tf.tile(tf.expand_dims(new_points,2),[1,1,8,1]), new_points1-tf.tile(tf.expand_dims(new_points,2),[1,1,8,1])], axis=-1)
        
        # Point Feature Embedding
        if use_nchw: new_points1 = tf.transpose(new_points1, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='convl%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        if use_nchw: new_points1 = tf.transpose(new_points1, [0,2,3,1])
        new_points1 = tf.reduce_max(new_points1, axis=[2], keep_dims=False, name='maxpool')'''
        #new_points = tf.concat([new_points, new_points1], axis=-1)

        return new_xyz, new_points, idx
 
def pointnet_fp_module_geo(xyz1, xyz2, points1, points2, geo_xyz, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)
        interpolated_points_geo_xyz = three_interpolate(geo_xyz, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1,interpolated_points_geo_xyz



def spiderConv(grouped_points,
              feat,
              mlp,
              taylor_channel,
              bn=False,
              is_training=None,
              bn_decay=None,
              gn=False,
              is_multi_GPU=False,
              activation_fn=tf.nn.relu,
              scope='taylor'):
  """ 2D convolution with non-linear operation.

  Args:
    feat: 3-D tensor variable BxNxC
    idx: 3-D tensor variable BxNxk
    delta: 4-D tensor variable BxNxkx3
    num_conv: int
    taylor_channel: int    
    bn: bool, whether to use batch norm
    is_training: bool Tensor variable
    bn_decay: float or float tensor variable in [0,1]
    gn: bool, whether to use group norm
    G: int
    is_multi_GPU: bool, whether to use multi GPU
    activation_fn: function
    scope: string
    

  Returns:
    feat: 3-D tensor variable BxNxC
  """
  with tf.variable_scope(scope) as sc:

      batch_size = feat.get_shape()[0].value
      num_point = feat.get_shape()[1].value
      in_channels = grouped_points.get_shape()[2].value
      shape = [1, 1, taylor_channel]

      X = feat[:, :, 0]
      Y = feat[:, :, 1]
      Z = feat[:, :, 2]

      X = tf.expand_dims(X, -1)#[x, 1]
      Y = tf.expand_dims(Y, -1)
      Z = tf.expand_dims(Z, -1)

      #initialize
      initializer = tf.contrib.layers.xavier_initializer()
      
      w_x = tf.tile(tf_util._variable_on_cpu('weight_x', shape, initializer), [batch_size, num_point, 1])
      w_y = tf.tile(tf_util._variable_on_cpu('weight_y', shape, initializer), [batch_size, num_point, 1])
      w_z = tf.tile(tf_util._variable_on_cpu('weight_z', shape, initializer), [batch_size, num_point, 1])
      w_xyz = tf.tile(tf_util._variable_on_cpu('weight_xyz', shape, initializer), [batch_size, num_point, 1])
      
      w_xy = tf.tile(tf_util._variable_on_cpu('weight_xy', shape, initializer), [batch_size, num_point, 1])
      w_yz = tf.tile(tf_util._variable_on_cpu('weight_yz', shape, initializer), [batch_size, num_point, 1])
      w_xz = tf.tile(tf_util._variable_on_cpu('weight_xz', shape, initializer), [batch_size, num_point, 1])
      biases = tf.tile(tf_util._variable_on_cpu('biases', shape, 
                        tf.constant_initializer(0.0)), [batch_size, num_point, 1])
      
      w_xx = tf.tile(tf_util._variable_on_cpu('weight_xx', shape, initializer), [batch_size, num_point, 1])
      w_yy = tf.tile(tf_util._variable_on_cpu('weight_yy', shape, initializer), [batch_size, num_point, 1])
      w_zz = tf.tile(tf_util._variable_on_cpu('weight_zz', shape, initializer), [batch_size, num_point, 1])

      w_xxy = tf.tile(tf_util._variable_on_cpu('weight_xxy', shape, initializer), [batch_size, num_point, 1])
      w_xyy = tf.tile(tf_util._variable_on_cpu('weight_xyy', shape, initializer), [batch_size, num_point, 1])
      w_xxz = tf.tile(tf_util._variable_on_cpu('weight_xxz', shape, initializer), [batch_size, num_point, 1])

      w_xzz = tf.tile(tf_util._variable_on_cpu('weight_xzz', shape, initializer), [batch_size, num_point, 1])
      w_yyz = tf.tile(tf_util._variable_on_cpu('weight_yyz', shape, initializer), [batch_size, num_point, 1])
      w_yzz = tf.tile(tf_util._variable_on_cpu('weight_yzz', shape, initializer), [batch_size, num_point, 1])

      
      w_xxx = tf.tile(tf_util._variable_on_cpu('weight_xxx', shape, initializer), [batch_size, num_point, 1])
      w_yyy = tf.tile(tf_util._variable_on_cpu('weight_yyy', shape, initializer), [batch_size, num_point, 1])
      w_zzz = tf.tile(tf_util._variable_on_cpu('weight_zzz', shape, initializer), [batch_size, num_point, 1])

      alpha0 = tf.tile(tf_util._variable_on_cpu('alpha0', shape, initializer), [batch_size, num_point, 1])
      alpha1 = tf.tile(tf_util._variable_on_cpu('alpha1', shape, initializer), [batch_size, num_point, 1])
      alpha2 = tf.tile(tf_util._variable_on_cpu('alpha2', shape, initializer), [batch_size, num_point, 1])
      alpha3 = tf.tile(tf_util._variable_on_cpu('alpha3', shape, initializer), [batch_size, num_point, 1])

      biases1 = tf.tile(tf_util._variable_on_cpu('biases1', shape, 
                        tf.constant_initializer(0.0)), [batch_size, num_point, 1])
      biases2 = tf.tile(tf_util._variable_on_cpu('biases2', shape, 
                        tf.constant_initializer(0.0)), [batch_size, num_point, 1])
      biases3 = tf.tile(tf_util._variable_on_cpu('biases3', shape, 
                        tf.constant_initializer(0.0)), [batch_size, num_point, 1])
      biases4 = tf.tile(tf_util._variable_on_cpu('biases4', shape, 
                        tf.constant_initializer(0.0)), [batch_size, num_point, 1])
      biases5 = tf.tile(tf_util._variable_on_cpu('biases5', shape, 
                        tf.constant_initializer(0.0)), [batch_size, num_point, 1])

      
      g1 = w_x * X + w_y * Y + w_z * Z + w_xyz * X * Y * Z
      g2 = w_xy * X * Y + w_yz * Y * Z + w_xz * X * Z + biases
      g3 = w_xx * X * X + w_yy * Y * Y + w_zz * Z * Z
      g4 = w_xxy * X * X * Y + w_xyy * X * Y * Y + w_xxz * X * X * Z
      g5 = w_xzz * X * Z * Z + w_yyz * Y * Y * Z + w_yzz * Y * Z * Z
      g6 = w_xxx * X * X * X + w_yyy * Y * Y * Y + w_zzz * Z * Z * Z
      #g_d = g1 + g2 + g3 + g4 + g5 + g6
      g_d = g1 + g2 + g3
      g_d = 100*((0.5*alpha0/math.pi)*tf.exp(-(g_d-biases1)*(g_d-biases1))+biases4)
      #g_d = alpha0*tf.exp(g_d+biases1)+biases2
      grouped_points = tf.expand_dims(grouped_points, -1)
      g_d = tf.expand_dims(g_d, 2)#[batch_size, num_point, K_knn, 1, 1]
      g_d = tf.tile(g_d, [1, 1, in_channels, 1])#[batch_size, num_point, K_knn, in_channels, 1]
      grouped_points = grouped_points * g_d
      grouped_points = tf.reshape(grouped_points, [batch_size, num_point, in_channels*taylor_channel])

      for i, num_out_channel in enumerate(mlp):
          grouped_points = tf_util.conv1d(grouped_points, num_out_channel, 1, padding='VALID', bn=True, is_training=is_training, scope='convf%d'%(i), bn_decay=bn_decay) 

      return grouped_points








