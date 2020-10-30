import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import pickle
from sklearn.cluster import DBSCAN
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_so.so'))


def query_ball_point(radius, nsample, xyz1, xyz2):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    #return grouping_module.query_ball_point(radius, nsample, xyz1, xyz2)
    return grouping_module.query_ball_point(xyz1, xyz2, radius, nsample)
ops.NoGradient('QueryBallPoint')
def select_top_k(k, dist):
    '''
    Input:
        k: int32, number of k SMALLEST elements selected
        dist: (b,m,n) float32 array, distance matrix, m query points, n dataset points
    Output:
        idx: (b,m,n) int32 array, first k in n are indices to the top k
        dist_out: (b,m,n) float32 array, first k in n are the top k
    '''
    return grouping_module.selection_sort(dist, k)
ops.NoGradient('SelectionSort')
def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return grouping_module.group_point(points, idx)
@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_grad(points, idx, grad_out), None]

def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.get_shape()[0].value #batch size
    n = xyz1.get_shape()[1].value #number of points
    c = xyz1.get_shape()[2].value #channels
    m = xyz2.get_shape()[1].value #number of points in query
    print(b, n, c, m)
    print(xyz1, (b,1,n,c))
    xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    print(dist, k)
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0,0,0], [-1,-1,k])#take out the top k value
    val = tf.slice(out, [0,0,0], [-1,-1,k])
    print(idx, val)
    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx


def kmeans_point(xyz1, clusterNum):
    '''
    Input:
        xyz1: (batch_size, ndataset, k, 3) float32 array, input points
        clusterNum : int
    Output:
        label: (batch_size, npoint, k*k) int32 array, indices to input points
    '''
    b = xyz1.get_shape()[0].value #batch size
    n = xyz1.get_shape()[1].value #number of points
    k = xyz1.get_shape()[2].value 
    c = xyz1.get_shape()[3].value #channels

    xyz1 = tf.reshape(xyz1, (b*n,k,c))
    #select the first three point as initial cluster central
    xyz = xyz1[:,0:3, :]
    K_knn = 3
    _, idx = knn_point(K_knn, xyz, xyz1) #
   
    seg_idx = idx[:,:,0]
    seg_ori = tf.tile(tf.reshape(seg_idx, (b*n,1,k)), [1,k,1])
    seg_trans = tf.tile(tf.reshape(seg_idx, (b*n,k,1)), [1,1,k])
    mask = tf.equal(seg_ori, seg_trans)
    mask1 = tf.reshape(tf.cast(mask, tf.float32), (b, n, k*k))
   
    return(mask1)

    '''
    print(mask)
    for i in range(clusterNum):
        new_centroids[:,:,i]+ =  
   
    for i in range(b*n):
        xyz_line = xyz1[i,...]
        ids = seg_id[i,...]
        new_centroids = tf.segment_mean(xyz1, seg_id[i,:], name=None)        
    sess = tf.Session()
    print(sess.run(mask1))
    '''

def DBSCAN_Simi(delta, min_point, eps_dist):
    """
    Input:
        delta: 4-D tensor variable BxNxkx3
        min_point: int
        eps_dist: float
      
    Returns:
      feat: 3-D tensor variable BxNx(k*k)
    """
    B = delta.get_shape()[0].value
    N = delta.get_shape()[1].value
    K = delta.get_shape()[2].value
    #lab = np.zeros((B, N, K*K), dtype=np.int32)
    #labpts = tf.zeros((B, N, K), tf.int32)#(B, N, K)

    #DBSCAN
    with tf.Session() as sess:
        '''
    for i in range(B):
            for j in range(N):
                sess.run(tf.global_variables_initializers())
                one_in = delta[i, j,...].eval() 
                db = DBSCAN(eps=eps_dist, min_samples=min_point).fit(one_in)
                labels = db.labels_
                labels1 = np.reshape(np.tile(labels, K), (K, K))
                labels1_T = labels1.T
                simi = labels1 - labels1_T
                simi[simi<0] = 1 
                simi[simi>0] = 1
                simi_fet = np.reshape(simi, K*K)
                lab[i, j, :] = simi_fet[:]

    labpts = tf.convert_to_tensor(lab)
        '''
    return lab

'''
if __name__=='__main__':
    knn=True
    import numpy as np
    import time
    np.random.seed(100)
    pts = np.random.random((32,512,64)).astype('float32')
    tmp1 = np.random.random((32,512,3)).astype('float32')
    tmp2 = np.random.random((32,128,3)).astype('float32')
    with tf.device('/gpu:1'):
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        radius = 0.1 
        nsample = 64
        if knn:
            _, idx = knn_point(nsample, xyz1, xyz2)#find the nearset 64 points
            grouped_points = group_point(points, idx)
        else:
            idx, _ = query_ball_point(radius, nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)
            #grouped_points_grad = tf.ones_like(grouped_points)
            #points_grad = tf.gradients(grouped_points, points, grouped_points_grad)
    with tf.Session('') as sess:
        now = time.time() 
        for _ in range(100):
            ret = sess.run(grouped_points)
        print(time.time() - now)
        print(ret.shape, ret.dtype)
        print(ret)
'''


np.random.seed(100)
pts = np.random.random((10, 10, 6, 3))
pts = tf.convert_to_tensor(pts, dtype=tf.float32)
clusterNum = 2
iterNum = 5
#eps_dist = 0.5
#simi_feat = DBSCAN_Simi(pts, min_point, eps_dist)
simi_feat = kmeans_point(pts, clusterNum, iterNum)

