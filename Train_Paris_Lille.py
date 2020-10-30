import argparse
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os
import sys
import data_utils
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider
#import pointnet_part_seg as model
#import NCNN as model
#import Simple_seg as model
#import pointnet_sem_seg as model
#import sample_sem_seg as model
#import pointnet2_sem_seg as model
import tgnet as model
import h5py
from hdf5_util import *

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=12, help='Batch Size during training [default: 32]')
parser.add_argument('--epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--point_num', type=int, default=2048, help='Point Number [256/512/1024/2048]')
parser.add_argument('--output_dir', type=str, default='train_results', help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
FLAGS = parser.parse_args()

hdf5_data_dir = os.path.join(BASE_DIR, 'hdf5_data')

# MAIN SCRIPT
point_num = FLAGS.point_num
batch_size = FLAGS.batch
output_dir = FLAGS.output_dir
OPTIMIZER = FLAGS.optimizer

sample_num1 = round(point_num/64)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

NUM_CATEGORIES = 10 #class number 50


all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories_10.txt') #***********change
fin = open(all_obj_cats_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
print(lines)
all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
fin.close()

print('#### Batch Size: {0}'.format(batch_size))
print('#### Point Number: {0}'.format(point_num))
print('#### Training using GPU: {0}'.format(FLAGS.gpu))

DECAY_STEP = 16881 * 20
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-5

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.001
MOMENTUM = 0.9
TRAINING_EPOCHES = FLAGS.epoch
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'Paris_train_file_list.txt')
TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'Paris_test_file_list.txt')

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER =  os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label_seg'][:]
    return (data, label)

def printout(flog, data):
	print(data)
	flog.write(data + '\n')

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * batch_size,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate   

'''
def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))#32,2048,3
    seg_ph = tf.placeholder(tf.int32, shape=(batch_size, point_num))
    return pointclouds_ph, seg_ph
'''

def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))#32,2048,3
    seg_ph = tf.placeholder(tf.int32, shape=(batch_size, point_num))
    return pointclouds_ph, seg_ph

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_ph, seg_ph = placeholder_inputs()
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize. 
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
 
            #getting model and loss
            
            seg_pred = model.get_model(pointclouds_ph, \
                    is_training=is_training_pl, bn_decay=bn_decay, part_num=NUM_CATEGORIES, \
                    batch_size=batch_size, num_point=point_num, weight_decay=FLAGS.wd)
            '''seg_pred = model.get_model(pointclouds_ph, sample_ph1,\
                    is_training=is_training_pl, bn_decay=bn_decay, sample_num1= sample_num1,\
                    part_num=NUM_CATEGORIES, batch_size=batch_size, num_point=point_num, weight_decay=FLAGS.wd)'''

            seg_loss = model.get_loss(seg_pred, seg_ph)             

            correct = tf.equal(tf.argmax(seg_pred, 2), tf.to_int64(seg_ph))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(batch_size*point_num)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(seg_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
 
        #create a config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

        train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
        num_train_file = len(train_file_list)
        test_file_list = provider.getDataFiles(TESTING_FILE_LIST)
        num_test_file = len(test_file_list)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_ph,
               'labels_pl': seg_ph,
               'is_training_pl': is_training_pl,
               'pred': seg_pred,
               'loss': seg_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        # write logs to the disk
        flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')

        def train_one_epoch(train_file_idx, epoch_num):
            is_training = True

            total_correct = 0
            total_seen = 0
            loss_sum = 0
            num_batch = 0
             
            total_batch = 0

            for i in range(num_train_file):
                cur_train_filename = os.path.join(train_file_list[train_file_idx[i]])
                printout(flog, 'Loading train file ' + cur_train_filename)

                #================load train data=======================
                #load train data
                cur_data_all, cur_seg = load_h5(cur_train_filename)
                cur_data_all, cur_seg, order = provider.shuffle_data(cur_data_all, np.squeeze(cur_seg))

                '''#sample data
                a =  np.arange(cur_data.shape[0])[:,None]

                sample_top1 = np.argpartition(curv, -sample_num1, axis=1)[:, -sample_num1:]
                cur_data_sample1 = cur_data[a, sample_top1,:]'''
                
                num_data = len(cur_seg)
                num_batch = num_data // batch_size
                total_batch += num_batch

                for batch_idx in range(num_batch):
                    if batch_idx % 100 == 0:
                        print('Current batch/total batch num: %d/%d'%(batch_idx,num_batch))

                    start_idx = batch_idx * batch_size
                    end_idx = (batch_idx+1) * batch_size

                    feed_dict = {ops['pointclouds_pl']: cur_data_all[start_idx:end_idx, :, :],
                                 ops['labels_pl']: cur_seg[start_idx: end_idx, ...],
                                 ops['is_training_pl']: is_training,}
                    summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

                    train_writer.add_summary(summary, step)
                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum(pred_val == cur_seg[start_idx:end_idx,:])
                    total_correct += correct
                    total_seen += (batch_size*point_num)
                    loss_sum += loss_val
            
            printout(flog, '\t\tTraining Mean_loss: %f' % (loss_sum / float(total_batch)))
            printout(flog, '\t\tTraining Seg Accuracy: %f' % (total_correct / float(total_seen)))

        def eval_one_epoch(epoch_num):
            is_training = False

            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_seen_class = [0 for _ in range(NUM_CATEGORIES)]
            total_correct_class = [0 for _ in range(NUM_CATEGORIES)]
            total_per_cat_iou = np.zeros((NUM_CATEGORIES)).astype(np.float32)
            total_batch = 0


            for i in range(num_test_file):
                print(num_test_file)
                cur_test_filename = os.path.join(test_file_list[i])
                printout(flog, 'Loading test file ' + cur_test_filename)
                
                #================load val data=======================
                #load train data
                cur_data_all, cur_seg = load_h5(cur_test_filename)
                cur_data_all, cur_seg, order = provider.shuffle_data(cur_data_all, np.squeeze(cur_seg))

                #split curve
                cur_data = cur_data_all[:,:,0:3]
                
                '''
                #sample data
                a =  np.arange(cur_data.shape[0])[:,None]

                sample_top1 = np.argpartition(curv, -sample_num1, axis=1)[:, -sample_num1:]
                cur_data_sample1 = cur_data[a, sample_top1,:]'''

                num_data = len(cur_seg)
                num_batch = num_data // batch_size
                total_batch += num_batch

                batch_data = np.zeros((batch_size, point_num, 3))
                batch_label = np.zeros((batch_size, point_num), dtype=np.int32)

                for batch_idx in range(num_batch):
                    '''if j %20==0:
                        printout(flog, '%03d/%03d'%(j, num_batch))'''
                    start_idx = batch_idx * batch_size
                    end_idx = (batch_idx+1) * batch_size
              
                    batch_data = cur_data[start_idx: end_idx, ...]
                    batch_label = cur_seg[start_idx: end_idx, ...]

                    feed_dict = {ops['pointclouds_pl']: batch_data,
                                 ops['labels_pl']: batch_label,
                                 ops['is_training_pl']: is_training}
                    summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                                  feed_dict=feed_dict)
                    test_writer.add_summary(summary, step)
                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum(pred_val == batch_label)
                    total_correct += correct
                    total_seen += (batch_size*point_num)
                    loss_sum += loss_val*batch_size

                    for l in range(NUM_CATEGORIES):
                        total_seen_class[l] +=  (np.sum((pred_val==l) | (batch_label==l)))
                        total_correct_class[l] += (np.sum((pred_val==l) & (batch_label==l)))

            printout(flog, 'eval mean loss: %f' % (loss_sum / float(total_batch)))
            printout(flog, 'overal accuracy: %f'% (total_correct / float(total_seen)))

            ave_iou = 0.0
            for cat_idx in range(1, NUM_CATEGORIES):
                total_per_cat_iou[cat_idx] = total_correct_class[cat_idx]/float(total_seen_class[cat_idx]+1e-7)
                printout(flog, '\t\tCategory %s IoU is: %f' % (all_obj_cats[cat_idx][0], total_per_cat_iou[cat_idx]))

                ave_iou += total_per_cat_iou[cat_idx]/(NUM_CATEGORIES-1)
            printout(flog, '\n\t\tMean IoU is: %f' % (ave_iou))
            return ave_iou

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)
            print(MODEL_STORAGE_PATH)


        eval_iou_max = 0
        maxIoU_epoch = 0

        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n<<< Testing on the test dataset ...')
            iou = eval_one_epoch(epoch)

            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))
            train_file_idx = np.arange(0, len(train_file_list))
            np.random.shuffle(train_file_idx)

            train_one_epoch(train_file_idx, epoch)

            # Save the variables to disk.
            if iou > eval_iou_max:
                max_save_path = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'model_max_%d.ckpt' % epoch))
                maxIoU_epoch = epoch
                eval_iou_max = iou
                printout(flog, 'Model saved in file: %s' % (max_save_path))
            if epoch == (TRAINING_EPOCHES-1):
                save_path = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, "model.ckpt"))
                printout(flog, 'Model saved in file: %s' % (save_path))
                printout(flog, 'Max iou model saved in epoch: %d' % (maxIoU_epoch))
                printout(flog, 'Max iou is: %f' % (eval_iou_max))


            if (epoch+1) % 2 == 0:
                cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch+1)+'.ckpt'))
                printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

            flog.flush()


        flog.close()

if __name__=='__main__':
    train()
