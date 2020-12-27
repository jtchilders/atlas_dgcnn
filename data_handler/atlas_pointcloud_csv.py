import tensorflow as tf
import logging,glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
logger = logging.getLogger(__name__)

gconfig = None
labels_dict =  {0:0,1:0, -1:0, 2:0, -2:0, 3:0, -3:0, 4:0, -4:0, 
                5:0, -5:0, 11:1,-11:1,13:2,-13:2,-99:2, 15:2, -15: 2}

col_names    = ['id', 'index', 'x', 'y', 'z', 'r', 'eta', 'phi',
                'Et','pid','pn','peta','pphi','ppt','trk_good','trk_id','trk_pt']
col_dtype    = {'id': np.int64, 'index': np.int32, 'x': np.float32, 'y': np.float32,
                'z': np.float32, 'eta': np.float32, 'phi': np.float32, 'r': np.float32,
                'Et': np.float32, 'pid': np.int32, 'pn': np.int32, 'peta': np.float32,
                'pphi': np.float32, 'ppt': np.float32,
                'trk_good': np.float32, 'trk_id': np.float32, 'trk_pt': np.float32}
include_cols = ['x','y','z','eta','phi','r','Et']
gnum_points = None
gnum_features = None


def get_datasets(config):
   global gconfig,gnum_points,gnum_features
   gconfig = config
   gnum_points = gconfig['data']['num_points']
   gnum_features = gconfig['data']['num_features']

   train = from_filelist(config['data']['train_filelist'],config)
   valid = from_filelist(config['data']['test_filelist'],config)
   
   return train,valid


def from_filelist(filelist_filename,config):
   logger.debug(f'build dataset {filelist_filename}')
   dc = config['data']

   numranks = 1
   if config['hvd'] is not None:
      numranks = config['hvd'].size()

   # extract filelist
   filelist = []
   with open(filelist_filename) as file:
      for line in file:
         filelist.append(line.strip())

   # estimate batches per MPI rank
   total_batches = int(len(filelist) / dc['batch_size'])
   batches_per_rank = int(total_batches / numranks)
   total_even_batches = batches_per_rank * numranks
   total_events = total_even_batches * dc['batch_size']
   logger.info(f'input filelist contains {len(filelist)} files, estimated batches per rank {batches_per_rank}')
   
   # glob for the input files
   filelist = tf.data.Dataset.from_tensor_slices(filelist[0:total_events])
   
   # shard the data
   if config['hvd']:
      filelist = filelist.shard(config['hvd'].size(), config['hvd'].rank())
   
   # shuffle and repeat at the input file level
   logger.debug('starting shuffle')
   filelist = filelist.shuffle(dc['shuffle_buffer'],reshuffle_each_iteration=dc['reshuffle_each_iteration'])

   # map to read files in parallel
   logger.debug('starting map')
   ds = filelist.map(load_csv_file,
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)  # dc['num_parallel_readers']) #
   # ds = tf.data.experimental.CsvDataset(filelist,
   #                                      record_defaults=[tf.int64,tf.int32,tf.float32,tf.float32,
   #                                                       tf.float32,tf.float32,tf.float32,tf.float32,
   #                                                       tf.float32,tf.int32,tf.int32,tf.float32,
   #                                                       tf.float32,tf.float32],
   #                                      field_delim='\t',
   #                                      select_cols=[0,2,3,4,5,6,7,8,9])
   # how many inputs to prefetch to improve pipeline performance
   ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # dc['prefectch_buffer_size']) #

   # ds = ds.apply(tf.data.Dataset.unbatch)

   # batch the data
   ds = ds.batch(dc['batch_size'],drop_remainder=True)

   return ds


def load_csv_file(filename):
   return tf.py_function(load_csv_file_py,[filename],[tf.float32,tf.int32,tf.int8,tf.int8])


def load_csv_file_py(filename):
   filename = bytes.decode(filename.numpy())

   df = pd.read_csv(filename,header=None,names=col_names, dtype=col_dtype, sep='\t')

   # clip the number of points from the input based on the config num_points
   if len(df) > gnum_points:
      df = df[0:gnum_points]

   # logger.info('1 df_inputs: %s',df_inputs.shape)
   # normalize variables
   if False:
      # build the model inputs
      df_inputs = df[include_cols].to_numpy()
      scaler = MinMaxScaler()
      df_inputs = scaler.fit_transform(df_inputs)
   else:
      r_mean = df['r'].mean()
      r_sigma = df['r'].std()
      df['r'] -= r_mean
      df['r'] /= (r_sigma + np.finfo(np.float32).eps)

      et_mean = df['Et'].mean()
      et_sigma = df['Et'].std()
      df['Et'] -= et_mean
      df['Et'] /= (et_sigma + np.finfo(np.float32).eps)

      # build the model inputs
      df_inputs = df[include_cols].to_numpy()
   # logger.info('2 df_inputs: %s',df_inputs.shape)
   # tf.print('df_inputs: ',df_inputs[0:10,...])

   # stuff ragged event sizes into fixed size
   inputs = np.zeros([gnum_points,gnum_features])
   # logger.info('3 inputs: %s',inputs.shape)
   inputs[0:df_inputs.shape[0],...] = df_inputs[0:df_inputs.shape[0],...]

   # build the labels
   df_labels = df[['pid']]
   # map pid to class label
   df_labels = df_labels.replace({'pid':labels_dict})
   # logger.info('5 df_labels: %s %s',df_labels.shape,np.unique(df_labels,return_counts=True))

   # convert to numpy
   df_labels = df_labels.to_numpy()
   df_labels = np.squeeze(df_labels,-1)

   # count number of each class
   # use the lowest to decide weights for loss function
   # get list of unique classes and their occurance count
   unique_classes,unique_counts = np.unique(df_labels,return_counts=True)
   # get mininum class occurance count
   min_class_count = np.min(unique_counts)
   # logger.info('66 min_class_count: %s',min_class_count)
   # tf.print('min_class_count:',min_class_count,unique_classes,unique_counts)
   # create class weights to be applied to loss as mask
   # this will balance the loss function across classes
   class_weights = np.zeros([gnum_points],dtype=np.int8)
   # set weights to one for an equal number of classes
   for class_label in unique_classes:
      # tf.print('class_label:',class_label)
      class_indices = np.nonzero(df_labels == class_label)[0]
      # logger.info('class_indices: %s %s',class_label,df_labels[class_indices])
      class_indices = np.random.choice(class_indices,size=[min_class_count],replace=False)
      class_weights[class_indices] = 1
      # tf.print('3: ',np.unique(class_weights,return_counts=True))
   
   # logger.info('7 class_weights: %s %s %s',class_weights.shape,class_weights.sum(),1)

   nonzero_mask = np.zeros([gnum_points],dtype=np.int8)
   nonzero_mask[0:df_labels.shape[0]] = 1
   # logger.info('8 nonzero_mask: %s %s',nonzero_mask.shape,nonzero_mask.sum())

   # pad with zeros or clip some points
   labels = np.zeros([gconfig['data']['num_points']])
   labels[0:df_labels.shape[0]] = df_labels[0:df_labels.shape[0]]

   # return inputs and labels
   return inputs,labels,class_weights,nonzero_mask
