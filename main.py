#!/usr/bin/env python
import argparse,logging,json,time,os,sys
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow.python.client import device_lib
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import data_handler
import model,lr_func,losses,accuracies
import sklearn.metrics
import epoch_loop
logger = logging.getLogger(__name__)
DEFAULT_CONFIG = 'config.json'
DEFAULT_INTEROP = int(os.cpu_count() / 4)
DEFAULT_INTRAOP = int(os.cpu_count() / 4)
DEFAULT_LOGDIR = '/tmp/tf-' + str(os.getpid())


def main():
   ''' simple starter program for tensorflow models. '''
   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config',dest='config_filename',help='configuration filename in json format [default: %s]' % DEFAULT_CONFIG,default=DEFAULT_CONFIG)
   parser.add_argument('--interop',type=int,help='set Tensorflow "inter_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTEROP,default=DEFAULT_INTEROP)
   parser.add_argument('--intraop',type=int,help='set Tensorflow "intra_op_parallelism_threads" session config varaible [default: %s]' % DEFAULT_INTRAOP,default=DEFAULT_INTRAOP)
   parser.add_argument('-l','--logdir',default=DEFAULT_LOGDIR,help='define location to save log information [default: %s]' % DEFAULT_LOGDIR)

   parser.add_argument('--horovod', default=False, action='store_true', help="Use MPI with horovod")
   parser.add_argument('--profiler',default=False, action='store_true', help='Use TF profiler, needs CUPTI in LD_LIBRARY_PATH for Cuda')
   parser.add_argument('--profrank',default=0,type=int,help='set which rank to profile')

   parser.add_argument('--precision',default='float32',help='set which precision to use; options include: "float32","mixed_float16","mixed_bfloat16"')

   parser.add_argument('--batch-term',dest='batch_term',type=int,help='if set, terminates training after the specified number of batches',default=0)

   parser.add_argument('--evaluate',help='evaluate a pre-trained model file on the test data set only.')
   parser.add_argument('--train-more',dest='train_more',help='load a pre-trained model file and continue training.')

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()
   
   hvd = None
   rank = 0
   nranks = 1
   logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   if args.horovod:
      print('importing horovod')
      sys.stdout.flush()
      sys.stderr.flush()

      import horovod
      import horovod.tensorflow as hvd
      hvd.init()
      logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:' + (
                 '%05d' % hvd.rank()) + ':%(name)s:%(message)s'
      rank = hvd.rank()
      nranks = hvd.size()
      if rank > 0:
         logging_level = logging.WARNING
   
   # Setup Logging
   if args.debug and not args.error and not args.warning:
      logging_level = logging.DEBUG
      os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
   elif not args.debug and args.error and not args.warning:
      logging_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      logging_level = logging.WARNING

   logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)
   
   if hvd:
      logging.warning('rank: %5d   size: %5d  local rank: %5d  local size: %5d',
                      hvd.rank(), hvd.size(),
                      hvd.local_rank(), hvd.local_size())
   
   tf.config.threading.set_inter_op_parallelism_threads(args.interop)
   tf.config.threading.set_intra_op_parallelism_threads(args.intraop)

   # Setup GPUs
   gpus = tf.config.list_physical_devices('GPU')
   logger.info(   'number of gpus:              %s',len(gpus))
   for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
   if hvd and len(gpus) > 0:
      tf.config.set_visible_devices(gpus[hvd.local_rank() % len(gpus)],'GPU')
   
   logging.info(   'using tensorflow version:   %s (%s)',tf.__version__,tf.__git_version__)
   logging.info(   'using tensorflow from:      %s',tf.__file__)
   if hvd:
      logging.info('using horovod version:      %s',horovod.__version__)
      logging.info('using horovod from:         %s',horovod.__file__)
   logging.info(   'logdir:                     %s',args.logdir)
   logging.info(   'interop:                    %s',args.interop)
   logging.info(   'intraop:                    %s',args.intraop)

   config = json.load(open(args.config_filename))
   # config['device'] = device_str
   
   config['profrank'] = args.profrank
   config['profiler'] = args.profiler
   config['logdir'] = args.logdir
   config['rank'] = rank
   config['nranks'] = nranks
   config['evaluate'] = False
   config['batch_term'] = args.batch_term
   if args.batch_term > 0:
      config['training']['epochs'] = 1

   if args.evaluate is not None:
      config['evaluate'] = True
      config['model_file'] = args.evaluate
      config['training']['epochs'] = 1
      logger.info('evaluating model file:      %s',args.evaluate)
   elif args.train_more is not None:
      config['train_more'] = True
      config['model_file'] = args.train_more
      logger.info('continuing model file:      %s',args.train_more)

   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   logger.info('%s = \n %s',args.config_filename,json.dumps(config,indent=4,sort_keys=True))
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   config['hvd'] = hvd

   sys.stdout.flush()
   sys.stderr.flush()

   trainds,testds = data_handler.get_datasets(config)
   
   logger.info('get model')
   net = model.get_model(config)

   if args.evaluate:
      net.load_weights(args.evaluate)
   elif args.train_more:
      net.load_weights(args.train_more)

   loss_func = losses.get_loss(config)

   opt = get_optimizer(config)

   train_summary_writer = None
   test_summary_writer = None
   test_jet_writer = None
   test_ele_writer = None
   test_bkg_writer = None
   if rank == 0:
      train_summary_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'train')
      test_summary_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'test')
      
      test_jet_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'jet_iou')
      test_ele_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'ele_iou')
      test_bkg_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'bkg_iou')

      #tf.keras.utils.plot_model(net, "network_model.png", show_shapes=True)
      
      #with train_summary_writer.as_default():
        #tf.summary.graph(train_step.get_concrete_function().graph)

   first_batch = True
   batches_per_epoch = 0
   exit = False
   status_count = config['training']['status']
   batch_size = config['data']['batch_size']
   num_points = config['data']['num_points']
   num_classes = config['data']['num_classes']
   softmax = tf.keras.layers.Softmax()
   for epoch_num in range(config['training']['epochs']):
      
      logger.info('begin epoch %s',epoch_num)

      loss = 0.
      acc = 0.
      confusion_matrix = None
      batch_num = 0.
      if not config['evaluate']:
         loss,acc,confusion_matrix,batch_num,batches_per_epoch = \
               epoch_loop.one_train_epoch(config,trainds,net,
                                 loss_func,opt,epoch_num,train_summary_writer,
                                 batches_per_epoch)

      loss,acc,confusion_matrix,batch_num,batches_per_epoch = \
            epoch_loop.one_eval_epoch(config,testds,net,
                                 loss_func,opt,epoch_num,test_summary_writer,
                                 batches_per_epoch,test_jet_writer,
                                 test_ele_writer,test_bkg_writer)
      
      


def train_one_epoch(config,dataset,net,loss_func,opt,epoch_num,tbwriter,
                    batches_per_epoch,profiler=None,profrank=0,logdir=None):

   first_batch    = (epoch_num == 0)
   status_count   = config['training']['status']
   batch_size     = config['data']['batch_size']
   num_points     = config['data']['num_points']
   num_classes    = config['data']['num_classes']
   rank           = config['rank']
   batch_num      = 0
   start          = time.time()
   train_loss_metric = 0
   softmax        = tf.keras.layers.Softmax()

   # used for ongoing image rate calculation
   image_rate_sum = 0.
   image_rate_sum2 = 0.
   image_rate_n = 0.

   # used for accuracy (status_ gets zeroed periodically)
   total_correct = 0.
   status_correct = 0.
   total_nonzero = 0.
   status_nonzero = 0.

   # track confusion matrix
   confusion_matrix = np.zeros([num_classes,num_classes])

   partial_img_rate = np.zeros(10)
   partial_img_rate_counter = 0
   
   # run Tensorflow Profiling for some number of loops
   if rank == profrank and profiler:
      logger.info('profiling')
      tf.profiler.experimental.start(logdir)

   for inputs, labels, class_weights, nonzero_mask in dataset:
      # logger.error('%03d start train step',batch_num)
      loss_value,pred = train_step(net,loss_func,opt,inputs,labels,first_batch,config['hvd'],class_weights)
      # logger.error('%03d done train step',batch_num)
      
      tf.summary.experimental.set_step(batch_num + batches_per_epoch * epoch_num)

      class_weights = tf.cast(class_weights,tf.int32)
      # number of non-zero points in this batch
      nonzero = tf.math.reduce_sum(class_weights).numpy()
      status_nonzero += nonzero

      first_batch = False
      batch_num += 1

      # average loss value over batches for status message
      train_loss_metric += loss_value
      
      pred = tf.cast(tf.argmax(softmax(pred),-1,),tf.int32)
      correct = tf.math.reduce_sum(class_weights * tf.cast(tf.math.equal(pred,labels),tf.int32))
      
      confusion_matrix += sklearn.metrics.confusion_matrix(labels.numpy().flatten(),pred.numpy().flatten(),sample_weight=class_weights.numpy().flatten(),normalize='true')
      
      status_correct += correct.numpy()
      
      # logger.error('%03d run status',batch_num)
      if batch_num % status_count == 0:
         img_per_sec = status_count * batch_size * config['nranks'] / (time.time() - start)
         img_per_sec_std = 0
         if batch_num > 10:
            image_rate_n += 1
            image_rate_sum += img_per_sec
            image_rate_sum2 += img_per_sec * img_per_sec
            partial_img_rate[partial_img_rate_counter % 10] = img_per_sec
            partial_img_rate_counter += 1
            img_per_sec = np.mean(partial_img_rate[partial_img_rate > 0])
            img_per_sec_std = np.std(partial_img_rate[partial_img_rate > 0])
         loss = train_loss_metric / status_count
         acc = float(status_correct) / status_nonzero
         total_correct += status_correct
         total_nonzero += status_nonzero
         status_correct = 0
         status_nonzero = 0
         train_loss_metric = 0
         logger.info(" [%5d:%5d]: loss = %10.5f acc = %10.5f  imgs/sec = %7.1f +/- %7.1f",
                        epoch_num,batch_num,loss,acc,img_per_sec,img_per_sec_std)
         if rank == 0:
            with tbwriter.as_default():
               step = epoch_num * batches_per_epoch + batch_num
               tf.summary.experimental.set_step(step)
               tf.summary.scalar('metrics/loss', loss, step=step)
               tf.summary.scalar('metrics/accuracy', acc, step=step)
               tf.summary.scalar('monitors/img_per_sec',img_per_sec,step=step)
               tf.summary.scalar('monitors/learning_rate',opt.lr(step))
         start = time.time()
      # logger.error('%03d after status',batch_num)
             
      if config['batch_term'] == batch_num:
         logger.info('terminating batch training after %s batches',batch_num)
         if config['rank'] == profrank and profiler:
            logger.info('stop profiling')
            tf.profiler.experimental.stop()
         break
      # for testing
      # logger.error('%03d end of loop',batch_num)
      # if batch_num == 10: break
   # logger.error('exited loop')
   if rank == 0:
      batches_per_epoch = batch_num
      ave_img_rate = image_rate_sum / image_rate_n
      std_img_rate = np.sqrt((1 / image_rate_n) * image_rate_sum2 - ave_img_rate * ave_img_rate)
      logger.info('batches_per_epoch = %s  Ave Img Rate: %10.5f +/- %10.5f',batches_per_epoch,ave_img_rate,std_img_rate)

      # confustion matrix calc
      confusion_matrix = confusion_matrix / batch_num
      logger.info('confusion_matrix = \n %s',confusion_matrix)
   
   return loss,acc,confusion_matrix,batch_num,batches_per_epoch



def get_optimizer(config):

   # setup learning rate
   lr_schedule = None
   if 'lr_schedule' in config:
      lrs_name = config['lr_schedule']['name']
      lrs_args = config['lr_schedule'].get('args',None)
      if hasattr(tf.keras.optimizers.schedules, lrs_name):
         logger.info('using learning rate schedule %s', lrs_name)
         lr_schedule = getattr(tf.keras.optimizers.schedules, lrs_name)
         if lrs_args:
            lr_schedule = lr_schedule(**lrs_args)
         else:
            raise Exception('missing args for learning rate schedule %s',lrs_name)
      elif lrs_name in globals():
         logger.info('using learning rate schedule %s', lrs_name)
         lr_schedule = globals()[lrs_name]
         if lrs_args:
            lr_schedule = lr_schedule(**lrs_args)
         else:
            raise Exception('missing args for learning rate schedule %s',lrs_name)
      elif hasattr(tfa.optimizers,lrs_name):
         logger.info('using learning rate schedule %s', lrs_name)
         lr_schedule = getattr(tfa.optimizers, lrs_name)
         if lrs_args:
            lr_schedule = lr_schedule(**lrs_args)
         else:
            raise Exception('missing args for learning rate schedule %s',lrs_name)
      else:
         raise Exception('failed to find lr_schedule: %s' % lrs_name)

   opt_name = config['optimizer']['name']
   opt_args = config['optimizer'].get('args',{})
   if hasattr(tf.keras.optimizers, opt_name):
      logger.info('using optimizer: %s',opt_name)
      if opt_args:
         if lr_schedule:
            opt_args['learning_rate'] = lr_schedule
         logger.info('passing args to optimizer: %s', opt_args)
         return getattr(tf.keras.optimizers, opt_name)(**opt_args)
      else:
         if lr_schedule:
            opt_args['learning_rate'] = lr_schedule
         logger.info('passing args to optimizer: %s', opt_args)
         return getattr(tf.keras.optimizers, opt_name)(**opt_args)
   else:
      raise Exception('could not locate optimizer %s',opt_name)


if __name__ == "__main__":
   main()
