#!/usr/bin/env python
import argparse,logging,json,time,os,sys,socket
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
import tensorflow_addons as tfa
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
      logging.warning('host: %s rank: %5d   size: %5d  local rank: %5d  local size: %5d',
                      socket.gethostname(),hvd.rank(), hvd.size(),
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
      config['training']['status'] = 1 if args.batch_term < config['training']['status'] else config['training']['status']

   if args.evaluate is not None:
      config['evaluate'] = True
      config['model_file'] = args.evaluate
      config['training']['epochs'] = 1
      logger.info('evaluating model file:      %s',args.evaluate)
   elif args.train_more is not None:
      config['train_more'] = True
      config['model_file'] = args.train_more
      logger.info('continuing model file:      %s',args.train_more)


   # using mixed precision?
   if isinstance(config['model']['mixed_precision'],str):
      logger.info('using mixed precsion:       %s',config['model']['mixed_precision'])
      tf.keras.mixed_precision.set_global_policy(config['model']['mixed_precision'])

   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   logger.info('%s = \n %s',args.config_filename,json.dumps(config,indent=4,sort_keys=True))
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   config['hvd'] = hvd

   sys.stdout.flush()
   sys.stderr.flush()



   trainds,testds = data_handler.get_datasets(config)
   
   logger.info('get model')
   net = model.get_model(config)
   loss_func = losses.get_loss(config)
   opt = get_optimizer(config)

   # initialize and create the model
   # input_shape = [config['data']['batch_size'],config['data']['num_points'],config['data']['num_features']]
   # output = net(tf.random.uniform(input_shape))

   # load previous model weights
   if args.evaluate:
      net.load_weights(args.evaluate)
   elif args.train_more:
      net.load_weights(args.train_more)

   # # synchronize models across ranks
   # if hvd:
   #    hvd.broadcast_variables(net.variables, root_rank=0)
   #    hvd.broadcast_variables(opt.variables(), root_rank=0)

   train_summary_writer = None
   test_summary_writer = None
   test_jet_writer = None
   test_ele_writer = None
   test_bkg_writer = None
   test_mean_writer = None
   if rank == 0:
      train_summary_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'train')
      test_summary_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'test')
      
      test_jet_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'jet_iou')
      test_ele_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'ele_iou')
      test_bkg_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'bkg_iou')
      test_mean_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'mean_iou')

      #tf.keras.utils.plot_model(net, "network_model.png", show_shapes=True)
      
      #with train_summary_writer.as_default():
        #tf.summary.graph(train_step.get_concrete_function().graph)

   batches_per_epoch = 0
   train_mIoU_sum = 0.
   test_mIoU_sum = 0.
   for epoch_num in range(config['training']['epochs']):
      
      logger.info('begin epoch %s',epoch_num)

      if not config['evaluate']:
         train_output = epoch_loop.one_train_epoch(config,trainds,net,
                                                   loss_func,opt,epoch_num,
                                                   train_summary_writer,
                                                   batches_per_epoch)
         batches_per_epoch = train_output['batches_per_epoch']
         train_mIoU_sum += train_output['mIoU']
         logger.info('train mIoU sum: %10.4f',train_mIoU_sum / (epoch_num + 1))

      test_output = epoch_loop.one_eval_epoch(config,testds,net,
                                              loss_func,opt,epoch_num,
                                              test_summary_writer,
                                              batches_per_epoch,
                                              test_jet_writer,
                                              test_ele_writer,
                                              test_bkg_writer,
                                              test_mean_writer)
      test_mIoU_sum += test_output['mIoU']
      logger.info('test mIoU sum: %10.4f',test_mIoU_sum / (epoch_num + 1))

      if rank == 0:
         with test_summary_writer.as_default():
            step = (epoch_num + 1) * batches_per_epoch
            tf.summary.scalar('metrics/mIoU_AOC', test_mIoU_sum / (epoch_num + 1),step=step)


def get_optimizer(config):

   # setup learning rate
   lr_schedule = None
   if 'lr_schedule' in config:
      lrs_name = config['lr_schedule']['name']
      lrs_args = config['lr_schedule'].get('args',None)
      if hasattr(tf.keras.optimizers.schedules, lrs_name):
         logger.info('using tf.keras.optimizers.schedules learning rate schedule %s', lrs_name)
         lr_schedule = getattr(tf.keras.optimizers.schedules, lrs_name)
         if lrs_args:
            lr_schedule = lr_schedule(**lrs_args)
         else:
            raise Exception('missing args for learning rate schedule %s',lrs_name)
      elif lrs_name in globals():
         logger.info('using global learning rate schedule %s', lrs_name)
         lr_schedule = globals()[lrs_name]
         if lrs_args:
            lr_schedule = lr_schedule(**lrs_args)
         else:
            raise Exception('missing args for learning rate schedule %s',lrs_name)
      elif hasattr(tfa.optimizers,lrs_name):
         logger.info('using tfa.optimizers learning rate schedule %s', lrs_name)
         lr_schedule = getattr(tfa.optimizers, lrs_name)
         if lrs_args:
            lr_schedule = lr_schedule(**lrs_args)
         else:
            raise Exception('missing args for learning rate schedule %s',lrs_name)
      elif hasattr(tf.keras.experimental,lrs_name):
         logger.info('using tf.keras.experimental learning rate schedule %s', lrs_name)
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
