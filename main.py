#!/usr/bin/env python
import argparse,logging,json,time,os,sys
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
#from tensorflow.python.client import device_lib
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import data_handler
import model,lr_func,losses,accuracies
import sklearn.metrics

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
   
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   logger.info('%s = \n %s',args.config_filename,json.dumps(config,indent=4,sort_keys=True))
   logger.info('-=-=-=-=-=-=-=-=-  CONFIG FILE -=-=-=-=-=-=-=-=-')
   config['hvd'] = hvd

   trainds,testds = data_handler.get_datasets(config)
   
   logger.info('get model')
   net = model.get_model(config)

   loss_func = losses.get_loss(config)

   opt = get_optimizer(config)

   if rank == 0:
      train_summary_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'train')
      test_summary_writer = tf.summary.create_file_writer(args.logdir + os.path.sep + 'test')

   first_batch = True
   batches_per_epoch = 0
   exit = False
   status_count = config['training']['status']
   batch_size = config['data']['batch_size']
   num_points = config['data']['num_points']
   for epoch_num in range(config['training']['epochs']):
      
      train_loss_metric = 0

      logger.info('begin epoch %s',epoch_num)

      batch_num = 0
      start = time.time()
      image_rate_sum = 0.
      image_rate_sum2 = 0.
      image_rate_n = 0.
      total_correct = 0.
      status_correct = 0.
      total_nonzero = 0.
      status_nonzero = 0.
      confusion_matrix = np.zeros([config['data']['num_classes'],config['data']['num_classes']])
      partial_img_rate = np.zeros(10)
      partial_img_rate_counter = 0
      if rank == args.profrank and args.profiler:
         logger.info('profiling')
         tf.profiler.experimental.start(args.logdir)
      for inputs, labels, class_weights in trainds:
         # logger.info('start train step')
         loss_value,pred = train_step(net,loss_func,opt,inputs,labels,first_batch,hvd,config['model']['name'],class_weights)
         # logger.info('end train step')
         tf.summary.experimental.set_step(batch_num + batches_per_epoch * epoch_num)

         class_weights = tf.cast(class_weights,tf.int32)
         nonzero = tf.math.reduce_sum(class_weights).numpy()
         status_nonzero += nonzero

         first_batch = False
         batch_num += 1

         train_loss_metric += tf.reduce_mean(loss_value) * (num_points/nonzero)

         pred = tf.cast(tf.argmax(pred,-1,),tf.int32)
         correct = tf.math.reduce_sum(class_weights * tf.cast(tf.math.equal(pred,labels),tf.int32))

         confusion_matrix += sklearn.metrics.confusion_matrix(labels.numpy().flatten(),pred.numpy().flatten(),sample_weight=class_weights.numpy().flatten(),normalize='true')
         
         status_correct += correct.numpy()
         
         if batch_num % status_count == 0:
            img_per_sec = status_count * batch_size * nranks / (time.time() - start)
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
               with train_summary_writer.as_default():
                  step = epoch_num * batches_per_epoch + batch_num
                  tf.summary.experimental.set_step(step)
                  tf.summary.scalar('loss', loss, step=step)
                  tf.summary.scalar('accuracy', acc, step=step)
                  tf.summary.scalar('img_per_sec',img_per_sec,step=step)
                  tf.summary.scalar('learning_rate',opt.lr(step))
            start = time.time()
                
         if args.batch_term == batch_num:
            logger.info('terminating batch training after %s batches',batch_num)
            if rank == args.profrank and args.profiler:
               logger.info('stop profiling')
               tf.profiler.experimental.stop()
            exit = True
            break
         # for testing
         # if batch_num == 20: break
      if exit:
         break
      if rank == 0:
         batches_per_epoch = batch_num
         ave_img_rate = image_rate_sum / image_rate_n
         std_img_rate = np.sqrt((1/image_rate_n) * image_rate_sum2 - ave_img_rate * ave_img_rate)
         logger.info('batches_per_epoch = %s  Ave Img Rate: %10.5f +/- %10.5f',batches_per_epoch,ave_img_rate,std_img_rate)

         # confustion matrix calc
         confusion_matrix = confusion_matrix / batch_num
         logger.info('confusion_matrix = \n %s',confusion_matrix)
      
      total_loss = 0.
      total_correct = 0.
      total_nonzero = 0.
      test_confusion_matrix = np.zeros([config['data']['num_classes'],config['data']['num_classes']])
      for test_num,(inputs,labels,class_weights) in enumerate(testds):
         # logger.info('inputs: %s labels: %s class_weights: %s',inputs.shape,labels.shape,class_weights.shape)
         loss_value,pred = test_step(net,loss_func,inputs,labels,class_weights)
         
         class_weights = tf.cast(class_weights,tf.int32)
         nonzero = tf.math.reduce_sum(class_weights).numpy()
         
         all_loss = tf.reduce_mean(loss_value)
         if hvd:
            all_loss = hvd.allreduce(all_loss)
         
         total_loss += all_loss * (num_points / nonzero)

         pred = tf.cast(tf.argmax(pred,-1,),tf.int32)
         correct = tf.math.reduce_sum(class_weights * tf.cast(tf.math.equal(pred,labels),tf.int32))
         all_correct = correct
         all_nonzero = nonzero
         if hvd:
            all_correct = hvd.allreduce(all_correct,op=hvd.mpi_ops.Sum)
            all_nonzero = hvd.allreduce(all_nonzero,op=hvd.mpi_ops.Sum)
         total_nonzero += all_nonzero.numpy()
         total_correct += all_correct.numpy()
         
         all_confusion_matrix = sklearn.metrics.confusion_matrix(labels.numpy().flatten(),pred.numpy().flatten(),sample_weight=class_weights.numpy().flatten(),normalize='true')
         
         if hvd: 
            all_confusion_matrix = hvd.allreduce(all_confusion_matrix)

         test_confusion_matrix += all_confusion_matrix

         
         
         if test_num > 0 and test_num % status_count == 0:
            test_loss = total_loss / test_num
            test_accuracy = total_correct / total_nonzero
            logger.info(' [%5d:%5d]: test loss = %10.5f  test acc = %10.5f',
                        epoch_num,test_num,test_loss,test_accuracy)
      if rank == 0:
         test_loss = total_loss / test_num
         test_accuracy = total_correct / total_nonzero
         with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss, step=epoch_num * batches_per_epoch + batch_num)
            tf.summary.scalar('accuracy', test_accuracy, step=epoch_num * batches_per_epoch + batch_num)
         ave_img_rate = image_rate_sum / image_rate_n
         std_img_rate = np.sqrt((1/image_rate_n) * image_rate_sum2 - ave_img_rate * ave_img_rate)
         template = 'Epoch {:10d}, Loss: {:10.5f}, Accuracy: {:10.5f}, Test Loss: {:10.5f}, Test Accuracy: {:10.5f} Average Image Rate: {:10.5f} +/- {:10.5f}'
         logger.info(template.format(epoch_num + 1,
                               loss,
                               acc,
                               test_loss,
                               test_accuracy,
                               ave_img_rate,
                               std_img_rate))
         test_confusion_matrix = test_confusion_matrix.numpy() / test_num
         logger.info('Train Confusion Matrix: \n %s',confusion_matrix)
         logger.info('Test  Confusion Matrix: \n %s',test_confusion_matrix)
         json.dump(confusion_matrix.tolist(),open(os.path.join(args.logdir,f'epoch{epoch_num+1:03d}_confustion_matrix_train.json'),'w'))
         json.dump(test_confusion_matrix.tolist(),open(os.path.join(args.logdir,f'epoch{epoch_num+1:03d}_confustion_matrix_test.json'),'w'))
         net.save_weights(os.path.join(args.logdir,f'epoch{epoch_num+1:03d}_model_weights.ckpt'))


@tf.function
def train_step(net,loss_func,opt,inputs,labels,first_batch=False,hvd=None,model_name='',class_weights=None,root_rank=0):
   
   with tf.GradientTape() as tape:
      pred = net(inputs, training=True)
      loss_value = loss_func(labels, pred,sample_weight=class_weights)
   
   if hvd:
      tape = hvd.DistributedGradientTape(tape)
   grads = tape.gradient(loss_value, net.trainable_variables)
   opt.apply_gradients(zip(grads, net.trainable_variables))
   # Horovod: broadcast initial variable states from rank 0 to all other processes.
   # This is necessary to ensure consistent initialization of all workers when
   # training is started with random weights or restored from a checkpoint.
   #
   # Note: broadcast should be done after the first gradient step to ensure optimizer
   # initialization.
   if hvd and first_batch:
      hvd.broadcast_variables(net.variables, root_rank=root_rank)
      hvd.broadcast_variables(opt.variables(), root_rank=root_rank)

   # tf.print('exiting train step')
   return loss_value,pred


@tf.function
def test_step(net,loss_func,inputs,labels,class_weights):
   # training=False is only needed if there are layers with different
   # behavior during training versus inference (e.g. Dropout).
   pred = net(inputs, training=False)

   loss_value = loss_func(labels, tf.cast(pred,tf.float32),sample_weight=class_weights)
   # tf.print(tf.math.reduce_sum(inputs),tf.argmax(tf.nn.softmax(predictions,-1),-1),labels,loss_value)

   return loss_value,pred


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
