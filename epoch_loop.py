import tensorflow as tf
import numpy as np
from tools.CalcMean import CalcMean,FifoMean
import time,logging,sklearn,json,os
logger = logging.getLogger(__name__)


def one_train_epoch(config,dataset,net,loss_func,opt,epoch_num,tbwriter,batches_per_epoch):
   return one_epoch(config,dataset,net,train_step,loss_func,opt,epoch_num,tbwriter,batches_per_epoch,True)


def one_eval_epoch(config,dataset,net,loss_func,opt,epoch_num,tbwriter,batches_per_epoch,jet_writer,ele_writer,bkg_writer):
   return one_epoch(config,dataset,net,test_step,loss_func,opt,epoch_num,tbwriter,batches_per_epoch,False,jet_writer,ele_writer,bkg_writer)


def one_epoch(config,dataset,net,step_func,loss_func,opt,epoch_num,tbwriter,
              batches_per_epoch,training,jet_writer=None,ele_writer=None,bkg_writer=None):
   
   # get configuration information
   first_batch    = (epoch_num == 0)
   status_count   = config['training']['status']
   batch_size     = config['data']['batch_size']
   num_classes    = config['data']['num_classes']
   rank           = config['rank']
   nranks         = config['nranks']
   profrank       = config.get('profrank',0)
   profiler       = config.get('profiler',False)
   batch_term     = config['batch_term']
   logdir         = config['logdir']
   hvd            = config.get('hvd',None)
   training_str   = 'training' if training else 'testing'
   
   # used for accuracy check
   softmax        = tf.keras.layers.Softmax()

   # create status monitoring variables
   total_loss     = CalcMean()
   status_loss    = CalcMean()
   # used for ongoing image rate calculation
   image_rate     = CalcMean()
   # and a recent image rate
   partial_img_rate = FifoMean(10)

   # used for accuracy (status_ gets zeroed periodically)
   total_correct = 0.
   status_correct = 0.
   total_nonzero = 0.
   status_nonzero = 0.

   # track iou
   status_iou = np.zeros(num_classes)
   total_iou = np.zeros(num_classes)

   # track confusion matrix
   status_confusion_matrix = np.zeros([num_classes,num_classes])
   total_confusion_matrix  = np.zeros([num_classes,num_classes])

   # run Tensorflow Profiling for some number of loops
   if rank == profrank and profiler:
      logger.info('profiling')
      tf.profiler.experimental.start(logdir)

   logger.info(' starting epoch: %s training: %s',epoch_num,training)
   # loop over batches
   status_start = time.time()
   batch_num = 0
   for inputs, labels, class_weights, nonzero_mask in dataset:
      
      # set the weights based on training flag
      weights = class_weights
      if not training:
         weights = nonzero_mask

      # run forward/backward pass
      loss_value,logits = step_func(net,loss_func,inputs,labels,weights,opt,first_batch,hvd)

      # cast from int8 to int32 for calculations
      weights = tf.cast(weights,tf.int32)
      # number of non-zero points in this batch
      nonzero = tf.math.reduce_sum(weights)

      # convert logits to predicted class
      pred = tf.cast(tf.argmax(softmax(logits),-1,),tf.int32)

      # logger.info('labels: %s %s',labels.shape,np.unique(labels,return_counts=True))
      # logger.info('pred: %s %s',pred.shape,np.unique(pred,return_counts=True))
      # logger.info('weights: %s %s',weights.shape,np.unique(weights,return_counts=True))

      # count how many predictions were correct, weighted for balance
      correct = tf.math.reduce_sum(weights * tf.cast(tf.math.equal(pred,labels),tf.int32))

      # iou for this batch
      iou = get_iou(labels,pred,num_classes,weights)

      # confusion matrix for this batch
      confusion_matrix = sklearn.metrics.confusion_matrix(labels.numpy().flatten(),pred.numpy().flatten(),sample_weight=weights.numpy().flatten())

      # add batch values to status monitoring variables for later output
      status_loss.add_value(loss_value.numpy())
      status_nonzero += nonzero.numpy()
      status_correct += correct.numpy()
      status_confusion_matrix += confusion_matrix
      status_iou += iou

      first_batch = False
      batch_num += 1
      
      # do monitoring periodically
      if batch_num % status_count == 0:

         # things for all ranks to do
         status_stop = time.time()
         
         # if Horovod is present, average status values across ranks
         if hvd:
            status_loss.allreduce(hvd)
            status_nonzero    = hvd.allreduce(status_nonzero,op=hvd.mpi_ops.Sum)
            status_nonzero    = status_nonzero.numpy()
            status_correct    = hvd.allreduce(status_correct,op=hvd.mpi_ops.Sum)
            status_correct    = status_correct.numpy()
            status_iou        = hvd.allreduce(status_iou,op=hvd.mpi_ops.Sum)
            status_iou        = status_iou.numpy()
            status_confusion_matrix = hvd.allreduce(status_confusion_matrix,op=hvd.mpi_ops.Sum)
            status_confusion_matrix = status_confusion_matrix.numpy()
         # calculate the number of images processed since last status message
         img_per_sec = status_count * batch_size * nranks / (status_stop - status_start)
         img_per_sec_std = 0
         
         # The first few batches are slow due to jit-compilation, etc.
         # so we don't include them in running averages
         if batch_num > 10:
            # keep running average
            image_rate.add_value(img_per_sec)
            # average last few image rates
            partial_img_rate.add_value(img_per_sec)

            img_per_sec = partial_img_rate.mean()
            img_per_sec_std = partial_img_rate.std()
         
         # calculate accuracy as total correct over total nonzero points
         acc = status_correct / status_nonzero
         # add totals to running correct/nonzero counters
         total_correct           += status_correct
         total_nonzero           += status_nonzero
         total_iou               += status_iou
         total_confusion_matrix  += status_confusion_matrix
         total_loss              += status_loss

         logger.info(" [%5d:%5d]: %s loss = %10.5f +/- %10.5f acc = %10.5f  imgs/sec = %7.1f +/- %7.1f",
                     epoch_num,batch_num,training_str,status_loss.mean(),
                     status_loss.std(),acc,img_per_sec,img_per_sec_std)
         
         # rank zero outputs tensorboard monitoring information during training only
         # during eval, tensorboard monitoring is added at end of full epoch
         if rank == 0 and training:
            with tbwriter.as_default():
               step = epoch_num * batches_per_epoch + batch_num
               tf.summary.experimental.set_step(step)
               tf.summary.scalar('metrics/loss', status_loss.mean(), step=step)
               tf.summary.scalar('metrics/accuracy', acc, step=step)
               tf.summary.scalar('monitors/img_per_sec',img_per_sec,step=step)
               tf.summary.scalar('monitors/learning_rate',opt.lr(step))

         # reset status counters
         status_correct = 0
         status_nonzero = 0
         status_loss.reset()
         status_iou = np.zeros(num_classes)
         status_confusion_matrix = np.zeros([num_classes,num_classes])
         
         status_start = time.time()
      
      # if profiling, need to exit and stop profiler
      if batch_term == batch_num:
         logger.info('terminating batch training after %s batches',batch_num)
         if rank == profrank and profiler:
            logger.info('stop profiling')
            tf.profiler.experimental.stop()
         break
   # end epoch loop

   # rank 0 will now dump final information
   if rank == 0:
      if training:
         batches_per_epoch = batch_num
      logger.info('batches_per_epoch = %s  Ave Img Rate: %10.5f +/- %10.5f',batches_per_epoch,image_rate.mean(),image_rate.std())

      # normalize the confusion matrix
      total_confusion_matrix = total_confusion_matrix
      for row in range(total_confusion_matrix.shape[0]):
         total_confusion_matrix[row,:] = total_confusion_matrix[row,:] / np.sum(total_confusion_matrix[row,:])

      logger.info('confusion_matrix = \n %s',total_confusion_matrix)

      total_iou = total_iou / batch_num / nranks
      logger.info('iou = %s',total_iou)
      if not training:
         step = epoch_num * batches_per_epoch + batch_num
         with tbwriter.as_default():
            #tf.summary.experimental.set_step(step)
            tf.summary.scalar('metrics/loss', total_loss.mean(),step=step)
            tf.summary.scalar('metrics/accuracy', acc,step=step)
         with jet_writer.as_default():
            tf.summary.scalar('metrics/iou',total_iou[0],step=step)
         with ele_writer.as_default():
            tf.summary.scalar('metrics/iou',total_iou[1],step=step)
         with bkg_writer.as_default():
            tf.summary.scalar('metrics/iou',total_iou[2],step=step)

         json.dump(confusion_matrix.tolist(),open(os.path.join(logdir,f'epoch{epoch_num+1:03d}_confustion_matrix_{training_str}.json'),'w'))
      else:
         model_weight_fn = os.path.join(logdir,f'epoch{epoch_num+1:03d}_model_weights.ckpt')
         logger.info('saving model weights: %s',model_weight_fn)
         net.save_weights(model_weight_fn)

         json.dump(confusion_matrix.tolist(),open(os.path.join(logdir,f'epoch{epoch_num+1:03d}_confustion_matrix_{training_str}.json'),'w'))
   
   return loss_value.numpy(),acc,total_confusion_matrix,batch_num,batches_per_epoch


@tf.function
def train_step(net,loss_func,inputs,labels,weights,opt=None,first_batch=False,hvd=None,root_rank=0):
   
   with tf.GradientTape() as tape:
      logits = net(inputs, training=True)
      # pred shape: [batches,points,classes]
      # labels shape: [batches,points]
      loss_value = loss_func(labels, logits)
      # loss_value shape: [batches,points]
      weights = tf.cast(weights,tf.float32)
      # zero out non useful points
      loss_value *= weights
      # loss_value shape: [batches,points]
      loss_value = tf.math.reduce_mean(loss_value)  # * (tf.size(weights,out_type=tf.float32) / tf.math.reduce_sum(weights))
      # loss_value shape: [1]

      # include regularization losses
      loss_value += tf.reduce_sum(net.losses)

      # tf.print('net.losses',net.losses)
   
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
   
   return loss_value,logits


@tf.function
def test_step(net,loss_func,inputs,labels,weights,opt=None,first_batch=False,hvd=None,root_rank=0):
   # training=False is only needed if there are layers with different
   # behavior during training versus inference (e.g. Dropout).
   logits = net(inputs, training=False)
   # run loss function
   loss_value = loss_func(labels, logits)
   # cast to float for calculations
   weights = tf.cast(weights,tf.float32)
   # zero out non useful points
   loss_value *= weights
   # reduce by mean and scale by the number of non-zero points
   loss_value = tf.math.reduce_mean(loss_value)  # * (tf.size(weights,out_type=tf.float32) / tf.math.reduce_sum(weights))
   
   # include regularization losses
   loss_value += tf.reduce_sum(net.losses)
   
   return loss_value,logits


def get_iou(truth,pred,num_classes,mask):
   # truth shape: [batch,points]
   # pred shape: [batch,points]
   # mask shape: [batch,points]
   # num_classes = int
   # empty holder of output
   # logger.info('get_iou  truth: %s',np.unique(truth,return_counts=True))
   # logger.info('get_iou  pred: %s',np.unique(pred,return_counts=True))
   # logger.info('get_iou  mask: %s',np.unique(mask,return_counts=True))
   iou = np.zeros(num_classes)
   # loop over classes
   for i in range(num_classes):
      # truth entries equal to this class label
      truth_classes = tf.math.equal(truth,i)
      # logger.info('class %d truth_classes: %s %s',i,tf.math.reduce_sum(tf.cast(truth_classes,tf.int32)),truth_classes.shape)
      # prediction entries equal to this class label
      pred_classes  = tf.math.equal(pred,i)
      # logger.info('class %d pred_classes: %s %s',i,tf.math.reduce_sum(tf.cast(pred_classes,tf.int32)),pred_classes.shape)
      
      # (truth_classes == pred_classes) * mask
      intersection = tf.cast(tf.math.logical_and(truth_classes,pred_classes),tf.int32) * mask
      # logger.info('intersection: %s',intersection.shape)
      intersection = tf.math.reduce_sum(intersection)
      # logger.info('class %d intersection: %s',i,intersection)

      # (truth_classes || pred_classes) * mask
      union = tf.cast(tf.math.logical_or(truth_classes,pred_classes),tf.int32) * mask
      # logger.info('union: %s',union.shape)
      union = tf.math.reduce_sum(union)
      # logger.info('class %d union: %s',i,union)

      iou[i] = intersection / union
   
   iou[np.isnan(iou)] = 0.
   # logger.info('iou: %s',iou)
   return iou
