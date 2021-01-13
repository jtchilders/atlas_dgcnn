import tensorflow as tf,logging
import numpy as np
logger = logging.getLogger(__name__)


def get_model(config):
   net = DGCNN(config)
   return net


class ConvBnLayer(tf.keras.layers.Layer):
   def __init__(self,features,kernel=[1,1],padding='valid',strides=[1,1],
                kernel_initializer='GlorotNormal',kernel_regularizer=None,
                activation='ReLU'):
      super(ConvBnLayer,self).__init__()
      self.conv = tf.keras.layers.Conv2D(features,kernel,padding=padding,
                                         strides=strides,
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer)
      self.bn   = tf.keras.layers.BatchNormalization()

      if activation is not None and hasattr(tf.keras.layers,activation):
         self.activ = getattr(tf.keras.layers,activation)
         self.activ = self.activ()
      else:
         self.activ = None

   def call(self,inputs,training=False):
      conv = self.conv(inputs)
      bn = self.bn(conv,training=training)
      if self.activ:
         return self.activ(bn)
      else:
         return bn


class DenseBnLayer(tf.keras.layers.Layer):
   def __init__(self,features,kernel_initializer='GlorotNormal',kernel_regularizer=None,
                activation='ReLU'):
      super(DenseBnLayer,self).__init__()
      self.dense = tf.keras.layers.Dense(features,
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer)
      self.bn   = tf.keras.layers.BatchNormalization()

      if activation is not None and hasattr(tf.keras.layers,activation):
         self.activ = getattr(tf.keras.layers,activation)
         self.activ = self.activ()
      else:
         self.activ = None

   def call(self,inputs,training=False):
      dense = self.dense(inputs)
      bn = self.bn(dense,training=training)
      if self.activ:
         return self.activ(bn)
      else:
         return bn


class EdgeLayer(tf.keras.layers.Layer):
   def __init__(self,k):
      super(EdgeLayer,self).__init__()
      self.k = k

   def build(self,input_shape):
      if len(input_shape) == 3:
         self.batch_size,self.num_points,self.nfeatures = input_shape
      elif len(input_shape) == 4:
         self.batch_size,self.num_points,_,self.nfeatures = input_shape


   def call(self,inputs,training=False):

      """Compute pairwise distance of a point cloud.

         Args:
          inputs: tensor (batch_size, num_points, num_dims)

         Returns:
          pairwise distance: (batch_size, num_points, num_points)
      """
      inputs = tf.squeeze(inputs)
      if self.batch_size == 1:
         inputs = tf.expand_dims(inputs, 0)
      
      # batch-wise sqaure, transpose point_cloud like (0,2,1), then multiply
      point_cloud_inner = -2 * tf.matmul(inputs, inputs, transpose_b=True)
      point_cloud_square = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
      point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
      adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

      """Get KNN based on the pairwise distance.
      Args:
       pairwise distance: (batch_size, num_points, num_points)
       k: int

      Returns:
       nearest neighbors: (batch_size, num_points, k)
      """
      dist, nn_idx = tf.math.top_k(tf.negative(adj_matrix), k=self.k)
      
      """Construct edge feature for each point
      Args:
       point_cloud: (batch_size, num_points, 1, num_dims)
       nn_idx: (batch_size, num_points, k)
       k: int

      Returns:
       edge features: (batch_size, num_points, k, num_dims)
      """

      # shape = [batch_size]
      idx_ = tf.range(self.batch_size) * self.num_points
      # shape = [batch_size,1,1]
      idx_ = tf.reshape(idx_, [self.batch_size, 1, 1])

      # shape = [batch_size*num_points,nfeatures]
      point_cloud_flat = tf.reshape(inputs, [-1, self.nfeatures])
      # shape = [batch_size*num_points,nfeatures]
      point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
      point_cloud_central = tf.expand_dims(inputs, axis=-2)

      point_cloud_central = tf.tile(point_cloud_central, [1, 1, self.k, 1])

      edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)

      return edge_feature


class TransformOutputLayer(tf.keras.layers.Layer):
   def __init__(self,nfeatures):
      super(TransformOutputLayer,self).__init__()
      self.flatten = tf.keras.layers.Flatten()
      self.reshape = tf.keras.layers.Reshape([nfeatures,nfeatures])
      self.nfeatures = nfeatures

   def build(self,input_shape):
      units = input_shape[1]

      self.w = self.add_weight("input_transform_weights",shape=[units,self.nfeatures * self.nfeatures],initializer='zeros',trainable=True)
      self.b = self.add_weight("input_transform_biases",shape=[self.nfeatures * self.nfeatures],initializer='zeros',trainable=True)
      eye = tf.Variable(tf.reshape(tf.eye(self.nfeatures),[-1]),dtype=tf.float32,trainable=False)
      self.b.assign_add(eye)

   def call(self,inputs,training=False):
      transform = tf.matmul(inputs, self.w) + self.b
      transform = self.reshape(transform)
      return transform


class InputTransformNet(tf.keras.layers.Layer):
   """ Input (XYZ) Transform Net, input is BxNx3 gray image
    Return:
      Transformation matrix of size 3xK """
   def __init__(self,config):
      super(InputTransformNet,self).__init__()

      self.use_kernel_reg = config['model']['use_kernel_reg']
      self.kernel_reg = config['model']['kernel_reg']
      self.nfeatures = config['data']['num_features']
      self.batch_size = config['data']['batch_size']
      
      if self.use_kernel_reg:
         kernel_regularizer = tf.keras.regularizers.L2(self.kernel_reg)
      else:
         kernel_regularizer = None

      self.conv2d_A = ConvBnLayer(64,kernel_regularizer=kernel_regularizer)
      self.conv2d_B = ConvBnLayer(128,kernel_regularizer=kernel_regularizer)
      
      # reduce_max happens in between

      self.conv2d_C = ConvBnLayer(1024,kernel_regularizer=kernel_regularizer)

      # max pool

      # reshape

      self.dense_D = DenseBnLayer(512,kernel_regularizer=kernel_regularizer)
      self.dense_E = DenseBnLayer(256,kernel_regularizer=kernel_regularizer)

      # last layer in build function
      self.trans_output = TransformOutputLayer(self.nfeatures)

   def build(self,input_shape):
      num_points = input_shape[1]

      self.max_pool2d_C = tf.keras.layers.MaxPool2D(pool_size=[num_points,1],strides=[2,2])


   def call(self,edge_feature,training=False):

      # first layer
      net = self.conv2d_A(edge_feature,training)
      net = self.conv2d_B(net,training)
      net = tf.reduce_max(net, axis=-2, keepdims=True)
      net = self.conv2d_C(net,training)
      net = self.max_pool2d_C(net)

      net = tf.reshape(net, [self.batch_size, -1])

      net = self.dense_D(net,training)
      net = self.dense_E(net,training)

      # build transformation
      transform = self.trans_output(net,training)
      return transform


class DGCNN(tf.keras.Model):

   def __init__(self,config):
      super(DGCNN, self).__init__()
      self.knn_k = config['model']['knn']
      self.conv2d_size = config['model']['conv2d_size']
      self.dropout = config['model']['dropout']
      self.use_kernel_reg = config['model']['use_kernel_reg']
      self.kernel_reg = config['model']['kernel_reg']
      self.num_features = config['data']['num_features']
      self.num_classes = config['data']['num_classes']
      self.batch_size = config['data']['batch_size']
      self.num_points = config['data']['num_points']
      
      if self.use_kernel_reg:
         kernel_regularizer = tf.keras.regularizers.L2(self.kernel_reg)
      else:
         kernel_regularizer = None

      self.input_edge = EdgeLayer(self.knn_k)
      self.transform = InputTransformNet(config)

      self.edge_A   = EdgeLayer(self.knn_k)
      self.conv2d_A = ConvBnLayer(self.conv2d_size,kernel_regularizer=kernel_regularizer)
      self.conv2d_B = ConvBnLayer(self.conv2d_size,kernel_regularizer=kernel_regularizer)
      # reduce_max

      self.edge_C   = EdgeLayer(self.knn_k)
      self.conv2d_C = ConvBnLayer(self.conv2d_size,kernel_regularizer=kernel_regularizer)
      self.conv2d_D = ConvBnLayer(self.conv2d_size,kernel_regularizer=kernel_regularizer)
      # reduce_max

      self.edge_E   = EdgeLayer(self.knn_k)
      self.conv2d_E = ConvBnLayer(self.conv2d_size,kernel_regularizer=kernel_regularizer)

      self.conv2d_F = ConvBnLayer(1024,kernel_regularizer=kernel_regularizer)

      # max pool in build

      self.conv2d_G = ConvBnLayer(256,kernel_regularizer=kernel_regularizer)
      self.dropout_G = tf.keras.layers.Dropout(self.dropout)

      self.conv2d_H = ConvBnLayer(256,kernel_regularizer=kernel_regularizer)
      self.dropout_H = tf.keras.layers.Dropout(self.dropout)

      self.conv2d_I = ConvBnLayer(128,kernel_regularizer=kernel_regularizer)

      self.conv2d_J = tf.keras.layers.Conv2D(self.num_classes,[1,1],
                                             padding='valid',strides=[1,1],
                                             kernel_initializer='GlorotNormal',
                                             kernel_regularizer=kernel_regularizer)
   
   def build(self,input_shape):
      batch_size,num_points,nfeatures = input_shape

      self.max_pool2d_F = tf.keras.layers.MaxPool2D(pool_size=[num_points,1],strides=[2,2])
      self.num_points = num_points
      self.batch_size = batch_size
      self.nfeatures = nfeatures

   def call(self, point_cloud, training):
      # point_cloud shape: [batch,points,features]
      # calculate edge features of the graph
      edge_feature = self.input_edge(point_cloud,training)
      transform = self.transform(edge_feature, training)
      point_cloud_transformed = tf.matmul(point_cloud, transform)

      edge_feature = self.edge_A(point_cloud_transformed,training)
      net1 = self.conv2d_A(edge_feature,training)
      net1 = self.conv2d_B(net1,training)
      net1 = tf.reduce_max(net1, axis=-2, keepdims=True)

      edge_feature = self.edge_C(net1,training)
      net2 = self.conv2d_C(edge_feature,training)
      net2 = self.conv2d_D(net2,training)
      net2 = tf.reduce_max(net2, axis=-2, keepdims=True)

      edge_feature = self.edge_C(net2,training)
      net3 = self.conv2d_E(edge_feature,training)
      net3 = tf.reduce_max(net3, axis=-2, keepdims=True)

      combo_features = tf.concat([net1,net2,net3],axis=-1)
      net = self.conv2d_F(combo_features,training)
      net = self.max_pool2d_F(net)

      net = tf.tile(net, [1,self.num_points,1,1])

      net = tf.concat([net,net1,net2,net3],axis=3)

      net = self.conv2d_G(net,training)
      net = self.dropout_G(net,training=training)

      net = self.conv2d_H(net,training)
      net = self.dropout_H(net,training=training)

      net = self.conv2d_I(net,training)

      # this layer is ONLY a Conv2D
      logits = self.conv2d_J(net)
      # go from [batch,points,1,classes] to [batch,points,classes]
      logits = tf.squeeze(logits)
      
      return logits



