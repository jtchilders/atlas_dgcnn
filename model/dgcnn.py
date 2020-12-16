import tensorflow as tf,logging
import numpy as np
logger = logging.getLogger(__name__)


def get_model(config):
   net = DGCNN(config)
   return net

class InputTransformNet(tf.keras.Model):
   """ Input (XYZ) Transform Net, input is BxNx3 gray image
    Return:
      Transformation matrix of size 3xK """
   def __init__(self,config,nfeatures):
      super(InputTransformNet,self).__init__()
      self.batch_size = config['data']['batch_size']
      self.num_points = config['data']['num_points']
      self.nfeatures = nfeatures

      # first layer
      self.conv2d_A = tf.keras.layers.Conv2D(64,[1,1],
                        padding='valid',
                        strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))
      self.bn_A = tf.keras.layers.BatchNormalization()
      self.relu = tf.keras.layers.ReLU()

      # second layer
      self.conv2d_B = tf.keras.layers.Conv2D(128,[1,1],
                        padding='valid',
                        strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))
      self.bn_B = tf.keras.layers.BatchNormalization()

      # third layer
      self.conv2d_C = tf.keras.layers.Conv2D(1024,[1,1],
                        padding='valid',
                        strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))
      self.bn_C = tf.keras.layers.BatchNormalization()
      self.max_pool2d_C = tf.keras.layers.MaxPool2D(pool_size=[self.num_points,1],strides=[2,2])

      # fourth layer
      self.dense_D = tf.keras.layers.Dense(512,kernel_regularizer=tf.keras.regularizers.L2(1e-3))
      self.bn_D = tf.keras.layers.BatchNormalization()
      # fifth layer
      self.dense_E = tf.keras.layers.Dense(256,kernel_regularizer=tf.keras.regularizers.L2(1e-3))
      self.bn_E = tf.keras.layers.BatchNormalization()

      # sixth layer
      self.const_init = tf.constant_initializer(0.0)
      self.input_transform_weights = tf.Variable(self.const_init([256,self.nfeatures * self.nfeatures],dtype=tf.float32),name='input_transform_weights')
      self.input_transform_biases = tf.Variable(self.const_init([self.nfeatures * self.nfeatures],dtype=tf.float32),name='input_transform_biases')
      self.input_transform_biases.assign_add(tf.constant(np.eye(self.nfeatures).flatten(), dtype=tf.float32))

   def call(self,edge_feature,point_cloud,training=False):

      # first layer
      net = self.conv2d_A(edge_feature)
      net = self.bn_A(net,training)
      net = self.relu(net)

      # second layer
      net = self.conv2d_B(net)
      net = self.bn_B(net,training)
      net = self.relu(net)

      net = tf.reduce_max(net, axis=-2, keepdims=True)

      # third layer
      net = self.conv2d_C(net)
      net = self.bn_C(net,training)
      net = self.relu(net)
      net = self.max_pool2d_C(net)

      net = tf.reshape(net, [self.batch_size, -1])

      # fourth layer
      net = self.dense_D(net)
      net = self.bn_D(net,training)
      net = self.relu(net)

      # fifth layer
      net = self.dense_E(net)
      net = self.bn_E(net,training)
      net = self.relu(net)

      # build transformation
      transform = tf.matmul(net, self.input_transform_weights)
      transform = tf.nn.bias_add(transform, self.input_transform_biases)
      transform = tf.reshape(transform, [self.batch_size, self.nfeatures, self.nfeatures])

      return transform


class DGCNN(tf.keras.Model):

   def __init__(self,config):
      super(DGCNN, self).__init__()
      self.knn_k = config['model']['knn']
      self.num_features = config['data']['num_features']
      self.num_classes = config['data']['num_classes']
      self.batch_size = config['data']['batch_size']
      self.num_points = config['data']['num_points']
      
      self.transform = InputTransformNet(config,self.num_features)

      self.conv2d_A = tf.keras.layers.Conv2D(64,[1,1],padding='valid',strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))
      self.bn_A = tf.keras.layers.BatchNormalization()
      self.relu = tf.keras.layers.ReLU()

      self.conv2d_B = tf.keras.layers.Conv2D(64,[1,1],padding='valid',strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))
      self.bn_B = tf.keras.layers.BatchNormalization()

      self.conv2d_C = tf.keras.layers.Conv2D(64,[1,1],padding='valid',strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))
      self.bn_C = tf.keras.layers.BatchNormalization()

      self.conv2d_D = tf.keras.layers.Conv2D(64,[1,1],padding='valid',strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))
      self.bn_D = tf.keras.layers.BatchNormalization()

      self.conv2d_E = tf.keras.layers.Conv2D(64,[1,1],padding='valid',strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))
      self.bn_E = tf.keras.layers.BatchNormalization()

      self.conv2d_F = tf.keras.layers.Conv2D(1024,[1,1],padding='valid',strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))
      self.bn_F = tf.keras.layers.BatchNormalization()

      self.max_pool2d_F = tf.keras.layers.MaxPool2D(pool_size=[self.num_points,1],strides=[2,2])

      self.conv2d_G = tf.keras.layers.Conv2D(256,[1,1],padding='valid',strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))

      self.dropout_G = tf.keras.layers.Dropout(0.6)

      self.conv2d_H = tf.keras.layers.Conv2D(256,[1,1],padding='valid',strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))

      self.dropout_H = tf.keras.layers.Dropout(0.6)
      
      self.conv2d_I = tf.keras.layers.Conv2D(128,[1,1],padding='valid',strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))

      self.conv2d_J = tf.keras.layers.Conv2D(self.num_classes,[1,1],padding='valid',strides=[1,1],
                        kernel_initializer='GlorotNormal',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-3))
   
   def call(self, point_cloud, training=False):

      # calculate edge features of the graph
      adj = self.pairwise_distance(point_cloud)
      nn_idx = self.knn(adj, k=self.knn_k)
      input_image = tf.expand_dims(point_cloud, -1)
      edge_feature = self.get_edge_feature(input_image, nn_idx=nn_idx, k=self.knn_k)
      
      transform = self.transform(edge_feature, training)
      point_cloud_transformed = tf.matmul(point_cloud, transform)

      input_image = tf.expand_dims(point_cloud_transformed, -1)
      adj = self.pairwise_distance(point_cloud_transformed)
      nn_idx = self.knn(adj, k=self.knn_k)
      edge_feature = self.get_edge_feature(input_image, nn_idx=nn_idx, k=self.knn_k)

      net1 = self.conv2d_A(edge_feature)
      net1 = self.bn_A(net1,training)
      net1 = self.relu(net1)
      
      net1 = self.conv2d_B(net1)
      net1 = self.bn_B(net1,training)
      net1 = self.relu(net1)

      net1 = tf.reduce_max(net1, axis=-2, keepdims=True)

      adj = self.pairwise_distance(net1)
      nn_idx = self.knn(adj, k=self.knn_k)
      edge_feature = self.get_edge_feature(net1, nn_idx=nn_idx, k=self.knn_k)

      net2 = self.conv2d_C(edge_feature)
      net2 = self.bn_C(net2,training)
      net2 = self.relu(net2)
      
      net2 = self.conv2d_D(net2)
      net2 = self.bn_D(net2,training)
      net2 = self.relu(net2)

      net2 = tf.reduce_max(net2, axis=-2, keepdims=True)

      adj = self.pairwise_distance(net2)
      nn_idx = self.knn(adj, k=self.knn_k)
      edge_feature = self.get_edge_feature(net2, nn_idx=nn_idx, k=self.knn_k)

      net3 = self.conv2d_E(edge_feature)
      net3 = self.bn_E(net3,training)
      net3 = self.relu(net3)
      
      net3 = tf.reduce_max(net3, axis=-2, keepdims=True)

      combo_features = tf.concat([net1,net2,net3],axis=-1)
      net = self.conv2d_F(combo_features)
      net = self.bn_F(net,training)
      net = self.relu(net)

      net = self.max_pool2d_F(net)

      net = tf.tile(net, [1,self.num_points,1,1])

      net = tf.concat([net,net1,net2,net3],axis=3)

      net = self.conv2d_G(net)
      net = self.dropout_G(net,training)

      net = self.conv2d_H(net)
      net = self.dropout_H(net,training)

      net = self.conv2d_I(net)

      net = self.conv2d_J(net)

      net = tf.reshape(net,[self.batch_size,self.num_points,self.num_classes])
      
      return net

   @staticmethod
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
      point_cloud_inner = -2 * point_cloud_inner
      point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True)
      point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
      return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

   @staticmethod
   def knn(adj_matrix, k=20, returnDist=False):
      """Get KNN based on the pairwise distance.
      Args:
       pairwise distance: (batch_size, num_points, num_points)
       k: int

      Returns:
       nearest neighbors: (batch_size, num_points, k)
      """
      neg_adj = -adj_matrix
      dist, nn_idx = tf.nn.top_k(neg_adj, k=tf.cast(k,tf.int32))
      if not returnDist:
         return nn_idx
      else:
         return nn_idx, dist

   @staticmethod
   def get_edge_feature(point_cloud, nn_idx, k=20,concat=True):
      """Construct edge feature for each point
      Args:
       point_cloud: (batch_size, num_points, 1, num_dims)
       nn_idx: (batch_size, num_points, k)
       k: int

      Returns:
       edge features: (batch_size, num_points, k, num_dims)
      """
      og_batch_size = point_cloud.get_shape().as_list()[0]
      point_cloud = tf.squeeze(point_cloud)
      if og_batch_size == 1:
         point_cloud = tf.expand_dims(point_cloud, 0)

      point_cloud_central = point_cloud

      idx_ = tf.range(point_cloud.shape[0]) * point_cloud.shape[1]
      idx_ = tf.reshape(idx_, [point_cloud.shape[0], 1, 1])

      point_cloud_flat = tf.reshape(point_cloud, [-1, point_cloud.shape[2]])
      point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
      point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

      point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

      if concat:
         edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
         return edge_feature
      else:
         return point_cloud_neighbors


