import numpy as np
import tensorflow as tf

class BaseModel:
    """
    The base model used for the binary classification
    """
    def __init__(self,size=1, layers=2,num_neurons=64,num_classes=5,input_ph=None,name_suf="Rep"):
        if input_ph is None:
            self.input_ph = tf.placeholder(tf.float32, shape=(None, size))
        else:
            self.input_ph = input_ph
        self.out = self.input_ph
        for i in range(layers):
            self.out = tf.layers.Dense(num_neurons,name = f"{name_suf}_Dense_{i}")(self.out)
            self.out = tf.layers.batch_normalization(self.out, training=True,name = f"{name_suf}_Batch_{i}")
            self.out = tf.nn.relu(self.out)

        self.logits = tf.layers.Dense(num_classes,name = f"{name_suf}_Dense_Out")(self.out)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.prediction = tf.argmax(self.logits,axis=-1)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.label_ph,num_classes),logits=self.logits)
        self.loss_op = tf.reduce_mean(self.loss)
        self.lossE = tf.constant(0.0)
        self.totalLoss = self.loss_op+self.lossE

        self.predictions = self.loss
        self.minimize_op = None
        

    def setTrainOP(self,train_op):
        self.minimize_op = train_op
"""
	The Chameleon component as described in the Paper
"""
class Chameleon:
    
    def __init__(self, input_shape,init=tf.glorot_uniform_initializer(),activation=tf.nn.relu,num_dense=2,permMatrix=True,regC=1.0,num_filter=128,num_conv=1,num_neurons=256,kernel_size=3,conv_act=False,name_suf="Perm"):

        self.input_shape = input_shape
        self.init = init
        self.activation = activation
        self.denseL = num_dense
        self.permMatrix = permMatrix
        self.name_suf = name_suf

        self.regC = regC 
        self.num_filter = num_filter
        self.num_conv = num_conv
        self.num_neurons = num_neurons
        self.conv_act = conv_act
        self.kernel_size = kernel_size
        
    def build(self):
        
        # Input task
        self.task = tf.placeholder(tf.float32, [self.input_shape[0],None])  ## (30,f)
        self.train_mode = tf.placeholder(tf.bool)
        self.label = tf.placeholder(tf.float32, [None,self.input_shape[1]])

        # Transposed
        self.inp   = tf.expand_dims(self.task,axis=0)
        self.trans = tf.transpose(self.inp,perm=[0,2,1]) #(1,f,30)
        
        # Conv layers
        self.conv1 = tf.layers.conv1d(self.trans,filters=self.num_filter,kernel_size=1,padding="SAME",name = f"{self.name_suf}_conv1")
        if self.conv_act:
            self.conv1 = tf.nn.relu(self.conv1)
        # (1,f,16)
        self.conv2 = self.conv1
        for l in range(self.num_conv):
            self.conv2 = tf.layers.conv1d(self.conv2,filters=self.num_filter,kernel_size=3,padding="SAME",name = f"{self.name_suf}_conv{l+2}")
            if self.conv_act:
               self.conv2 = tf.nn.relu(self.conv2)
        # (1,f,16)
        
        # Dense Layer
        self.dense1 = tf.squeeze(self.conv2) #(f,16)
        self.dense1  = tf.layers.Dense(self.num_neurons,name = f"{self.name_suf}_dense1")(tf.reshape(self.dense1,[-1,self.num_filter]))#(f,neurons)
        self.dense1  = tf.nn.relu(self.dense1)

        # Output Layer
        self.dense2  = tf.layers.Dense(self.input_shape[1],name = f"{self.name_suf}_dense2")(self.dense1) #(f,feats)
        self.perm  = tf.nn.softmax(self.dense2)
        self.permMat = self.getPermMatrix(self.perm) 

        # Regularize columns if reg>0
        self.regCols = tf.reduce_sum(self.perm,axis=0)
        self.regCols = (tf.ones(tf.shape([self.regCols]))-self.regCols)**2

        # If True, use discrete permutation matrix (not used in the paper)
        if self.permMatrix:
            self.out = tf.matmul(self.task,self.permMat)
        else:
        	# Used in the paper
            self.out = tf.matmul(self.task,self.perm)

        self.out = tf.reshape(self.out,self.input_shape)

        # Permutation loss
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.label,logits=self.dense2))+self.regC*tf.reduce_mean(self.regCols)

    # Takes a softmax alignment matrix and returns a true discrete permutation matrix
    def getPermMatrix(self,permutationMatrix):
        paddings = [[0, self.input_shape[1]-tf.shape(permutationMatrix)[0]], [0, 0]]
        prediction_full = tf.pad(permutationMatrix, paddings, 'CONSTANT', constant_values=0) # pad prediction to 13 features
        
        def delFeat(idx):
            feat_idx = tf.gather(tf.gather(p,idx),0)
            return tf.concat([tf.slice(tf.gather(orderedIn,idx),[0],[feat_idx]),tf.slice(tf.gather(orderedIn,idx),[feat_idx+1],[-1])],axis=0)

        def getColOrder(inp):
            return tf.nn.top_k(prediction_full,k=self.input_shape[1])[1][inp]
        
        
        rowOrder = tf.reverse(tf.contrib.framework.argsort(tf.reduce_max(prediction_full,axis=1)),axis=[0]) 
        orderedIn = tf.map_fn(getColOrder,rowOrder) 

        colOrder = []
        for i in range(self.input_shape[1]):
            firstEl = tf.gather(tf.gather(orderedIn,i),0)
            colOrder.append(firstEl)
            p = tf.slice(tf.where(tf.equal(orderedIn,firstEl)),[0,1],[-1,-1])
            o = []
            for j in range(self.input_shape[1]):
                o.append(delFeat(j))
            orderedIn = o

        perm = tf.matmul(tf.transpose(tf.one_hot(rowOrder,self.input_shape[1])),tf.one_hot(colOrder,self.input_shape[1]))
        perm = tf.slice(perm,[0,0],[tf.shape(permutationMatrix)[0],-1])
        
        return perm