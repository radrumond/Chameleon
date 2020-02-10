## Code modified by: Lukas Brinkmeyer and Rafael Rego Drumond

import numpy as np
import tensorflow as tf

class BaseModel:
    """
    The base model used for the binary classification
    """
    def __init__(self,size=1, layers=[64,64],num_classes=5,input_ph=None,name_suf="Rep",frozen=False,untrained=False):
        if input_ph is None:
            self.input_ph = tf.placeholder(tf.float32, shape=(None, size))
        else:
            self.input_ph = input_ph
        self.out = self.input_ph
        for i,num_neurons in enumerate(layers):
            self.out = tf.layers.Dense(num_neurons,name = f"{name_suf}_Dense_{i}")(self.out)
            self.out = tf.layers.batch_normalization(self.out, training=True,name = f"{name_suf}_Batch_{i}")
            self.out = tf.nn.relu(self.out)

        self.logits = tf.layers.Dense(num_classes,name = f"{name_suf}_Dense_Out")(self.out)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,None))
        self.prediction = tf.argmax(self.logits,axis=-1)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label_ph,logits=self.logits)
        self.loss_op = tf.reduce_mean(self.loss)
        self.lossE = tf.constant(0.0)
        self.totalLoss = self.loss_op+self.lossE

        self.predictions = self.loss
        self.minimize_op = None
        

        self.frozen = frozen
        self.untrained = untrained

    def setTrainOP(self,train_op):
        self.minimize_op = train_op
"""
	The Chameleon component as described in the Paper
"""
class Chameleon:
    
    def __init__(self, num_instances, maxK,init=tf.glorot_uniform_initializer(),activation=tf.nn.relu,name_suf="New",conv_layers=[8,16,14]):

        self.num_instances = num_instances
        self.maxK        = maxK
        self.init        = init
        self.activation  = activation
        self.name_suf    = name_suf
        self.conv_layers = conv_layers
        
    def build(self):
        
        # Input task
        self.task  = tf.placeholder(tf.float32, [None,self.num_instances,None])  ## (MB,Instances,FeaturesBefore)
        self.train = tf.placeholder(tf.bool)
        self.label = tf.placeholder(tf.float32, [None,None,self.maxK]) #(MB,FeaturesBefore,featuersAfter)

        # Transposed
        self.trans = tf.transpose(self.task,perm=[0,2,1]) #(1,f,30)
        self.conv  = self.trans
        
        for idx,c in enumerate(self.conv_layers):
            self.conv = tf.keras.layers.Conv1D(filters=self.conv_layers[idx],kernel_size=1,padding="SAME",name = f"{self.name_suf}_conv{idx}")(self.conv)
            self.conv = tf.nn.relu(self.conv)
        
        self.logit = tf.keras.layers.Conv1D(filters=self.maxK,kernel_size=1,padding="SAME",name = f"{self.name_suf}_conv3")(self.conv)
        self.perm = tf.nn.softmax(self.logit)
        
        self.out = tf.matmul(self.task,self.perm)

        
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.label,logits=self.logit))

def one_hot(targets,nb_classes):    
    return np.eye(nb_classes)[targets]