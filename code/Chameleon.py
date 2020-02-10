## Code created by: Lukas Brinkmeyer and Rafael Rego Drumond

import tensorflow as tf
import numpy as np

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