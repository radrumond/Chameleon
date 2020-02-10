"""
Supervised Reptile learning and evaluation on arbitrary
datasets as stated in https://github.com/openai/supervised-reptile
"""

import random
import numpy as np
import tensorflow as tf
import time

from sklearn.model_selection import train_test_split
from .variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)
from .dataGen import unison_shuffled_copies, sample_mini_dataset, mini_batches, padCols

"""
	Reptile class
"""
class Reptile:


    def __init__(self, session, variables=None, pre_step_op=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._pre_step_op = pre_step_op
        self.loss = 0
        self.loss_before = 0

    def train_step(self,
                   repModel,
                   evalModel,
                   permModel,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   init_evalVars,
                   min_class,
                   max_class,
                   feats,
                   meta_step_size,
                   meta_batch_size,
                   data_X,
                   data_Y):

        old_vars = self._model_state.export_variables()
        new_vars = []

        # Logs
        self.loss_before = 0
        self.loss = 0
        self.loss_before_eval = 0
        self.loss_eval = 0
        self.loss_train = 0
        self.loss_train_eval = 0

        # If Chameleon component is used, the input is taken from it, otherwise the input is taken from the base model
        if permModel is not None:
            input_tensor     = permModel.task
            pad = 0 
        else:
            input_tensor     = repModel.input_ph
            pad = feats

        for _ in range(meta_batch_size):
        	# Perform one task update

            losses = []

            # Reset eval network
            self.session.run(init_evalVars)

            # Sample task data set
            mini_dataset = sample_mini_dataset(num_classes, num_shots,min_class,max_class,data_X,data_Y,pad)
            test_batch = []

            # Split off validation data for sampled task
            for ib in range(inner_batch_size):
                test_batch.append(next(mini_dataset))

            # Train on sampled task for 10 epochs and perform reptile update
            for epoch,batch in enumerate(mini_batches(mini_dataset, inner_batch_size, inner_iters, False)):

                inputs, labels = zip(*batch)

                if self._pre_step_op:
                    self.session.run(self._pre_step_op)

                # Compute Initialization loss
                if epoch == 0:
                    loss      = self.session.run(repModel.loss_op, feed_dict={input_tensor: np.array(inputs), repModel.label_ph: labels})
                    loss_eval = self.session.run(evalModel.loss_op, feed_dict={evalModel.input_ph: padCols(np.array(inputs),feats), evalModel.label_ph: labels})
                    self.loss_before      += loss
                    self.loss_before_eval += loss_eval

                # Train update
                self.session.run(repModel.minimize_op, feed_dict={input_tensor: np.array(inputs), repModel.label_ph: labels})
                self.session.run(evalModel.minimize_op, feed_dict={evalModel.input_ph: padCols(np.array(inputs),feats), evalModel.label_ph: labels})                

            # Compute the final training loss
            loss      = self.session.run(repModel.loss_op, feed_dict={input_tensor: np.array(inputs), repModel.label_ph: labels})
            loss_eval = self.session.run(evalModel.loss_op, feed_dict={evalModel.input_ph: padCols(np.array(inputs),feats), evalModel.label_ph: labels})
            self.loss_train      += loss
            self.loss_train_eval += loss_eval
                
            # Compute final validation loss
            inputs, labels = zip(*test_batch) 
            loss      = self.session.run(repModel.loss_op, feed_dict={input_tensor: np.array(inputs), repModel.label_ph: labels})
            loss_eval = self.session.run(evalModel.loss_op, feed_dict={evalModel.input_ph: padCols(np.array(inputs),feats), evalModel.label_ph: labels})
            self.loss      += loss
            self.loss_eval += loss_eval

            # Reptile update
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)

        self.loss_before      =  self.loss_before/float(meta_batch_size) 
        self.loss             =  self.loss/float(meta_batch_size) 
        self.loss_before_eval =  self.loss_before_eval/float(meta_batch_size)
        self.loss_eval        =  self.loss_eval/float(meta_batch_size) 
        self.loss_train       =  self.loss_train/float(meta_batch_size) 
        self.loss_train_eval  =  self.loss_train_eval/float(meta_batch_size) 
        new_vars = average_vars(new_vars)
        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))

    def evaluate(self,
                 repModel,
                 evalModel,
                 permModel,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 init_evalVars,
                 min_class,
                 max_class,
                 feats,
                 meta_batch_size,
                 data_X,
                 data_Y,
                 bootstrap):

        old_vars = self._model_state.export_variables()

        # Logs
        self.loss_before      = 0
        self.loss_before_eval = 0
        self.loss_train       = 0
        self.loss_train_eval  = 0

        self.loss_test      = []
        self.loss_test_eval = []

        self.loss_test_acc    	 = []
        self.loss_test_acc_eval  = []


        # If Chameleon component is used, the input is taken from it, otherwise the input is taken from the base model
        if permModel is not None:
            input_tensor     = permModel.task
            pad = 0 
        else:
            input_tensor     = repModel.input_ph
            pad = feats
        for _ in range(meta_batch_size):
            
            losses = []
            # Reset eval network
            self.session.run(init_evalVars)
                
            # Split off test data
            mini_dataset = sample_mini_dataset(num_classes, num_shots,min_class,max_class,data_X,data_Y,pad,bootstrap)
            test_batch = [] 
            for ib in range(inner_batch_size):
                test_batch.append(next(mini_dataset))
            inputs_test, labels_test = zip(*test_batch) 

            # Log reptile
            loss_test      = []
            loss_test_eval = []

            # Log scratch
            loss_test_acc       = []
            loss_test_acc_eval  = []

            # Compute test loss before training
            l_test,pred_class      = self.session.run([repModel.loss_op,repModel.prediction], feed_dict={input_tensor: np.array(inputs_test), repModel.label_ph: labels_test})
            l_eval_test,pred_class_eval = self.session.run([evalModel.loss_op,evalModel.prediction], feed_dict={evalModel.input_ph: padCols(np.array(inputs_test),feats), evalModel.label_ph: labels_test})
            
            #Log loss
            loss_test.append(l_test)
            loss_test_eval.append(l_eval_test)

            #Log acc
            loss_test_acc.append(np.mean(labels_test == pred_class))
            loss_test_acc_eval.append(np.mean(labels_test == pred_class_eval))

            for epoch,batch in enumerate(mini_batches(mini_dataset, inner_batch_size, inner_iters, False)):
                # One update step on the minibatch
                inputs, labels = zip(*batch)

                if self._pre_step_op:
                    self.session.run(self._pre_step_op)

                if epoch == 0:
                    loss      = self.session.run(repModel.loss_op, feed_dict={input_tensor: np.array(inputs), repModel.label_ph: labels})
                    loss_eval = self.session.run(evalModel.loss_op, feed_dict={evalModel.input_ph: padCols(np.array(inputs),feats), evalModel.label_ph: labels})
                    self.loss_before      += loss
                    self.loss_before_eval += loss_eval

                self.session.run(repModel.minimize_op, feed_dict={input_tensor: np.array(inputs), repModel.label_ph: labels})
                self.session.run(evalModel.minimize_op, feed_dict={evalModel.input_ph: padCols(np.array(inputs),feats), evalModel.label_ph: labels})                

                # Track the first loss (before training on specific task) and the last loss
                if epoch == inner_iters-1:
                    loss      = self.session.run(repModel.loss_op, feed_dict={input_tensor: np.array(inputs), repModel.label_ph: labels})
                    loss_eval = self.session.run(evalModel.loss_op, feed_dict={evalModel.input_ph: padCols(np.array(inputs),feats), evalModel.label_ph: labels})
                    self.loss_train      += loss
                    self.loss_train_eval += loss_eval

                # Compute test loss
                l_test,pred_class      = self.session.run([repModel.loss_op,repModel.prediction], feed_dict={input_tensor: np.array(inputs_test), repModel.label_ph: labels_test})
                l_eval_test,pred_class_eval = self.session.run([evalModel.loss_op,evalModel.prediction], feed_dict={evalModel.input_ph: padCols(np.array(inputs_test),feats), evalModel.label_ph: labels_test})
                
                #Log loss
                loss_test.append(l_test)
                loss_test_eval.append(l_eval_test)

                #Log acc
                loss_test_acc.append(np.mean(labels_test == pred_class))
                loss_test_acc_eval.append(np.mean(labels_test == pred_class_eval))

            #Log loss
            self.loss_test.append(loss_test)
            self.loss_test_eval.append(loss_test_eval)

            #Log acc
            self.loss_test_acc.append(loss_test_acc)
            self.loss_test_acc_eval.append(loss_test_acc_eval)

            #Reset networks
            self._model_state.import_variables(old_vars)

        self.loss_before      =  self.loss_before/float(meta_batch_size) 
        self.loss_before_eval =  self.loss_before_eval/float(meta_batch_size)
        self.loss_train       =  self.loss_train/float(meta_batch_size) 
        self.loss_train_eval  =  self.loss_train_eval/float(meta_batch_size) 

        # Val logs
        self.loss_test        =  np.mean(np.array(self.loss_test),axis=0)
        self.loss_test_eval   =  np.mean(np.array(self.loss_test_eval),axis=0)
        self.loss_test_acc    =  np.mean(np.array(self.loss_test_acc),axis=0)
        self.loss_test_acc_eval    =  np.mean(np.array(self.loss_test_acc_eval),axis=0)