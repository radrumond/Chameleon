## Code modified by: Lukas Brinkmeyer and Rafael Rego Drumond
## Original code at: https://github.com/openai/supervised-reptile
"""
Supervised Reptile learning and evaluation on arbitrary
datasets as stated in https://github.com/openai/supervised-reptile
"""

import random
import numpy as np
import tensorflow as tf
import time

from .variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)
from .dataGen import unison_shuffled_copies, padCols
from .saver import savePlot

"""
    Reptile class
"""
class Reptile:


    def __init__(self, session, variables=None):#, pre_step_op=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        
        #self._full_state = VariableState(self.session, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    def train_step(self, repModel, permModel, inner_iters, meta_step_size, data, train_f, exp_name, meta_epoch=0,name="Run"):

        old_vars = self._model_state.export_variables()
        new_vars = []

        self.run_train_losses     = []
        self.run_test_loss_before = []
        self.run_test_loss_after  = []

        # If Chameleon component is used, the input is taken from it, otherwise the input is taken from the base model
        if permModel is not None:
            input_tensor     = permModel.task
        else:
            input_tensor     = repModel.input_ph

        mb_train_x, mb_train_y, mb_test_x, mb_test_y = next(data)

        for task in range(len(mb_train_x)):
            # Perform one task update

            losses = []
            if permModel is not None:
                    inp = [mb_test_x[task]]
            else:
                inp = padCols(mb_test_x[task],train_f)
            loss      = self.session.run(repModel.loss_op, feed_dict={input_tensor: inp, repModel.label_ph: mb_test_y[task]})
            self.run_test_loss_before.append(loss)
            # Train on sampled task for 10 epochs and perform reptile update
            for epoch in range(inner_iters):

                # Train update
                if permModel is not None:
                    inp = [mb_train_x[task]]
                else:
                    inp = padCols(mb_train_x[task],train_f)
                _,loss = self.session.run([repModel.minimize_op, repModel.loss_op], feed_dict={input_tensor: inp, repModel.label_ph: mb_train_y[task]})     
                losses.append(loss)         
                
            # Compute final validation loss
            if permModel is not None:
                inp = [mb_test_x[task]]
                permName = "Chameleon+Reptile"
            else:
                inp = padCols(mb_test_x[task],train_f)
                permName = "Reptile"

            loss      = self.session.run(repModel.loss_op, feed_dict={input_tensor: inp, repModel.label_ph: mb_test_y[task]})
            self.run_test_loss_after.append(loss)
            self.run_train_losses.append(losses)

            if repModel.frozen:
                permName+="_frozen"
            if repModel.untrained:
                permName+="_Untrained"

            if meta_epoch % 1000==0:
                savePlot(losses,"Plots",exp_name,"Inner_Metatrain_Loss",permName+f"_{meta_epoch}_{task}",xaxis="Inner Epoch",yaxis="Loss",run=name)
            # Reptile update
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)

        self.run_train_losses     = np.mean(np.array(self.run_train_losses),axis=0)
        self.run_test_loss_before = np.mean(np.array(self.run_test_loss_before),axis=0)
        self.run_test_loss_after  = np.mean(np.array(self.run_test_loss_after),axis=0)

        # Reptile update
        new_vars = average_vars(new_vars)
        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))


    def testScratch(self, evalModel, inner_iters, init_evalVars, data, train_f):

        self.scratch_loss = []
        self.scratch_acc  = []

        gen_len = 0

        for mb in data:
            gen_len+=1
            mb_train_x, mb_train_y, mb_test_x, mb_test_y = mb

            for task in range(len(mb_train_x)):
                # Reset eval network
                self.session.run(init_evalVars)

                # Train network
                for epoch in range(inner_iters):
                    self.session.run(evalModel.minimize_op, feed_dict={evalModel.input_ph: padCols(mb_train_x[task],train_f), evalModel.label_ph: mb_train_y[task]})                

                # Compute test loss
                test_loss,test_pred = self.session.run([evalModel.loss_op,evalModel.prediction], feed_dict={evalModel.input_ph: padCols(mb_test_x[task],train_f), evalModel.label_ph: mb_test_y[task]})
                # Results
                self.scratch_loss.append(test_loss)
                self.scratch_acc.append(np.mean(np.argmax(mb_test_y[task],-1) == test_pred))

        # Aggregate
        self.scratch_loss =  np.mean(np.array(self.scratch_loss),axis=0)
        self.scratch_acc  =  np.mean(np.array(self.scratch_acc),axis=0)

    def evaluate(self, repModel, permModel, inner_iters, data, train_f, exp_name, meta_epoch=0,name="Run"):

        old_vars = self._model_state.export_variables()

        # Logs
        self.eval_train_losses = []
        self.eval_test_loss    = []
        self.eval_test_acc     = []

        # If Chameleon component is used, the input is taken from it, otherwise the input is taken from the base model
        if permModel is not None:
            input_tensor     = permModel.task
        else:
            input_tensor     = repModel.input_ph

        for mb in data:

            mb_train_x, mb_train_y, mb_test_x, mb_test_y = mb

            for task in range(len(mb_train_x)):
                
                losses        = []             
                loss_test     = []
                loss_test_acc = []

                # Compute test loss before training
                if permModel is not None:
                    inp = [mb_test_x[task]]
                else:
                    inp = padCols(mb_test_x[task],train_f)
                l_test,pred_class = self.session.run([repModel.loss_op,repModel.prediction], feed_dict={input_tensor: inp, repModel.label_ph: mb_test_y[task]})
                
                #Log Test Results before training
                loss_test.append(l_test)
                loss_test_acc.append(np.mean(np.argmax(mb_test_y[task],-1) == pred_class))

                if permModel is not None:
                    inp = [mb_train_x[task]]
                else:
                    inp = padCols(mb_train_x[task],train_f)
                for epoch in range(inner_iters):
                    _,loss = self.session.run([repModel.minimize_op,repModel.loss_op], feed_dict={input_tensor: inp, repModel.label_ph: mb_train_y[task]})           
                    losses.append(loss)

                # Compute test loss
                if permModel is not None:
                    inp = [mb_test_x[task]]
                    permName = "Chameleon+Reptile"
                else:
                    inp = padCols(mb_test_x[task],train_f)
                    permName = "Reptile"
                l_test,pred_class      = self.session.run([repModel.loss_op,repModel.prediction], feed_dict={input_tensor: inp, repModel.label_ph: mb_test_y[task]})
                #Log loss
                self.eval_test_loss.append(l_test)
                self.eval_test_acc.append(np.mean(np.argmax(mb_test_y[task],-1) == pred_class))
                self.eval_train_losses.append(losses)

                if repModel.frozen:
                    permName+="_frozen"
                if repModel.untrained:
                    permName+="_Untrained"

                if meta_epoch % 1000==0:
                    savePlot(losses,"Plots",exp_name,"Inner_Metatest_Loss",permName+f"_{meta_epoch}_{task}",xaxis="Inner Epoch",yaxis="Loss",run=name)

                #Reset networks
                self._model_state.import_variables(old_vars)

        self.eval_train_losses    =  np.mean(np.array(self.eval_train_losses),axis=0)
        self.eval_test_loss =  np.mean(np.array(self.eval_test_loss),axis=0)
        self.eval_test_acc  =  np.mean(np.array(self.eval_test_acc),axis=0)