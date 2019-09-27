import os
import time
import tensorflow as tf
import numpy as np
import ast

from .reptile import Reptile
from .dataGen import unison_shuffled_copies, padCols, loadData,loadHeartData
from .variables import weight_decay

"""
    Training call
        - Runs the Reptile algorithm for N epochs to metatrain evalModel and permModel
        - Hyperparameters are passed from the args parser
"""


def train(sess,
          repModel,
          evalModel,
          permModel,
          save_dir,
          num_classes=1,
          num_shots=60,
          min_class=1,
          max_class=10,
          feats=13,
          inner_batch_size=5,
          inner_iters=20,
          learning_rate = 0.0001,
          meta_step_size=0.1,
          meta_step_size_final=0.,
          meta_batch_size=1,
          meta_iters=15001,
          weight_decay_rate=1,
          reptile_fn=Reptile,
          perm_epochs=501,
          perm_lr = 0.00001,
          feature_split=0,
          bootstrap = False,
          permutation="[]",
          name="Model",
          dataset="",
          log_fn=print):


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Save scratch model vars, so they can be reset during training
    evalVars = []
    for v in tf.trainable_variables():
        if "EvalMod" in v.name:
            evalVars.append(v)

    # init reptile process
    reptile = reptile_fn(sess,pre_step_op=weight_decay(weight_decay_rate))
    saver = tf.train.Saver()

    # Loading Data
    start = time.time()
    f_split = feature_split

    # If feature split is used, boot is set to number of training features
    if bootstrap and f_split!=0:
        boot = f_split
    else:
        boot = 0

    # Load data set as defined in dataGen. Must be one of {"Wine","Abalone","Telescope","Heart"}
    if dataset == "Heart":
        X_train,y_train,X_test,y_test = loadHeartData(f"./data",perm=ast.literal_eval(permutation)) 
        X_pre = X_train
        y_pre = y_train
    else:
        if dataset == "Telescope":
            callD = "tele"
        elif dataset == "Abalone":
            callD = "aba"
        elif dataset == "Wine":
            callD = "wine"
        else:
            raise ValueError("Dataset argument must be one of: Wine, Abalone, Telescope, Heart")

        X_train,y_train,X_test,y_test = loadData(f"./data/{callD}.npy",split=0.8,featSplits=f_split,perm=ast.literal_eval(permutation)) 

        # Split the pre training data
        X_pre = X_train[:int(0.4*len(X_train))]
        y_pre = y_train[:int(0.4*len(X_train))]
        y_train = y_train[int(0.4*len(X_train)):]
        X_train = X_train[int(0.4*len(X_train)):]
        feats = len(X_test[0])

    log_fn(f"------------------Finished loading data in {time.time()-start}--------------------------")
    log_fn(f"Pretraing data shape: X{X_pre.shape} y{y_pre.shape}")
    log_fn(f"Training data shape: X{X_train.shape} y{y_train.shape}")
    log_fn(f"Test data shape: {X_test.shape} y{y_test.shape}")
    log_fn()


    # Declare the tensorflow graph
    # Each component has an optimizer and update ops
    if permModel is not None:
        # Optimizer here used for pretraining
        perm_opt = tf.train.AdamOptimizer(learning_rate=perm_lr)
        perm_gradients, perm_variables = zip(*perm_opt.compute_gradients(permModel.loss))
        perm_train_op = perm_opt.apply_gradients(zip(perm_gradients, perm_variables))

    rep_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    rep_gradients, rep_variables = zip(*rep_opt.compute_gradients(repModel.totalLoss))
    rep_train_op = rep_opt.apply_gradients(zip(rep_gradients, rep_variables))
    repModel.setTrainOP(rep_train_op)

    eval_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    eval_gradients, eval_variables = zip(*eval_opt.compute_gradients(evalModel.totalLoss))
    eval_train_op = eval_opt.apply_gradients(zip(eval_gradients, eval_variables))
    evalModel.setTrainOP(eval_train_op)

    #Initializing variables
    sess.run(tf.initializers.global_variables())
    init_eval_vars = tf.initialize_variables(evalVars)


    #Pre-train permutation model
    if permModel is not None:
        print()
        print("-------------------------------Training Chameleon network:--------------------------------")
        start = time.time()

        # Sample a random task for pre training from meta data set "data"
        def sampleTask(data,minF,maxF):
            features = np.random.choice(range(len(data[0])),np.random.randint(minF,maxF+1),replace=False) 
            out = np.transpose(np.array([data[:,i] for i in features]))
          
            return out,features

        trainLoss_buffer = []

        for epoch in range(perm_epochs):

            X_pre,y_pre = unison_shuffled_copies(X_pre,y_pre)

            loss_per_epoch= []
            val_loss_per_epoch= []
            val_acc_per_epoch = []

            for minibatch in range(int(len(X_pre)/inner_batch_size)):

                X = X_pre[inner_batch_size*minibatch:inner_batch_size*minibatch + inner_batch_size]
                X_perm,order = sampleTask(X,len(X[0]),len(X[0]))
                order = np.eye(feats)[order.astype(int)]

                loss,_ = sess.run([permModel.loss,perm_train_op],feed_dict={permModel.task: X_perm,permModel.label: order, permModel.train_mode:True})
                loss_per_epoch.append(loss)

            trainLoss_buffer.append(np.mean(loss_per_epoch))

            if epoch%5 == 0:
                log_fn(f"Epoch {epoch}: Permutation loss: {np.mean(trainLoss_buffer)}")
                trainLoss_buffer = []
    log_fn(f"Finished pre-training in {time.time()-start}s")
    log_fn("")
    log_fn("-------------------------------Training Chameleon and Base Model with reptile:---------------")
    # Evaluate the initialized model
    reptile.evaluate(repModel = repModel, evalModel = evalModel, permModel = permModel,
                     num_classes=num_classes, num_shots=num_shots,
                     inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                     init_evalVars=init_eval_vars,
                     meta_batch_size=meta_batch_size, min_class=min_class,
                     max_class=max_class, feats=feats, data_X=X_test,data_Y=y_test,bootstrap=boot)
                                   
    log_fn(f"Validation Epoch {0}: Train Loss: {reptile.loss_train:.5f} -- Val Loss: {reptile.loss_test[inner_iters]:.5f} -- Train Loss - Scratch: {reptile.loss_train_eval:.5f} -- Val Acc: {reptile.loss_test_acc[inner_iters-1]:.5f} -- Val Loss - Scratch: {reptile.loss_test_eval[inner_iters]:.5f} -- Val Acc - Scratch: {reptile.loss_test_acc_eval[inner_iters-1]:.5f} --")
    log_fn()       

    val_final = []
    train_final = []                  

    val_buffer = []
    train_buffer = []
    full_start = time.time()

    # Perform reptile joint training on the model
    for i in range(1,meta_iters):
        frac_done = i / meta_iters
        cur_meta_step_size = meta_step_size
        start = time.time()

        # Perform one train step
        reptile.train_step(repModel = repModel, evalModel = evalModel,permModel = permModel,
                           num_classes=num_classes, num_shots=num_shots,
                           inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                           init_evalVars=init_eval_vars,
                           meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size,
                           min_class=min_class,max_class=max_class, feats=feats,data_X=X_train, 
                           data_Y=y_train)

        train_final.append([reptile.loss_train,reptile.loss,reptile.loss_train_eval,reptile.loss_eval])
        train_buffer.append([reptile.loss_train,reptile.loss,reptile.loss_train_eval,reptile.loss_eval])
        if i%50 == 0:
            train_buffer = np.mean(train_buffer,axis=0)
            log_fn(f"Train Epoch {i}: Train Loss: {train_buffer[0]:.5f} -- Val Loss: {train_buffer[1]:.5f} -- Train Loss - Scratch: {train_buffer[2]:.5f} -- Val Loss - Scratch: {train_buffer[3]:.5f} --")
            train_buffer = []
        
        # Performs a validation step
        reptile.evaluate(repModel = repModel, evalModel = evalModel,permModel = permModel,
                         num_classes=num_classes, num_shots=num_shots,
                         inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                         init_evalVars=init_eval_vars,
                         meta_batch_size=meta_batch_size, min_class=min_class,
                         max_class=max_class, feats=feats, data_X=X_test,data_Y=y_test,bootstrap=boot)

        val_final.append([reptile.loss_train,reptile.loss_test[inner_iters-1],reptile.loss_train_eval,reptile.loss_test_eval[inner_iters-1],reptile.loss_test_acc_eval[inner_iters-1],reptile.loss_test_acc[inner_iters-1]])
        val_buffer.append([reptile.loss_train,reptile.loss_test[inner_iters-1],reptile.loss_train_eval,reptile.loss_test_eval[inner_iters-1],reptile.loss_test_acc_eval[inner_iters-1],reptile.loss_test_acc[inner_iters-1]])
        if i%50 == 0:
            val_buffer = np.mean(val_buffer,axis=0)
            log_fn(f"Validation Epoch {i}: Train Loss: {val_buffer[0]:.5f} -- Val Loss: {val_buffer[1]:.5f} -- Val Acc: {val_buffer[5]:.5f}-- Train Loss - Scratch: {val_buffer[2]:.5f} -- Val Loss - Scratch: {val_buffer[3]:.5f} -- Val Acc - Scratch: {val_buffer[4]:.5f}  --")            
            log_fn(f"In {time.time()-start}s")
            log_fn()
            val_buffer = []

    log_fn(f"Finished joint training in {time.time()-full_start}s")
    log_fn()

    # Print final score and save the model
    val_step = min(len(val_final),2000)
    val_final = np.mean(val_final[-val_step:],axis=0)
    train_final = np.mean(train_final[-val_step:],axis=0)
    log_fn(f"Final Training Scores: Train Loss: {train_final[0]:.5f} -- Val Loss: {train_final[1]:.5f} -- Train Loss - Scratch: {train_final[2]:.5f} -- Val Loss - Scratch: {train_final[3]:.5f} --")
    log_fn(f"Final Validation Scores: Train Loss: {val_final[0]:.5f} -- Val Loss: {val_final[1]:.5f} -- Val Acc: {val_final[5]:.5f}-- Train Loss - Scratch: {val_final[2]:.5f} -- Val Loss - Scratch: {val_final[3]:.5f} -- Val Acc - Scratch: {val_final[4]:.5f}  --")            

    save_path = f"{save_dir}/{name}"
    log_fn(f"Model saved to {save_path}")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    saver.save(sess, os.path.join(save_path, f'{name}.ckpt'))
