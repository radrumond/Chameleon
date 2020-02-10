## Code modified by: Lukas Brinkmeyer and Rafael Rego Drumond
## Original code at: https://github.com/openai/supervised-reptile
import os
import time
import tensorflow as tf
import numpy as np
import ast

from .reptile import Reptile
from .dataGen import unison_shuffled_copies, padCols

saving_plots = False
saving_raw = True

if saving_plots:
    from .saver import savePlot
"""
    Training call
        - Runs the Reptile algorithm for N epochs to metatrain evalModel and permModel
        - Hyperparameters are passed from the args parser
"""
def id_print(id,*text):
    print(f"Job-{id}: ",*text)

def train(sess,
          repModel,
          evalModel,
          permModel,
          save_dir,
          dataset,
          oracle,
          freeze,
          min_feats="1",
          max_feats="10",
          inner_batch_size=5,
          inner_iters=20,
          learning_rate = 0.0001,
          meta_step_size=0.1,
          meta_batch_size=1,
          meta_iters=15001,
          reptile_fn=Reptile,

          perm_epochs=501,
          perm_lr = 0.00001,
          feature_split=0,

          name="Model",
          name_affix="",
          save_path="exp1",        
          log_fn=id_print,
          job_id=0):


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if saving_raw:
        raw_path = "./results/"+save_path+"/"+name

        if not os.path.exists(raw_path):
            os.system(f"mkdir -p {raw_path}")

        if not os.path.exists(os.path.join(raw_path,name_affix)):
            os.mkdir(os.path.join(raw_path,name_affix))

    # Save scratch model vars, so they can be reset during training
    evalVars = []
    for v in tf.trainable_variables():
        if "EvalMod" in v.name:
            evalVars.append(v)

    # init reptile process
    reptile = reptile_fn(sess)
    saver = tf.train.Saver()

    # Loading Data
    start = time.time()
    train_gen = dataset.generate(dataset.totalLabels, ast.literal_eval(min_feats), ast.literal_eval(max_feats), inner_batch_size, inner_batch_size, meta_batch_size, test=False, oracle=oracle)

    if oracle:
        test_data = dataset.test_data_oracle
    else:
        test_data = dataset.test_data

    log_fn(job_id,f"------------------Finished loading data in {time.time()-start}--------------------------")
    log_fn(job_id,f"Training data shape: X{dataset.train_x.shape} y{dataset.train_y.shape}")
    log_fn(job_id,f"Test data shape: X{dataset.val_x.shape} y{dataset.val_y.shape}")
    log_fn(job_id,"")

    # Declare the tensorflow graph
    # Each component has an optimizer and update ops
    if permModel is not None:
        # Optimizer here used for pretraining
        perm_opt = tf.train.AdamOptimizer(learning_rate=perm_lr)
        perm_gradients, perm_variables = zip(*perm_opt.compute_gradients(permModel.loss))
        perm_train_op = perm_opt.apply_gradients(zip(perm_gradients, perm_variables))

    
    if not freeze:
        rep_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        rep_gradients, rep_variables = zip(*rep_opt.compute_gradients(repModel.totalLoss))
        rep_train_op = rep_opt.apply_gradients(zip(rep_gradients, rep_variables))
        repModel.setTrainOP(rep_train_op)
    else:
        rep_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        rep_train_op = rep_opt.minimize(repModel.totalLoss,var_list = [var for var in tf.trainable_variables() if "Perm" not in var.name and "RepMod" in var.name])
        repModel.setTrainOP(rep_train_op)

    if evalModel is not None:
        eval_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        eval_gradients, eval_variables = zip(*eval_opt.compute_gradients(evalModel.totalLoss))
        eval_train_op = eval_opt.apply_gradients(zip(eval_gradients, eval_variables))
        evalModel.setTrainOP(eval_train_op)

    

    #Initializing variables
    sess.run(tf.initializers.global_variables())
    if evalModel is not None:
        init_eval_vars = tf.initialize_variables(evalVars)

    #Pre-train permutation model
    if permModel is not None:
        log_fn(job_id,"")
        log_fn(job_id,"-------------------------------Training Chameleon network:-------------------------------")
        start = time.time()

        # Sample a random task for pre training from meta data set "data"
        def sampleTask(data,minF,maxF,maxK, meta_batch=1):

            meta_x = []
            meta_y = []

            num_feat = np.random.randint(minF,maxF+1)
            for b in range(meta_batch):
                features = np.random.choice(range(len(data[0])),num_feat,replace=False) 
                out = np.transpose(np.array([data[:,i] for i in features]))
                order = np.eye(maxK)[features.astype(int)]

                meta_x.append(out)
                meta_y.append(order)

            return meta_x,meta_y

        trainLoss_buffer = []
        train_losses = []
        train_indexes = []

        for epoch in range(1,perm_epochs+1):

            X_pre= dataset.train_x[:,dataset.train_f]
            np.random.shuffle(X_pre)

            loss_per_epoch= []
            val_loss_per_epoch= []
            val_acc_per_epoch = []

            for minibatch in range(int(len(X_pre)/(inner_batch_size*dataset.totalLabels))):
                # CUrrently only mb = 1
                X = X_pre[(inner_batch_size*dataset.totalLabels)*minibatch:(inner_batch_size*dataset.totalLabels)*minibatch + (inner_batch_size*dataset.totalLabels)]
                X_perm,order = sampleTask(X,len(X[0]),len(X[0]),len(dataset.train_f),32)
                loss,_,out = sess.run([permModel.loss,perm_train_op,permModel.out],feed_dict={permModel.task: X_perm,permModel.label: order})#, permModel.train_mode:True})
                loss_per_epoch.append(loss)

            trainLoss_buffer.append(np.mean(loss_per_epoch))
            train_losses.append(np.mean(loss_per_epoch))
            train_indexes.append(epoch)

            if epoch%50 == 0:
                log_fn(job_id,f"Epoch {epoch}: Permutation loss: {np.mean(trainLoss_buffer):.3f}")
                trainLoss_buffer = []

        if perm_epochs != 0 and not freeze:
            if saving_plots:
                savePlot(train_losses,"Plots",dataset.path.split("/")[-1],"Permutation","Chameleon",xticks=train_indexes,xaxis="Meta Epochs",yaxis="Loss",run=name)
            if saving_raw:
                np.save(os.path.join(raw_path,name_affix)+"/perm_loss.npy",np.array([train_indexes,train_losses]))

    log_fn(job_id,f"Finished pre-training in {time.time()-start:.2f}s")
    log_fn(job_id,"")

    if evalModel is not None:
        log_fn(job_id,"-------------------------------Evaluating Test Data with Scratch Training:---------------")
        reptile.testScratch(evalModel,inner_iters,init_eval_vars,data=test_data,train_f=len(dataset.train_f))
        log_fn(job_id,f"Scratch Evaluation: -- Test Loss {reptile.scratch_loss} -- Test Acc {reptile.scratch_acc}")
        log_fn(job_id,"")
        if saving_plots:
            savePlot([reptile.scratch_loss,reptile.scratch_loss],"Plots",dataset.path.split("/")[-1],"Final_Metatest_ValLoss","Scratch",xticks=[0,meta_iters],xaxis="Meta Epochs",yaxis="Loss",run=name)
            savePlot([reptile.scratch_acc,reptile.scratch_acc],"Plots",dataset.path.split("/")[-1],"Final_Metatest_ValAcc","Scratch",xticks=[0,meta_iters],xaxis="Meta Epochs",yaxis="Accuracy",run=name)
        if saving_raw:
            np.save(os.path.join(raw_path,name_affix)+"/Scratch_Metatest_ValLoss.npy",np.array([reptile.scratch_loss,reptile.scratch_loss]))
            np.save(os.path.join(raw_path,name_affix)+"/Scratch_Metatest_ValAcc.npy",np.array([reptile.scratch_acc,reptile.scratch_acc]))

    if permModel is not None:
        log_fn(job_id,"-------------------------------Training Chameleon and Base Model with reptile:-----------")
    else:
        log_fn(job_id,"--------------------------------------Training Base Model with reptile:------------------")

    # Evaluate the initialized model
    val_final = []
    val_index = []
    train_final = []    
    train_index = []

    train_buffer = []
    full_start = time.time()
    start = time.time()
    if oracle:
        # make sure oracle is padded to testfeats
        t_f = dataset.totalFeatures
    else:
        t_f = len(dataset.train_f)

    reptile.evaluate(repModel = repModel, permModel = permModel, inner_iters=inner_iters,data=test_data, train_f=t_f, exp_name=dataset.path.split("/")[-1],meta_epoch=0,name=name)                   
    log_fn(job_id,f"Val Epoch {0}: Initial Train Loss: {reptile.eval_train_losses[0]:.2f} -- Final Train Loss: {reptile.eval_train_losses[-1]:.2f} -- Val Loss: {reptile.eval_test_loss:.2f} -- Val Acc: {reptile.eval_test_acc:.2f} in {time.time()-start:.2f}s")
    log_fn(job_id,"")   
    val_final.append([reptile.eval_train_losses[0],reptile.eval_train_losses[-1],reptile.eval_test_loss,reptile.eval_test_acc])     

    # Perform reptile joint training on the model
    for meta_epoch in range(1,meta_iters+1):
        start = time.time()
        # Perform one train step
        reptile.train_step(repModel = repModel, permModel = permModel, inner_iters=inner_iters, meta_step_size= meta_step_size, data = train_gen, train_f=t_f, exp_name=dataset.path.split("/")[-1],meta_epoch=meta_epoch,name=name)
        train_final.append([reptile.run_train_losses[0],reptile.run_train_losses[-1],reptile.run_test_loss_before,reptile.run_test_loss_after])
        train_index.append(meta_epoch)
        train_buffer.append([reptile.run_train_losses[0],reptile.run_train_losses[-1],reptile.run_test_loss_before,reptile.run_test_loss_after])

        # log_fn Train Step
        if meta_epoch%100 == 0:
            train_buffer = np.mean(train_buffer,axis=0)
            log_fn(job_id,f"Train Epoch {meta_epoch}: Initial Train Loss: {train_buffer[0]:.2f} -- Final Train Loss: {train_buffer[1]:.2f} -- Initial Val Loss: {train_buffer[2]:.2f} -- Final Val Loss: {train_buffer[3]:.2f}")
            train_buffer = []
        
        # Validates performance on test data
        if meta_epoch%100 == 0:
            reptile.evaluate(repModel = repModel, permModel = permModel, inner_iters=inner_iters,data=test_data, train_f=t_f, exp_name=dataset.path.split("/")[-1],meta_epoch=meta_epoch,name=name)  
            val_final.append([reptile.eval_train_losses[0],reptile.eval_train_losses[-1],reptile.eval_test_loss,reptile.eval_test_acc]) 
            val_index.append(meta_epoch)
            log_fn(job_id,f"Val Epoch {0}: Initial Train Loss: {reptile.eval_train_losses[0]:.2f} -- Final Train Loss: {reptile.eval_train_losses[-1]:.2f} -- Val Loss: {reptile.eval_test_loss:.2f} -- Val Acc: {reptile.eval_test_acc:.2f} in {time.time()-start:.2f}s")
    
    log_fn(job_id,f"Finished joint training in {time.time()-full_start}s")
    log_fn(job_id,"")

    if permModel is not None:
        permName = "Chameleon+Reptile"
    else:
        permName = "Reptile"

    if freeze:
        permName+="_Frozen"
    if perm_epochs==0:
        permName+="_Untrained"

    log_fn(job_id,"Final Shape",np.array(train_final).shape)
    log_fn(job_id,"Final Shape",np.array(val_final).shape)
    log_fn(job_id,val_index)

    if saving_plots:
        savePlot(np.array(train_final)[:,0],"Plots",dataset.path.split("/")[-1],"Initial_Metatrain_Loss",permName,xticks=train_index,xaxis="Meta Epochs",yaxis="Loss",run=name)
        savePlot(np.array(train_final)[:,1],"Plots",dataset.path.split("/")[-1],"Final_Metatrain_Loss",permName,xticks=train_index,xaxis="Meta Epochs",yaxis="Loss",run=name)
        savePlot(np.array(train_final)[:,2],"Plots",dataset.path.split("/")[-1],"Initial_Metatrain_ValLoss",permName,xticks=train_index,xaxis="Meta Epochs",yaxis="Loss",run=name)
        savePlot(np.array(train_final)[:,3],"Plots",dataset.path.split("/")[-1],"Final_Metatrain_ValLoss",permName,xticks=train_index,xaxis="Meta Epochs",yaxis="Loss",run=name)

    
        savePlot(np.array(val_final)[:,0],"Plots",dataset.path.split("/")[-1],"Initial_Metatest_Loss",permName,xticks=val_index,xaxis="Meta Epochs",yaxis="Loss",run=name)
        savePlot(np.array(val_final)[:,1],"Plots",dataset.path.split("/")[-1],"Final_Metatest_Loss",permName,xticks=val_index,xaxis="Meta Epochs",yaxis="Loss",run=name)
        savePlot(np.array(val_final)[:,2],"Plots",dataset.path.split("/")[-1],"Final_Metatest_ValLoss",permName,xticks=val_index,xaxis="Meta Epochs",yaxis="Loss",run=name)
        savePlot(np.array(val_final)[:,3],"Plots",dataset.path.split("/")[-1],"Final_Metatest_ValAcc",permName,xticks=val_index,xaxis="Meta Epochs",yaxis="Accuracy",run=name)

    if saving_raw:
        np.save(os.path.join(raw_path,name_affix)+"/TrainIndexes.npy",train_index)
        np.save(os.path.join(raw_path,name_affix)+"/Initial_Metatrain_Loss.npy",np.array(train_final)[:,0])
        np.save(os.path.join(raw_path,name_affix)+"/Final_Metatrain_Loss.npy",np.array(train_final)[:,1])
        np.save(os.path.join(raw_path,name_affix)+"/Initial_Metatrain_ValLoss.npy",np.array(train_final)[:,2],)
        np.save(os.path.join(raw_path,name_affix)+"/Final_Metatrain_ValLoss.npy",np.array(train_final)[:,3])

        np.save(os.path.join(raw_path,name_affix)+"/ValIndexes.npy",val_index)
        np.save(os.path.join(raw_path,name_affix)+"/Initial_Metatest_Loss.npy",np.array(val_final)[:,0])
        np.save(os.path.join(raw_path,name_affix)+"/Final_Metatest_Loss.npy",np.array(val_final)[:,1])
        np.save(os.path.join(raw_path,name_affix)+"/Final_Metatest_ValLoss.npy",np.array(val_final)[:,2])
        np.save(os.path.join(raw_path,name_affix)+"/Final_Metatest_ValAcc.npy",np.array(val_final)[:,3])

"""
    # Final Validation
    reptile.evaluate(repModel = repModel, permModel = permModel, inner_iters=inner_iters,data=test_data, train_f=len(dataset.train_f),exp_name=dataset.path.split("/")[-1],meta_epoch=meta_epoch,name=name)   
    log_fn(job_id,f"Val Finished: Initial Train Loss: {reptile.eval_train_losses[0]:.2f} -- Final Train Loss: {reptile.eval_train_losses[-1]:.2f} -- Val Loss: {reptile.eval_test_loss:.2f} -- Val Acc: {reptile.eval_test_acc:.2f} in {time.time()-start:.2f}s")
    val_final.append([reptile.eval_train_losses[0],reptile.eval_train_losses[-1],reptile.eval_test_loss,reptile.eval_test_acc]) 
    """
"""    if False:
        save_path = f"{save_dir}/{name}"
        log_fn(job_id,f"Model saved to {save_path}")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        saver.save(sess, os.path.join(save_path, f'{name}.ckpt'))"""