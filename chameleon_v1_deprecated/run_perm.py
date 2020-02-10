import random
import tensorflow as tf

from code.args   import argument_parser, train_kwargs
from code.models import BaseModel, Chameleon
from code.train import train

"""
    Main.
    Initialize the models and run the Chameleon Experiment
"""

def main():
    
    args = argument_parser().parse_args()
    random.seed(args.seed)

    # Select data set
    if args.dataset   == "Heart":
    	feats = 13
    elif args.dataset == "Wine":
    	feats = 12
    elif args.dataset == "Abalone":
    	feats = 8
    elif args.dataset == "Telescope":
    	feats = 10
    else:
    	raise ValueError("Dataset argument must be one of: Wine, Abalone, Telescope, Heart")

    
    # Initialize Chameleon component
    if args.permuter:
        print("Initializing Chameleon")
        mEnc        = Chameleon(input_shape=[args.inner_batch, feats],activation=tf.nn.relu,permMatrix=args.perm_matrix,regC=args.reg_columns,num_filter=args.num_filter,num_conv=args.num_conv,num_neurons=args.num_neurons,conv_act=args.conv_act,name_suf="Perm")
        mEnc.build()
        model_input = mEnc.out
    else:
        mEnc = None
        model_input = None

    # Initialize base model ŷ for reptile and scratch training
    repModel  = BaseModel(size=feats,layers=2,num_classes=2,num_neurons=64,input_ph=model_input,name_suf="RepMod")
    evalModel = BaseModel(size=feats,layers=2,num_classes=2,num_neurons=64,input_ph=None       ,name_suf="EvalMod")

    with tf.Session() as sess:
        
        print('Training...')
        try:
            train(sess, repModel,evalModel,mEnc,args.checkpoint, **train_kwargs(args))
        except TypeError:
            print("-----------------------------------------------------------------")
            raise


if __name__ == '__main__':
    main()
