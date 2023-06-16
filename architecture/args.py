import argparse

# Converter for console-parsed arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train_parser():
    parser = argparse.ArgumentParser()

    # Basic params
    parser.add_argument('--model', type=str, default='gta', help='model of the experiment')

    parser.add_argument('--dataset', type=str, default='swat', help='data')
    parser.add_argument('--keep_time', type=bool, default=False, help='keep time data as pos.emb. for the Informer')
    parser.add_argument('--features', type=str, default='M', help='features [S, M]')   # TODO. Check if unnecessary
    parser.add_argument('--target', type=str, default='OT', help='target feature')    # TODO. Check if unnecessary

    # Params of the input data.
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--train_start', type=int, default=0, help='starting index for training data')
    parser.add_argument('--train_end', type=int, default=None, help='end index for training data')
    parser.add_argument('--seq_len', type=int, default=60, help='input series length')
    parser.add_argument('--label_len', type=int, default=30, help='help series length')
    parser.add_argument('--pred_len', type=int, default=24, help='predict series length')

    # Params of the Hierarchical Dilated Conv.
    parser.add_argument('--num_nodes', type=int, default=7, help='encoder input size')
    parser.add_argument('--num_levels', type=int, default=3, help='number of dilated levels for graph embedding')

    # Details of the Transformer block
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='prob sparse factor')

    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention [prob, full]')
    parser.add_argument('--embed', type=str, default='fixed', help='embedding type [fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu',help='activation')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

    # Details of the training process
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--step_lr", type=int, default=10)
    # parser.add_argument('--gamma', type=float, default=1.0)
    # parser.add_argument('--gamma_lr', type=float, default=0.9)
    parser.add_argument('--itr', type=int, default=2, help='each params run iteration')
    parser.add_argument('--epochs', type=int, default=6, help='train epochs')
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--shuffle_dataset', type=str2bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test',help='exp description')
    parser.add_argument('--loss', type=str, default='mse',help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')

    #GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # For epsilon
    parser.add_argument("--reg_level", default=1, help="Controls the reg_level argument of the epsilon thresholding method. Set to None if you don't want to calculate this threshold during training.")
    parser.add_argument("--use_mov_av", type=str2bool, default=False)

    return parser


def predict_parser():
    parser = argparse.ArgumentParser()

    # --- Model ---
    parser.add_argument('--model', type=str, default='gta',help='model of the experiment')

    # --- Data params ---
    parser.add_argument("--dataset", type=str, default="swat")
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument("--eval_end", type=int, default=None)

    parser.add_argument('--seq_len', type=int, default=60, help='input series length')
    parser.add_argument('--label_len', type=int, default=30, help='help series length')
    parser.add_argument('--pred_len', type=int, default=24, help='predict series length')
    parser.add_argument('--keep_time', type=bool, default=False, help='keep time data as pos.emb. for the Informer')

    # --- Model params ---
    parser.add_argument("--run_name", type=str, default="-1", help="name of run to use, or -1, -2, etc. to find last, previous from last, etc. run.")

    # --- Predict params ---
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--show_details", type=str2bool, default=True)
    parser.add_argument("--threshold", type=str, default="POT")

    # If threshold is set to POT, these are the POT params
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--q", type=float, default=1e-3)
    parser.add_argument("--level", type=float, default=0.99)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)

    return parser

def get_args(train=True):

    parser = train_parser() if train else predict_parser()
    args = parser.parse_args()

    data_parser = {
    'WADI':{'data':'WADI_14days_downsampled.csv','T':'1_LS_001_AL','M':112,'S':1},
    'SMAP':{'data':'SMAP','T':0,'M':25,'S':1},
    'MSL':{'data':'MSL','T':0,'M':55,'S':1},
    'swat':{'data':'SWaT','T':'FIT_101','M':51,'S':1}
    }   

    if args.dataset in data_parser.keys():
        data_info = data_parser[args.dataset]
        # args.data_path = data_info['data']
        args.target = data_info['T']
        args.num_nodes = data_info[args.features]
    
    return args
