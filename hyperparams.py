import argparse
import ast
import yaml
from attrdict import AttrDict

def get_args_from_input():
	parser = argparse.ArgumentParser(description='modify network parameters', argument_default=argparse.SUPPRESS)
	parser.add_argument('--global_pool', type=str, help='type of global pooling layer to use')
	parser.add_argument('--config_path', metavar='', type=str, help='Path to config file')
	parser.add_argument('--learning_rate', metavar='', type=float, help='learning rate')
	parser.add_argument('--max_epochs', metavar='', type=int, help='maximum number of epochs for training')
	parser.add_argument('--layer_type', metavar='', help='type of layer in GNN (GCN, GIN, GAT, etc.)')
	parser.add_argument('--display', metavar='', type=bool, help='toggle display messages showing training progress')
	parser.add_argument('--device', metavar='', type=str, help='name of CUDA device to use or CPU')
	parser.add_argument('--eval_every', metavar='X', type=int, help='calculate validation/test accuracy every X epochs')
	parser.add_argument('--stopping_criterion', metavar='', type=str, help='model stops training when this criterion stops improving (can be train, validation, or test)')
	parser.add_argument('--stopping_threshold', metavar='T', type=float, help="model perceives no improvement when it does worse than (best loss) * T")
	parser.add_argument('--patience', metavar='P', type=int, help='model stops training after P epochs with no improvement')
	parser.add_argument('--train_fraction', metavar='', type=float, help='fraction of the dataset to be used for training')
	parser.add_argument('--validation_fraction', metavar='', type=float, help='fraction of the dataset to be used for validation')
	parser.add_argument('--test_fraction', metavar='', type=float, help='fraction of the dataset to be used for testing')
	parser.add_argument('--dropout', metavar='', type=float, help='layer dropout probability')
	parser.add_argument('--weight_decay', metavar='', type=float, help='weight decay added to loss function')
	parser.add_argument('--hidden_dim', metavar='', type=int, help='width of hidden layer')
	parser.add_argument('--hidden_layers', metavar='', type=ast.literal_eval, help='list containing dimensions of all hidden layers')
	parser.add_argument('--num_layers', metavar='', type=int, help='number of hidden layers')
	parser.add_argument('--num_splits', metavar='', type=int, default=3, help='Number of random splits')
	parser.add_argument('--batch_size', metavar='', type=int, help='number of samples in each training batch')
	parser.add_argument('--num_trials', metavar='', type=int, help='number of times the network is trained'),
	parser.add_argument('--rewiring', metavar='', type=str, help='type of rewiring to be performed'),
	parser.add_argument('--num_iterations', metavar='', type=int, help='number of iterations of rewiring')
	parser.add_argument('--alpha', type=float, help='alpha hyperparameter for DIGL')
	parser.add_argument('--k', type=int, help='k hyperparameter for DIGL')
	parser.add_argument('--eps', type=float, help='epsilon hyperparameter for DIGL')
	parser.add_argument('--datasets', nargs='*', type=str, help='names of datasets to use')
	parser.add_argument('--train_datasets', nargs='*', type=str, help='names of training datasets to use')
	parser.add_argument('--test_datasets', nargs='*', type=str, help='names of testing datasets to use')
	parser.add_argument('--last_layer_fa', type=str, help='whether or not to make last layer fully adjacent')
	parser.add_argument('--borf_batch_add', type=int)
	parser.add_argument('--borf_batch_remove', type=int)
	parser.add_argument('--encoding', type=str, help='type of encoding to use for node features') 
	arg_values = parser.parse_args()
	return AttrDict(vars(arg_values))

def convert_to_type(value, expected_type):
    if expected_type == int:
        return int(value)
    elif expected_type == float:
        return float(value)
    elif expected_type == bool:
        return value.lower() in ['true', '1', 't', 'y', 'yes']
    elif expected_type == list:
        return value.split(',')
    return value 

type_mapping = {
    'some_integer_field': int,
    'some_float_field': float,
    'some_bool_field': bool,
    'some_list_field': list
}

def get_args_from_config(config, type_mapping=None):
	with open(config, 'r') as f:
		config_data = yaml.safe_load(f)

#	if type_mapping:
#		config_data = convert_config_types(config_data, type_mapping)

	return AttrDict(config_data)