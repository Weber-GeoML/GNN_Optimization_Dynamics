from attrdict import AttrDict
from torch_geometric.datasets import ZINC
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
import torch_geometric.transforms as T

# import custom encodings
from torchvision.transforms import Compose
from custom_encodings import LocalCurvatureProfile, AltLocalCurvatureProfile

from experiments.algorithmic_tasks import Experiment

from itertools import chain

import time
import tqdm
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input, get_args_from_config
from preprocessing import rewiring, sdrf, fosr, digl, borf

import pickle
import wget
import zipfile
import os

max_degree_path = 'synthetic_data/max_degree_task'
with open(os.path.join(max_degree_path, 'complete_graphs.pkl'), 'rb') as file:
    max_complete = pickle.load(file)
with open(os.path.join(max_degree_path, 'cycle_graphs.pkl'), 'rb') as file:
    max_cycle = pickle.load(file)
with open(os.path.join(max_degree_path, 'path_graphs.pkl'), 'rb') as file:
    max_path = pickle.load(file)
with open(os.path.join(max_degree_path, 'regular_graphs.pkl'), 'rb') as file:
    max_regular = pickle.load(file)
with open(os.path.join(max_degree_path, 'tree_graphs.pkl'), 'rb') as file:
    max_tree = pickle.load(file)

min_distance_path = 'synthetic_data/shortest_path_task'
with open(os.path.join(min_distance_path, 'complete_graphs.pkl'), 'rb') as file:
    min_complete = pickle.load(file)
with open(os.path.join(min_distance_path, 'cycle_graphs.pkl'), 'rb') as file:
    min_cycle = pickle.load(file)
with open(os.path.join(min_distance_path, 'path_graphs.pkl'), 'rb') as file:
    min_path = pickle.load(file)
with open(os.path.join(min_distance_path, 'regular_graphs.pkl'), 'rb') as file:
    min_regular = pickle.load(file)
with open(os.path.join(min_distance_path, 'tree_graphs.pkl'), 'rb') as file:
    min_tree = pickle.load(file)

all_datasets = {"max_complete": max_complete, "max_cycle": max_cycle, "max_path": max_path, "max_regular": max_regular, "max_tree": max_tree, "min_complete": min_complete, "min_cycle": min_cycle, "min_path": min_path, "min_regular": min_regular, "min_tree": min_tree}

def log_to_file(message, filename="results/algorithmic_tasks.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

def feature_to_x(data):
    if hasattr(data, 'features'):
        data.x = data.features
        del data.features
    return data

default_args = AttrDict({
    "config_path": None,
    "dropout": 0.1,
    "num_layers": 16,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "layer_type": "GINE",
    "display": True,
    "num_trials": 15,
    "eval_every": 1,
    "rewiring": None,
    "num_iterations": 1,
    "patience": 250,
    "output_dim": 2,
    "alpha": 0.1,
    "eps": 0.001,
    "dataset": "max_complete",
    "train_dataset": None,
    "test_dataset": None,
    "last_layer_fa": False,
    "borf_batch_add" : 20,
    "borf_batch_remove" : 3,
    "sdrf_remove_edges" : False,
    "encoding" : None
})

hyperparams = {}

for key in all_datasets:
    hyperparams[key] = AttrDict({"output_dim": 1})

results = []
args = default_args
args += get_args_from_input()

if args.config_path:
    args += get_args_from_config(config=args.config_path)


# check if dataset starts with 'max' or 'min' and only keep the corresponding datasets
if args.dataset.startswith('max'):
    all_datasets = {k: v for k, v in all_datasets.items() if k.startswith('max')}
elif args.dataset.startswith('min'):
    all_datasets = {k: v for k, v in all_datasets.items() if k.startswith('min')}


for name, dataset in all_datasets.items():

    if args.encoding in ["LAPE", "RWPE", "LCP", "LDP", "SUB", "EGO", "VN"]:

        if os.path.exists(f"data/{name}_{args.encoding}.pt"):
            print('ENCODING ALREADY COMPLETED...')
            dataset = torch.load(f"data/{name}_{args.encoding}.pt")

        elif args.encoding == "LCP":
            print('ENCODING STARTED...')
            lcp = LocalCurvatureProfile()
            for k in range(len(dataset)):
                dataset[k] = lcp.compute_orc(dataset[k])
                print(f"Graph {k} of {len(dataset)} encoded with {args.encoding}")
            torch.save(dataset, f"data/{name}_{args.encoding}.pt")

        else:
            print('ENCODING STARTED...')
            org_dataset_len = len(dataset)
            drop_datasets = []
            current_graph = 0

            for k in range(org_dataset_len):
                if args.encoding == "LAPE":
                    num_nodes = dataset[k].num_nodes
                    eigvecs = np.min([num_nodes, 8]) - 2
                    transform = T.AddLaplacianEigenvectorPE(k=eigvecs, attr_name=None)

                elif args.encoding == "RWPE":
                    transform = T.AddRandomWalkPE(walk_length=16, attr_name=None)

                elif args.encoding == "LDP":
                    transform = T.LocalDegreeProfile()

                elif args.encoding == "SUB":
                    transform = T.RootedRWSubgraph(walk_length=10)

                elif args.encoding == "EGO":
                    transform = T.RootedEgoNets(num_hops=2)

                elif args.encoding == "VN":
                    transform = T.VirtualNode()

                try:
                    dataset[k] = transform(dataset[k])
                    print(f"Graph {current_graph} of {org_dataset_len} encoded with {args.encoding}")
                    current_graph += 1

                except:
                    print(f"Graph {current_graph} of {org_dataset_len} dropped due to encoding error")
                    drop_datasets.append(k)
                    current_graph += 1

            for k in sorted(drop_datasets, reverse=True):
                dataset.pop(k)

            # save the dataset to a file in the data folder
            torch.save(dataset, f"data/{name}_{args.encoding}.pt")

        print('ENCODED ', name)
        all_datasets[name] = dataset

train_accuracies = []
validation_accuracies = []
test_accuracies = []

print('TRAINING STARTED...')
start = time.time()

for trial in range(args.num_trials):
    print(f"Trial {trial + 1} of {args.num_trials}")
    train_acc, validation_acc, test_acc = Experiment(args=args, dataset=args.dataset, all_datasets=all_datasets).run()
    train_accuracies.append(train_acc)
    validation_accuracies.append(validation_acc)
    test_accuracies.append(test_acc)
end = time.time()
run_duration = end - start


train_mean = np.mean(train_accuracies)
val_mean = np.mean(validation_accuracies)
test_mean = np.mean(test_accuracies)
train_ci = 2 * np.std(train_accuracies)/(args.num_trials ** 0.5)
val_ci = 2 * np.std(validation_accuracies)/(args.num_trials ** 0.5)
test_ci = 2 * np.std(test_accuracies)/(args.num_trials ** 0.5)
results.append({
    "dataset": name,
    "rewiring": args.rewiring,
    "layer_type": args.layer_type,
    "num_iterations": args.num_iterations,
    "borf_batch_add" : args.borf_batch_add,
    "borf_batch_remove" : args.borf_batch_remove,
    "sdrf_remove_edges" : args.sdrf_remove_edges, 
    "alpha": args.alpha,
    "eps": args.eps,
    "test_mean": test_mean,
    "test_ci": test_ci,
    "val_mean": val_mean,
    "val_ci": val_ci,
    "train_mean": train_mean,
    "train_ci": train_ci,
    "last_layer_fa": args.last_layer_fa,
    "run_duration" : run_duration,
})

# Log every time a dataset is completed
df = pd.DataFrame(results)
with open(f'results/algorithmic_tasks{args.layer_type}_{args.rewiring}.csv', 'a') as f:
    df.to_csv(f, mode='a', header=f.tell()==0)