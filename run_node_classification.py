from attrdict import AttrDict
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid, HeterophilousGraphDataset
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, dropout_edge
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.node_classification import Experiment

import pickle
import time
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, borf

import torch_geometric.transforms as T
from custom_encodings import LocalCurvatureProfile


default_args = AttrDict({
    "dropout": 0.5,
    "num_layers": 3,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": True,
    "num_trials": 10,
    "eval_every": 1,
    "rewiring": None,
    "num_iterations": 50,
    "num_relations": 2,
    "patience": 100,
    "dataset": None,
    "borf_batch_add" : 4,
    "borf_batch_remove" : 2,
    "sdrf_remove_edges" : False,
    "encoding": None
})


results = []
args = default_args
args += get_args_from_input()

# encode the dataset using the given encoding, if args.encoding is not None
if args.encoding in ["LAPE", "RWPE", "LCP", "LDP", "SUB", "EGO"]:
    if args.encoding == "LAPE":
        eigvecs = 8
        transform = T.AddLaplacianEigenvectorPE(k=eigvecs)
        print(f"Encoding Laplacian Eigenvector PE (k={eigvecs})")

    elif args.encoding == "RWPE":
        transform = T.AddRandomWalkPE(walk_length=16)
        print("Encoding Random Walk PE")

    elif args.encoding == "LDP":
        transform = T.LocalDegreeProfile()
        print("Encoding Local Degree Profile")

    elif args.encoding == "SUB":
        transform = T.RootedRWSubgraph(walk_length=10)
        print("Encoding Rooted RW Subgraph")

    elif args.encoding == "EGO":
        transform = T.RootedEgoNets(num_hops=2)
        print("Encoding Rooted Ego Nets")

    elif args.encoding == "LCP":
        lcp = LocalCurvatureProfile()
        transform = lcp.forward
        print(f"Encoding Local Curvature Profile (ORC)")


platonov_path = '/n/holyscratch01/mweber_lab/platonov_datasets'

if args.encoding in ["LCP", "LAPE", "RWPE", "LDP", "SUB", "EGO"]:
    roman_empire = HeterophilousGraphDataset(root=platonov_path, name="Roman-empire", transform=transform)
    amazon_ratings = HeterophilousGraphDataset(root=platonov_path, name="Amazon-ratings", transform=transform)
    minesweeper = HeterophilousGraphDataset(root=platonov_path, name="Minesweeper", transform=transform)
    tolokers = HeterophilousGraphDataset(root=platonov_path, name="Tolokers", transform=transform)
    questions = HeterophilousGraphDataset(root=platonov_path, name="Questions", transform=transform)

else:
    largest_cc = LargestConnectedComponents()
    roman_empire = HeterophilousGraphDataset(root=platonov_path, name="Roman-empire", transform=largest_cc)
    amazon_ratings = HeterophilousGraphDataset(root=platonov_path, name="Amazon-ratings", transform=largest_cc)
    minesweeper = HeterophilousGraphDataset(root=platonov_path, name="Minesweeper", transform=largest_cc)
    tolokers = HeterophilousGraphDataset(root=platonov_path, name="Tolokers", transform=largest_cc)
    questions = HeterophilousGraphDataset(root=platonov_path, name="Questions", transform=largest_cc)

datasets = {roman_empire: "roman_empire", amazon_ratings: "amazon_ratings", minesweeper: "minesweeper", tolokers: "tolokers", questions: "questions"}

for key in datasets:
    dataset = datasets[key]
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)

def log_to_file(message, filename="results/node_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()


if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}


for key in datasets:
    accuracies = []
    print(f"TESTING: {key} ({args.rewiring})")
    dataset = datasets[key]
    

    start = time.time()
    for trial in range(args.num_trials):
        print(f"TRIAL #{trial+1}")
        test_accs = []
        for i in range(args.num_splits):
            train_acc, validation_acc, test_acc = Experiment(args=args, dataset=dataset).run()
            test_accs.append(test_acc)
        test_acc = max(test_accs)
        accuracies.append(test_acc)
    end = time.time()
    run_duration = end - start

    log_to_file(f"RESULTS FOR {key} ({args.rewiring}):\n")
    log_to_file(f"average acc: {np.mean(accuracies)}\n")
    log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(args.num_trials ** 0.5)}\n\n")
    results.append({
        "dataset": key,
        "rewiring": args.rewiring,
        "num_iterations": args.num_iterations,
        "borf_batch_add" : args.borf_batch_add,
        "borf_batch_remove" : args.borf_batch_remove,
        "avg_accuracy": np.mean(accuracies),
        "ci":  2 * np.std(accuracies)/(args.num_trials ** 0.5),
        "run_duration" : run_duration,
    })
    results_df = pd.DataFrame(results)
    with open(f'results/node_classification_{args.layer_type}_{args.rewiring}.csv', 'a') as f:
        results_df.to_csv(f, mode='a', header=f.tell()==0)