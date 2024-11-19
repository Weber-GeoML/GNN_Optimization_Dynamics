import torch
import os
import copy
import numpy as np
# from measure_smoothing import dirichlet_normalized
from attrdict import AttrDict
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import inf

import wandb

import random
from torch.utils.data import Dataset, Subset

from models.graph_regression_model import GNN, GINE, GPS, SANTransformer, Graphormer

default_args = AttrDict(
    {"learning_rate": 1e-3,
    "max_epochs": 1000,
    "display": True,
    "device": None,
    "eval_every": 1,
    "stopping_criterion": "validation",
    "stopping_threshold": 1.01,
    "patience": 150,
    "train_fraction": 0.83,
    "validation_fraction": 0.083,
    "test_fraction": 1-(0.083+0.83),
    "dropout": 0.5,
    "weight_decay": 1e-5,
    "input_dim": 10,
    "hidden_dim": 64,
    "output_dim": None,
    "hidden_layers": None,
    "num_layers": 1,
    "batch_size": 128,
    "layer_type": "R-GCN",
    "num_relations": 2,
    "last_layer_fa": False
    }
    )

class Experiment:
    def __init__(self, args=None, dataset=None, train_dataset=None, validation_dataset=None, test_dataset=None):
        wandb.init(project="gnn_optimization_dynamics", config=args)
        self.args = default_args + args
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.categories = None

        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers
        if self.args.input_dim is None:
            self.args.input_dim = self.dataset[0].x.shape[1]
        for graph in self.dataset:
            if not "edge_type" in graph.keys():
                num_edges = graph.edge_index.shape[1]
                graph.edge_type = torch.zeros(num_edges, dtype=int)
        if self.args.num_relations is None:
            if self.args.rewiring == "None":
                self.args.num_relations = 1
            else:
                self.args.num_relations = 2

        if self.args.layer_type == "GINE":
            self.model = GINE(self.args).to(self.args.device)
        elif self.args.layer_type == "GPS":
            self.model = GPS(self.args).to(self.args.device)
        elif self.args.layer_type == "SANTransformer":
            self.model = SANTransformer(self.args).to(self.args.device)
        elif self.args.layer_type == "Graphormer":
            self.model = Graphormer(self.args).to(self.args.device)
        else:
            self.model = GNN(self.args).to(self.args.device)
       
        if self.test_dataset is None:
            dataset_size = len(self.dataset)
            train_size = int(self.args.train_fraction * dataset_size)
            validation_size = int(self.args.validation_fraction * dataset_size)
            test_size = dataset_size - train_size - validation_size
            # self.train_dataset, self.validation_dataset, self.test_dataset = random_split(self.dataset,[train_size, validation_size, test_size])
            # self.train_dataset, self.validation_dataset, self.test_dataset, self.categories = custom_random_split(self.dataset, [self.args.train_fraction, self.args.validation_fraction, self.args.test_fraction])
            # set the first 10873 graphs as the training set, the next 2331 as the validation set, and the last 2331 as the test set
            if dataset_size > 12000:
                print("self.test_dataset is None. Custom split for Peptides will be used.")
                self.train_dataset = self.dataset[:10873]
                self.validation_dataset = self.dataset[10873:13204]
                self.test_dataset = self.dataset[13204:]
                self.categories = [[*range(10873)], [*range(10873, 13204)], [*range(13204, 15535)]]
            elif dataset_size == 1000:
                print("Algorithmic task. Using 50/25/25 split.")
                self.train_dataset = self.dataset[:500]
                self.validation_dataset = self.dataset[500:750]
                self.test_dataset = self.dataset[750:]
                self.categories = [[*range(500)], [*range(500, 750)], [*range(750, 1000)]]
            else:
                print("self.test_dataset is None. Custom split for Zinc will be used.")
                self.train_dataset = self.dataset[:10000]
                self.validation_dataset = self.dataset[10000:11000]
                self.test_dataset = self.dataset[11000:]
                self.categories = [[*range(10000)], [*range(10000, 11000)], [*range(11000, 12000)]]
                
        elif self.validation_dataset is None:
            print("self.validation_dataset is None. Custom split will not be used.")
            train_size = int(self.args.train_fraction * len(self.train_dataset))
            validation_size = len(self.args.train_data) - train_size
            self.args.train_data, self.args.validation_data = random_split(self.args.train_data, [train_size, validation_size])
        
    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer)

        best_validation_mae = 0.0
        best_train_mae = 0.0
        best_test_mae = 0.0
        train_goal = 0.0
        validation_goal = 0.0
        epochs_no_improve = 0

        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)
        # complete_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
        complete_loader = DataLoader(self.dataset, batch_size=1)

        for epoch in range(1, 1 + self.args.max_epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for graph in train_loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)

                if not graph.edge_attr:
                    graph.edge_attr = torch.ones(graph.edge_index.shape[1], self.args.hidden_dim)

                if self.args.layer_type in ["SANTransformer", "Graphormer"]:
                    print(graph)
                    out = self.model(graph)
                else:
                    out = self.model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                # loss = self.loss_fn(input=out, target=y)
                loss = (out.squeeze() - y).abs().mean()
                total_loss += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            new_best_str = ''
            scheduler.step(total_loss)
            if epoch % self.args.eval_every == 0:
                train_mae = self.eval(loader=train_loader)
                validation_mae = self.eval(loader=validation_loader)
                test_mae = self.eval(loader=test_loader)
                
                # Log metrics to W&B
                wandb.log({
                    "epoch": epoch,
                    "loss": total_loss.item(),
                    "train_mae": train_mae,
                    "validation_mae": validation_mae,
                    "test_mae": test_mae
                })

                if self.args.stopping_criterion == "train":
                    if train_mae < train_goal:
                        best_train_mae = train_mae
                        best_validation_mae = validation_mae
                        best_test_mae = test_mae
                        epochs_no_improve = 0
                        train_goal = train_mae * self.args.stopping_threshold
                        new_best_str = ' (new best train)'
                    elif train_mae < best_train_mae:
                        best_train_mae = train_mae
                        best_validation_mae = validation_mae
                        best_test_mae = test_mae
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                elif self.args.stopping_criterion == 'validation':
                    if validation_mae < validation_goal:
                        best_train_mae = train_mae
                        best_validation_mae = validation_mae
                        best_test_mae = test_mae
                        epochs_no_improve = 0
                        validation_goal = validation_mae * self.args.stopping_threshold
                        new_best_str = ' (new best validation)'
                    elif validation_mae < best_validation_mae:
                        best_train_mae = train_mae
                        best_validation_mae = validation_mae
                        best_test_mae = test_mae
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                if self.args.display:
                    print(f'Epoch {epoch}, Train mae: {train_mae}, Validation mae: {validation_mae}{new_best_str}, Test mae: {test_mae}')
                if epochs_no_improve > self.args.patience:
                    if self.args.display:
                        print(f'{self.args.patience} epochs without improvement, stopping training')
                        print(f'Best train mae: {train_mae}, Best validation mae: {validation_mae}, Best test mae: {test_mae}')
                        energy = 0

                    wandb.finish()
                    return train_mae, validation_mae, test_mae
                
        if self.args.display:
            print('Reached max epoch count, stopping training')
            print(f'Best train mae: {best_train_mae}, Best validation mae: {best_validation_mae}, Best test mae: {best_test_mae}')

        wandb.finish()
        return best_train_mae, best_validation_mae, best_test_mae

    def eval(self, loader):
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_mae = 0
            for graph in loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)

                if not graph.edge_attr:
                    graph.edge_attr = torch.ones(graph.edge_index.shape[1], self.args.hidden_dim)

                if self.args.layer_type in ["SANTransformer", "Graphormer"]:
                    out = self.model(graph)
                else:
                    out = self.model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                # _, pred = out.max(dim=1)
                error = (out.squeeze() - y).abs().mean()
                total_mae += error.item() * graph.num_graphs
                # total_mae += (out.squeeze() - y).abs().sum().item()
                
        return total_mae / sample_size
    
    def check_dirichlet(self, loader):
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_energy = 0
            for graph in loader:
                graph = graph.to(self.args.device)
                total_energy += self.model(graph, measure_dirichlet=True)
        return total_energy / sample_size


def custom_random_split(dataset, percentages):
    percentages = [100 * percentage for percentage in percentages]
    if sum(percentages) != 100:
        raise ValueError("Percentages must sum to 100")
    
    # Calculate the lengths of the three categories
    total_length = len(dataset)
    lengths = [int(total_length * p / 100) for p in percentages]
    
    # Shuffle the input list
    shuffled_list = [*range(total_length)]
    random.shuffle(shuffled_list)
    
    # Split the shuffled list into three categories
    categories = [shuffled_list[:lengths[0]],
                  shuffled_list[lengths[0]:lengths[0]+lengths[1]],
                  shuffled_list[lengths[0]+lengths[1]:]]
    
    train_dataset = Subset(dataset, categories[0])
    validation_dataset = Subset(dataset, categories[1])
    test_dataset = Subset(dataset, categories[2])
    
    return train_dataset, validation_dataset, test_dataset, categories
