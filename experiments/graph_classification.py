import torch
import os
import copy
import numpy as np
# from measure_smoothing import dirichlet_normalized
from attrdict import AttrDict
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torcheval.metrics import MultilabelAUPRC
from math import inf

import wandb

import random
from torch.utils.data import Dataset, Subset

from models.graph_model import GNN, GPS

default_args = AttrDict(
    {"learning_rate": 1e-3,
    "max_epochs": 1000000,
    "display": True,
    "device": None,
    "eval_every": 1,
    "stopping_criterion": "validation",
    "stopping_threshold": 1.01,
    "patience": 50,
    "train_fraction": 0.5,
    "validation_fraction": 0.25,
    "test_fraction": 0.25,
    "dropout": 0.0,
    "weight_decay": 1e-5,
    "input_dim": None,
    "hidden_dim": 64,
    "output_dim": 1,
    "hidden_layers": None,
    "num_layers": 2,
    "batch_size": 10,
    "layer_type": "R-GCN",
    "num_relations": 2,
    "last_layer_fa": False
    }
    )

class Experiment:
    def __init__(self, args=None, dataset=None, train_dataset=None, validation_dataset=None, test_dataset=None):
        # Initialize W&B with project name and experiment configuration
        wandb.init(project="gnn_optimization_dynamics", config=args)
        self.args = default_args + args
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.categories = None

        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers
        if self.args.input_dim is None:
            try:
                self.args.input_dim = self.dataset[0].x.shape[1]
            except:
                self.args.input_dim = 9  # peptides-func
        for graph in self.dataset:
            if not "edge_type" in graph.keys:
                num_edges = graph.edge_index.shape[1]
                graph.edge_type = torch.zeros(num_edges, dtype=int)
        if self.args.num_relations is None:
            if self.args.rewiring == "None":
                self.args.num_relations = 1
            else:
                self.args.num_relations = 2

        if self.args.layer_type == "GPS":
            self.model = GPS(self.args).to(self.args.device)
        else:
            self.model = GNN(self.args).to(self.args.device)
       
        if self.test_dataset is None:
            dataset_size = len(self.dataset)
            train_size = int(self.args.train_fraction * dataset_size)
            validation_size = int(self.args.validation_fraction * dataset_size)
            test_size = dataset_size - train_size - validation_size
            self.train_dataset, self.validation_dataset, self.test_dataset, self.categories = custom_random_split(self.dataset, [self.args.train_fraction, self.args.validation_fraction, self.args.test_fraction])
        elif self.validation_dataset is None:
            train_size = int(self.args.train_fraction * len(self.train_dataset))
            validation_size = len(self.args.train_data) - train_size
            self.args.train_data, self.args.validation_data = random_split(self.args.train_data, [train_size, validation_size])
        
    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer)

        best_validation_acc = 0.0
        best_train_acc = 0.0
        best_test_acc = 0.0
        train_goal = 0.0
        validation_goal = 0.0
        epochs_no_improve = 0
        best_model = copy.deepcopy(self.model)

        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)
        complete_loader = DataLoader(self.dataset, batch_size=1)

        for epoch in range(1, 1 + self.args.max_epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for graph in train_loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)

                out = self.model(graph)
                loss = self.loss_fn(input=out, target=y)
                total_loss += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step(total_loss)

            # Evaluate and log metrics with W&B after each epoch
            if epoch % self.args.eval_every == 0:
                if len(self.dataset) < 10000:
                    train_acc = self.eval(loader=train_loader)
                    validation_acc = self.eval(loader=validation_loader)
                    test_acc = self.eval(loader=test_loader)
                    
                else:
                    train_acc = self.test(loader=train_loader)
                    validation_acc = self.test(loader=validation_loader)
                    test_acc = self.test(loader=test_loader)

                # Log metrics to W&B
                wandb.log({
                    "epoch": epoch,
                    "loss": total_loss.item(),
                    "train_accuracy": train_acc,
                    "validation_accuracy": validation_acc,
                    "test_accuracy": test_acc
                })

                if self.args.display:
                    print(f'Epoch {epoch}, Loss: {total_loss.item()}, Train acc: {train_acc}, Validation acc: {validation_acc}, Test acc: {test_acc}')

                # Early stopping based on validation accuracy
                if validation_acc > validation_goal:
                    best_model = copy.deepcopy(self.model)
                    validation_goal = validation_acc * self.args.stopping_threshold
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve > self.args.patience:
                    print(f'{self.args.patience} epochs without improvement, stopping training')
                    break

        wandb.finish()
        return best_train_acc, best_validation_acc, best_test_acc

    def eval(self, loader):
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_correct = 0
            for graph in loader:
                graph = graph.to(self.args.device)
                out = self.model(graph)
                y = graph.y.to(self.args.device)
                # check if y contains more than one element
                if y.dim() > 1:
                    loss = self.loss_fn(input=out, target=y)
                    total_correct -= loss
                else:
                    _, pred = out.max(dim=1)
                    total_correct += pred.eq(y).sum().item()
                
        return total_correct / sample_size
    
    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        metric = MultilabelAUPRC(num_labels=10)

        total_error = 0
        for data in loader:
            error = 0
            data = data.to(self.args.device)
            out = self.model(data)
            # error_fnc = torch.nn.CrossEntropyLoss()
            # error += error_fnc(out, data.y)
            metric.update(out, data.y)
            error += metric.compute()
            total_error += error.item() * data.num_graphs
        return total_error / len(loader.dataset)
    
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