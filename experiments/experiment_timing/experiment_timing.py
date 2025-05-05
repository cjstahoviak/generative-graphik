import importlib.util
import json
import os
import sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
import random
from argparse import Namespace
import time
import pickle as pkl
import numpy as np
import argparse

import torch
import torch_geometric
from generative_graphik.utils.dataset_generation import (
    generate_data_point,
    random_revolute_robot_graph,
)


def model_arg_loader(path):
    """Load hyperparameters from trained model."""
    if os.path.isdir(path):
        with open(os.path.join(path, "hyperparameters.txt"), "r") as fp:
            return Namespace(**json.load(fp))

# NOTE generates all the initializations and stores them to a pickle file
def main(args):
    device = args.device
    print(f"Starting timing experiment using device: {device}")

    for model_path in args.model_path:
        print(f"Loading model from {model_path}")
        spec = importlib.util.spec_from_file_location("model", model_path + "model.py")
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)

        # load models
        model_args = model_arg_loader(model_path)
        model = model.Model(model_args).to(device)
        if model_path is not None:
            try:
                model.load_state_dict(
                    torch.load(model_path + f"/net.pth", map_location=device)
                )
                model.eval()
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # Remove the torch_geometric.compile call that's causing the error
        # If you need to use torch.compile instead (for PyTorch 2.0+), uncomment below:
        # if hasattr(torch, 'compile'):
        #     print("Using torch.compile")
        #     model = torch.compile(model, mode="max-autotune", fullgraph=True)

        sample_amounts = [1, 16, 64, 128, 256]
        joint_amounts = ["6", "12", "18", "24", "30", "36"]
        results = []

        print("\nStarting timing experiments:")
        print(f"Sample amounts: {sample_amounts}")
        print(f"Joint amounts: {joint_amounts}")

        for sample_amount in sample_amounts:
            for joint_amount in joint_amounts:
                print(f"\nTesting with {sample_amount} samples for {joint_amount}-DOF robot")
                
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                
                #Warm up forward pass
                print("  Warming up...", end="", flush=True)
                for _ in range(2):
                    num_joints = int(joint_amount)
                    graph = random_revolute_robot_graph(num_joints)
                
                    # Generate random problem
                    prob_data = generate_data_point(graph).to(device)
                    prob_data.num_graphs = 1
                    prob_data.T_ee = prob_data.T_ee.unsqueeze(0)
                    data = model.preprocess(prob_data)

                    # Compute solutions
                    _ = model.forward_eval(
                        x=data.pos,
                        h=torch.cat((data.type, data.goal_data_repeated_per_node), dim=-1), 
                        edge_attr=data.edge_attr,
                        edge_attr_partial=data.edge_attr_partial,
                        edge_index=data.edge_index_full,
                        partial_goal_mask=data.partial_goal_mask,
                        nodes_per_single_graph = int(data.num_nodes / 1),
                        num_samples=sample_amount,
                        batch_size=1
                    )
                print(" done")

                print(f"  Running {16} timing trials...", end="", flush=True)
                trial_times = []
                for trial in range(16):
                    num_joints = int(joint_amount)
                    graph = random_revolute_robot_graph(num_joints)
                
                    # Generate random problem
                    prob_data = generate_data_point(graph).to(device)
                    prob_data.num_graphs = 1
                    prob_data.T_ee = prob_data.T_ee.unsqueeze(0)
                    data = model.preprocess(prob_data)

                    # Compute solutions
                    starter.record()
                    _ = model.forward_eval(
                        x=data.pos,
                        h=torch.cat((data.type, data.goal_data_repeated_per_node), dim=-1), 
                        edge_attr=data.edge_attr,
                        edge_attr_partial=data.edge_attr_partial,
                        edge_index=data.edge_index_full,
                        partial_goal_mask=data.partial_goal_mask,
                        nodes_per_single_graph = int(data.num_nodes / 1),
                        num_samples=sample_amount,
                        batch_size=1
                    )
                    ender.record()
                    torch.cuda.synchronize()
                    t_sol = starter.elapsed_time(ender)
                    trial_times.append(t_sol)
                    results.append((sample_amount, t_sol, joint_amount))
                
                avg_time = sum(trial_times) / len(trial_times)
                print(f" completed (avg: {avg_time:.2f} ms)")

            print(f"\nSaving results to {args.id}/results.pkl")
            exp_dir = f"{sys.path[0]}/results/"+ f"{args.id}/"
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, 'results.pkl'), 'wb') as handle:
                pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)
            
        print("\nTiming experiment completed successfully!")

if __name__ == "__main__":
    random.seed(15)
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    parser.add_argument("--model_path", nargs="*", type=str, required=True, help="Path to folder with model data")
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for PyTorch')

    args = parser.parse_args()
    main(args)