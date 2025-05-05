import importlib.util
import json
import os
import sys
import argparse
import graphik
from graphik.graphs import ProblemGraphRevolute
from graphik.robots import RobotRevolute
from generative_graphik.utils.dataset_generation import generate_data_point

os.environ["PYOPENGL_PLATFORM"] = "egl"
import random
from argparse import Namespace
import pandas as pd
import time
import numpy as np
import torch
from graphik.utils.dgp import graph_from_pos
from liegroups.numpy import SE3

def model_arg_loader(path):
    if os.path.isdir(path):
        with open(os.path.join(path, "hyperparameters.txt"), "r") as fp:
            return Namespace(**json.load(fp))

def main(args):
    print(f"Starting evaluation with {args.n_evals} samples across {len(args.robots)} robot types")
    print(f"Using device: {args.device}")
    
    device = args.device
    num_evals = args.n_evals
    robot_types = args.robots
    evals_per_robot = num_evals // len(robot_types)
    
    print(f"Testing {args.num_samples} configurations per IK problem")
    
    total_models = len(args.model_path)
    for model_idx, model_path in enumerate(args.model_path):
        print(f"\n[{model_idx+1}/{total_models}] Loading model from {model_path}")
        
        spec = importlib.util.spec_from_file_location("model", model_path + "model.py")
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)

        model_args = model_arg_loader(model_path)
        model = model.Model(model_args).to(device)
        name = model_args.id.replace("model", "results")
        c = np.pi / 180

        if model_path is not None:
            try:
                try:
                    state_dict = torch.load(model_path + f"checkpoints/checkpoint.pth", map_location=device)
                    model.load_state_dict(state_dict["net"])
                    print(f"✓ Loaded model from checkpoint.pth")
                except:
                    state_dict = torch.load(model_path + f"net.pth", map_location=device)
                    model.load_state_dict(state_dict)
                    print(f"✓ Loaded model from net.pth")
                model.eval()
            except Exception as e:
                print(f"✗ Error loading model: {e}")

        all_sol_data = []
        for robot_type in robot_types:
            if robot_type == "ur10":
                modified_dh = False
                a = [0, -0.612, 0.5723, 0, 0, 0]
                d = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
                al = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
                th = [0, 0, 0, 0, 0, 0]

                params = {
                    "a": a,
                    "alpha": al,
                    "d": d,
                    "theta": th,
                    "modified_dh": modified_dh,
                    "num_joints": 6,
                }
                robot = RobotRevolute(params)
                graph = ProblemGraphRevolute(robot)
            elif robot_type == "kuka":
                modified_dh = False
                a = [0, 0, 0, 0, 0, 0, 0]
                d = [0.34, 0, 0.40, 0, 0.40, 0, 0.126]
                al = [-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]
                th = [0, 0, 0, 0, 0, 0, 0]

                params = {
                    "a": a,
                    "alpha": al,
                    "d": d,
                    "theta": th,
                    "modified_dh": modified_dh,
                    "num_joints": 7,
                }
                robot = RobotRevolute(params)
                graph = ProblemGraphRevolute(robot)
            elif robot_type == "lwa4d":
                modified_dh = False
                a = [0, 0, 0, 0, 0, 0, 0]
                d = [0.3, 0, 0.328, 0, 0.323, 0, 0.0824]
                al = [-np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, 0]
                th = [0, 0, 0, 0, 0, 0, 0]

                params = {
                    "a": a,
                    "alpha": al,
                    "d": d,
                    "theta": th,
                    "modified_dh": modified_dh,
                    "num_joints": 7,
                }
                robot = RobotRevolute(params)
                graph = ProblemGraphRevolute(robot)
            elif robot_type == "panda":
                modified_dh = False
                a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
                d = [0.333, 0, 0.316, 0, 0.384, 0, 0]
                al = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2]
                th = [0, 0, 0, 0, 0, 0, 0]

                params = {
                    "a": a,
                    "alpha": al,
                    "d": d,
                    "theta": th,
                    "modified_dh": modified_dh,
                    "num_joints": 7,
                }
                robot = RobotRevolute(params)
                graph = ProblemGraphRevolute(robot)
            elif robot_type == "lwa4p":
                modified_dh = False
                a = [0, 0.350, 0, 0, 0, 0]
                d = [0.205, 0, 0, 0.305, 0, 0.075]
                al = [-np.pi / 2, np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2, 0]
                th = [0, 0, 0, 0, 0, 0]

                params = {
                    "a": a,
                    "alpha": al,
                    "d": d,
                    "theta": th,
                    "modified_dh": modified_dh,
                    "num_joints": 6,
                }
                robot = RobotRevolute(params)
                graph = ProblemGraphRevolute(robot)
            else:
                raise NotImplementedError

            print(f"\nEvaluating {robot_type} robot ({evals_per_robot} samples)")
            for kdx in range(evals_per_robot):
                print(f"  Sample {kdx+1}/{evals_per_robot} ", end="", flush=True)
                sol_data = []

                prob_data = generate_data_point(graph).to(device)
                prob_data.num_graphs = 1
                data = model.preprocess(prob_data)
                P_goal = data.pos.cpu().numpy()

                try:
                    T_goal = SE3.exp(data.T_ee.cpu().numpy())
                except ValueError as e:
                    # Handle dimension mismatch silently
                    t_ee_data = data.T_ee.cpu().numpy()
                    if len(t_ee_data.shape) > 1:
                        t_ee_data = t_ee_data[0]
                    
                    if len(t_ee_data) != 6:
                        t_ee_data = t_ee_data[:6]
                        
                    T_goal = SE3.exp(t_ee_data)

                t0 = time.time()
                P_all = model.forward_eval(
                    x=data.pos, 
                    h=torch.cat((data.type, data.goal_data_repeated_per_node), dim=-1), 
                    edge_attr=data.edge_attr, 
                    edge_attr_partial=data.edge_attr_partial, 
                    edge_index=data.edge_index_full, 
                    partial_goal_mask=data.partial_goal_mask, 
                    nodes_per_single_graph=int(data.num_nodes / 1),
                    batch_size=1,
                    num_samples=args.num_samples
                ).cpu().detach().numpy()
                t_sol = time.time() - t0
                print(f"({t_sol:.3f}s)", end="", flush=True)

                e_pose = np.empty([P_all.shape[0]])
                e_pos = np.empty([P_all.shape[0]])
                e_rot = np.empty([P_all.shape[0]])
                q_sols_np = np.empty([P_all.shape[0], robot.n])
                q_sols = []

                for idx in range(P_all.shape[0]):
                    P = P_all[idx, :]

                    if isinstance(P, torch.Tensor):
                        P = P.cpu().numpy()

                    q_sol = graph.joint_variables(
                        graph_from_pos(P, graph.node_ids), {robot.end_effectors[0]: T_goal}
                    )

                    q_sols_np[idx] = np.fromiter(
                        (q_sol[f"p{jj}"] for jj in range(1, graph.robot.n + 1)), dtype=float
                    )

                    T_ee = graph.robot.pose(q_sol, robot.end_effectors[-1])
                    
                    # Calculate positional error as before
                    e_pos[idx] = np.linalg.norm(T_ee.trans - T_goal.trans)
                    
                    # More robust rotation error calculation
                    try:
                        rot_diff = T_ee.rot.inv().dot(T_goal.rot)
                        rot_log = rot_diff.log()
                        
                        # Check for numerical instability
                        if np.isfinite(rot_log).all() and np.max(np.abs(rot_log)) < 100:
                            e_rot[idx] = np.linalg.norm(rot_log)
                        else:
                            # Alternative method for singularity cases
                            # Convert to angle-axis representation
                            theta = np.arccos(np.clip((np.trace(rot_diff.as_matrix()) - 1) / 2, -1.0, 1.0))
                            e_rot[idx] = abs(theta)
                    except Exception as e:
                        print(f"\n    Warning: Rotation calculation error: {e}")
                        # Fallback method
                        e_rot[idx] = np.arccos(np.clip((np.trace(T_ee.rot.inv().dot(T_goal.rot).as_matrix()) - 1) / 2, -1.0, 1.0))
                    
                    # Calculate total pose error with better bounds
                    try:
                        e_pose[idx] = np.linalg.norm(T_ee.inv().dot(T_goal).log())
                        # Check for numerical instability
                        if not np.isfinite(e_pose[idx]) or e_pose[idx] > 100:
                            # Fallback to weighted sum of position and rotation errors
                            e_pose[idx] = e_pos[idx] + 0.5 * e_rot[idx]
                    except Exception as e:
                        print(f"\n    Warning: Pose calculation error: {e}")
                        # Fallback to weighted sum
                        e_pose[idx] = e_pos[idx] + 0.5 * e_rot[idx]

                    entry = {
                        "Id": kdx,
                        "Robot": robot_type,
                        "Goal Pose": T_goal.as_matrix(),
                        "Sol. Config": q_sols_np[idx],
                        "Err. Pose": e_pose[idx],
                        "Err. Position": e_pos[idx],
                        "Err. Rotation": e_rot[idx],
                        "Sol. Time": t_sol,
                    }
                    sol_data.append(entry)
                
                print(f" → Pose error: {np.mean(e_pose):.4f}")
                all_sol_data.append(pd.DataFrame(sol_data))

        pd_data = pd.concat(all_sol_data)

        exp_dir = f"{sys.path[0]}/results/"+ f"{args.id}/"
        os.makedirs(exp_dir, exist_ok=True)
        result_path = os.path.join(exp_dir, "results.pkl")
        pd_data.to_pickle(result_path)
        
        print(f"\n✓ Results saved to {result_path}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        for robot in robot_types:
            robot_data = pd_data[pd_data["Robot"] == robot]
            print(f"  {robot.upper()}: ")
            print(f"    Position Error (mm): mean={robot_data['Err. Position'].mean()*1000:.2f}, min={robot_data['Err. Position'].min()*1000:.2f}, max={robot_data['Err. Position'].max()*1000:.2f}")
            print(f"    Rotation Error (deg): mean={robot_data['Err. Rotation'].mean()*180/np.pi:.2f}, min={robot_data['Err. Rotation'].min()*180/np.pi:.2f}, max={robot_data['Err. Rotation'].max()*180/np.pi:.2f}")
            print(f"    Solution Time (ms): {robot_data['Sol. Time'].mean()*1000:.2f}")


if __name__ == "__main__":
    random.seed(17)
    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    parser.add_argument("--model_path", nargs="*", type=str, required=True, help="Path to folder with model data")
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for PyTorch')
    parser.add_argument("--robots", nargs="*", type=str, default=["planar_chain"], help="Type of robot used")
    parser.add_argument("--n_evals", type=int, default=100, help="Number of evaluations")
    parser.add_argument("--num_samples", type=int, default=100, help="Total number of samples per problem")

    args = parser.parse_args()
    main(args)