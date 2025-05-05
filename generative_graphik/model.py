import random
import time

import torch
import torch.nn as nn

from generative_graphik.networks.eqgraphatt import EqGraphAtt
from generative_graphik.networks.eqgraph import EqGraph
from generative_graphik.networks.gatgraph import GATGraph
from generative_graphik.networks.gcngraph import GCNGraph
from generative_graphik.networks.sagegraph import SAGEGraph
from generative_graphik.networks.mpnngraph import MPNNGraph
from generative_graphik.networks.linearvae import LinearVAE
from generative_graphik.utils.torch_utils import (MixtureGaussianDiag,
                                                  MultivariateNormalDiag,
                                                  kl_divergence,
                                                  torch_log_from_T,
                                                  repeat_offset_index)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.n_beta_scaling_epoch = args.n_beta_scaling_epoch
        self.rec_gain = args.rec_gain
        self.num_anchor_nodes = args.num_anchor_nodes
        self.train_prior = args.train_prior
        self.max_num_iterations = args.num_iterations

        if args.non_linearity == "relu":
            non_linearity = nn.ReLU()
        elif args.non_linearity == "silu":
            non_linearity = nn.SiLU()
        elif args.non_linearity == "elu":
            non_linearity = nn.ELU()
        else:
            raise NotImplementedError

        if args.gnn_type == "egnn":
            gnn = EqGraph
        elif args.gnn_type == "egnnatt":
            gnn = EqGraphAtt
        elif args.gnn_type == "gat":
            gnn = GATGraph
        elif args.gnn_type == "gcn":
            gnn = GCNGraph
        elif args.gnn_type == "mpnn":
            gnn = MPNNGraph
        elif args.gnn_type =="graphsage":
            gnn = SAGEGraph
        else:
            raise NotImplementedError

        self.goal_config_encoder = gnn(
            latent_dim=args.dim_latent,
            out_channels_node=args.dim_latent_node_out,
            coordinates_dim=args.num_coordinates_in,
            node_features_dim=args.num_features_in + args.dim_goal,
            mlp_hidden_size=args.graph_mlp_hidden_size,
            num_graph_mlp_layers=args.num_graph_mlp_layers,
            num_egnn_mlp_layers=args.num_egnn_mlp_layers,
            num_gnn_layers=args.num_gnn_layers,
            norm_layer=args.norm_layer,
            stochastic=False,
            num_mixture_components=1,
            non_linearity=non_linearity
        )

        self.inference_encoder = LinearVAE(
            dim_in=2 * args.dim_latent_node_out,
            dim_out=args.dim_latent_node_out,
            norm_layer=args.norm_layer,
            hidden_size=args.mlp_hidden_size,
            stochastic=True,
            non_linearity=non_linearity
        )
        self.qz_x_dist = MultivariateNormalDiag

        self.goal_partial_config_encoder = gnn(
            latent_dim=args.dim_latent,
            out_channels_node=args.dim_latent_node_out,
            coordinates_dim=args.num_coordinates_in,
            node_features_dim=args.num_features_in + args.dim_goal,
            mlp_hidden_size=args.graph_mlp_hidden_size,
            num_graph_mlp_layers=args.num_graph_mlp_layers,
            num_egnn_mlp_layers=args.num_egnn_mlp_layers,
            num_gnn_layers=args.num_gnn_layers,
            norm_layer=args.norm_layer,
            stochastic=False,
            num_mixture_components=1,
            non_linearity=non_linearity
        )
        if self.train_prior:
            self.prior_encoder = LinearVAE(
                dim_in=args.dim_latent_node_out,
                dim_out=args.dim_latent_node_out,
                norm_layer=args.norm_layer,
                hidden_size=args.mlp_hidden_size,
                stochastic=True,
                num_mixture_components=args.num_prior_mixture_components,
                non_linearity=non_linearity
            )

            if args.num_prior_mixture_components == 1:
                self.pz_c_dist = MultivariateNormalDiag
            else:
                self.pz_c_dist = MixtureGaussianDiag
        else:
            self.pz_c_dist = MultivariateNormalDiag

        self.decoder = gnn(
            latent_dim=args.dim_latent,
            out_channels_node=args.num_node_features_out,
            coordinates_dim=2 * args.dim_latent_node_out,
            node_features_dim=args.num_features_in + args.dim_goal,
            mlp_hidden_size=args.graph_mlp_hidden_size,
            num_graph_mlp_layers=args.num_graph_mlp_layers,
            num_egnn_mlp_layers=args.num_egnn_mlp_layers,
            num_gnn_layers=args.num_gnn_layers,
            norm_layer=args.norm_layer,
            stochastic=False,
            num_mixture_components=args.num_likelihood_mixture_components,
            non_linearity=non_linearity,
        )
        
        if args.num_likelihood_mixture_components == 1:
            self.px_z_dist = MultivariateNormalDiag
        else:
            self.px_z_dist = MixtureGaussianDiag

    def preprocess(self, data):
        data["edge_index_full"] = data.edge_index_full.type(torch.long)
        data["edge_index_partial"] = data.edge_index_full[:, data.partial_mask].type(torch.long)
        data["edge_attr_partial"] = data.partial_mask.unsqueeze(-1) * data.edge_attr
        data["T_ee"] = torch_log_from_T(data.T_ee)

        # Workaround for LieGroups batching
        if data["T_ee"].dim() == 1:
            data["T_ee"] = data["T_ee"].unsqueeze(0)

        dim = data.T_ee.shape[-1] // 3 + 1
        goal_repeated = torch.repeat_interleave(
            data.T_ee,
            (dim - 1) * data.num_joints + self.num_anchor_nodes,
            dim=0
        )
        data["goal_data_repeated_per_node"] = goal_repeated

        ## Always build node features as type + goal encoding
        data["h"] = torch.cat([data["type"], data["goal_data_repeated_per_node"]], dim=-1)

        return data

    def forward(self, x, h, edge_attr, edge_attr_partial, edge_index, partial_goal_mask):
        # Goal T_g encoder, all edge attributes (distances)
        # Sanity check for edge_attr shape and distance
        if edge_attr.shape[-1] != 4:
            raise ValueError(f"Expected edge_attr to have 4 features (dist + 3D vec), but got shape {edge_attr.shape}")

        if not torch.all(edge_attr[:, 0] >= 0):
            print("⚠️ Warning: Some edge distances (edge_attr[:, 0]) are negative or NaN")
        
        z_goal = self.goal_config_encoder(
            x=x,
            h=h,
            edge_attr=edge_attr,
            edge_index=edge_index,
        )
        
        # AlphaFold style sampling of iterations to encourage fast convergence
        num_iterations = random.randint(1, self.max_num_iterations)

        for _ in range(num_iterations):

            # unknown distances and positions transformed to 0
            z_goal_partial = self.goal_partial_config_encoder(
                x=partial_goal_mask[:, None] * x,
                h=h,
                edge_attr=edge_attr_partial,
                edge_index=edge_index,
            )

            # Encode conditional prior p(z | c)
            if self.train_prior:            
                params = self.prior_encoder(z_goal_partial)
                pz_c =  self.pz_c_dist(*params)
            else:
                pz_c = self.pz_c_dist(
                    loc=torch.zeros_like(params[0]),
                    scale=torch.ones_like(params[1])
                )

            # Encode inference distribution q(z | x, c)
            inp = torch.cat((
                z_goal,
                z_goal_partial,
            ), dim=-1)
            params = self.inference_encoder(inp)
            qz_xc = self.qz_x_dist(*params)
            z = qz_xc.rsample()

            # Decode distribution p(x | z, c)
            inp_decoder = torch.cat((
                z,
                z_goal_partial,
            ), dim=-1)
            mu_x_sample = self.decoder(
                x=inp_decoder,
                h=h,
                edge_attr=0.0 * edge_attr,
                edge_index=edge_index,
            )

            # Decode distribution p(x | z, c) if we're going to iterate
            if self.max_num_iterations > 1 and self.train_prior:
                z_prior = pz_c.sample()
                inp_decoder_prior = torch.cat((
                    z_prior,
                    z_goal_partial
                ), dim=-1)
                mu_x_sample_prior = self.decoder(
                    x=inp_decoder_prior,
                    h=h,
                    edge_attr=0.0 * edge_attr,
                    edge_index=edge_index,
                )
                nodes = mu_x_sample_prior
                src, dst = edge_index  
                edges = ((nodes[src] - nodes[dst])**2).sum(dim=-1).sqrt()
                edges = edges.unsqueeze(-1)

        return {
            "mu_x_sample": mu_x_sample,
            "qz_xc": qz_xc,
            "pz_c": pz_c
        }

    def forward_eval(self, x, h, edge_attr, edge_attr_partial, edge_index, partial_goal_mask, nodes_per_single_graph, num_samples, batch_size):
        # Sanity check
        if edge_attr.shape[-1] != 4:
            raise ValueError(f"Expected edge_attr to have 4 features (dist + 3D vec), but got shape {edge_attr.shape}")
        if not torch.all(edge_attr[:, 0] >= 0):
            print("⚠️ Warning: Some edge distances (edge_attr[:, 0]) are negative or NaN")
        for ii in range(self.max_num_iterations):
            with torch.no_grad():
                # unknown distances and positions transformed to 0
                # tic = time.time()
                # torch.cuda.synchronize()
                z_goal_partial = self.goal_partial_config_encoder(
                    x=partial_goal_mask[:, None] * x,
                    h=h,
                    edge_attr=edge_attr_partial,
                    edge_index=edge_index,
                )
                # torch.cuda.synchronize()
                # print(f"Goal encoder time: {time.time() - tic}")

                # Encode conditional prior p(z | c)
                # tic = time.time()
                # torch.cuda.synchronize()
                if self.train_prior:
                    params = self.prior_encoder(z_goal_partial)
                    pz_c =  self.pz_c_dist(*params)
                else:
                    pz_c = self.pz_c_dist(
                        loc=torch.zeros((batch_size * nodes_per_single_graph, z_goal_partial.shape[-1])).to(device=z_goal_partial.device),
                        scale=torch.ones((batch_size * nodes_per_single_graph, z_goal_partial.shape[-1])).to(device=z_goal_partial.device),
                    )
                # torch.cuda.synchronize()
                # print(f"Prior encoder time: {time.time() - tic}")

                # tic = time.time()
                # torch.cuda.synchronize()
                # Repeat data num_samples times
                if ii == 0:
                    z_prior = pz_c.sample([num_samples])
                    z_prior = z_prior.reshape(-1, z_prior.shape[-1])
                    z_goal_partial = z_goal_partial.unsqueeze(0).expand(num_samples, -1, -1)
                    z_goal_partial = z_goal_partial.reshape(-1, z_goal_partial.shape[-1])
                    h = h.unsqueeze(0).expand(num_samples,-1,-1)    
                    h = h.reshape(-1, h.shape[-1])
                    data_index = edge_index
                    data_index = repeat_offset_index(data_index, num_samples, nodes_per_single_graph)
                    data_index = data_index.reshape(data_index.shape[0], -1)
                    data_edge_attr = edge_attr.unsqueeze(0).expand(num_samples, -1, -1)
                    data_edge_attr = data_edge_attr.reshape(-1, data_edge_attr.shape[-1])
                else:
                    z_prior = pz_c.sample()
                # torch.cuda.synchronize()
                # print(f"Sampling time: {time.time() - tic}")

                # tic = time.time()
                # torch.cuda.synchronize()
                # Decode distribution p(x | z, c)
                inp_decoder_prior = torch.cat((
                    z_prior,
                    z_goal_partial
                ), dim=-1)
                mu_x_sample = self.decoder(
                    x=inp_decoder_prior,
                    h=h,
                    edge_attr=0.0 * data_edge_attr,
                    edge_index=data_index,
                )
                # torch.cuda.synchronize()
                # print(f"Decoder time: {time.time() - tic}")

                if self.max_num_iterations > 1 and self.train_prior:
                    nodes = mu_x_sample
                    src, dst = data_index
                    edges = ((nodes[src] - nodes[dst])**2).sum(dim=-1).sqrt()
                    edges = edges.unsqueeze(-1)

        mu_x_sample = mu_x_sample.reshape(num_samples, batch_size * nodes_per_single_graph, -1)
        return mu_x_sample
    def loss(self, res, epoch, batch_size, goal_pos, partial_goal_mask):
        mu_x_sample = res["mu_x_sample"]
        partial_non_goal_mask = torch.ones_like(partial_goal_mask) - partial_goal_mask
        beta_kl = min(((epoch + 1) / self.n_beta_scaling_epoch), 1.0)
        stats = {}

        # Point loss
        loss_anchor = torch.sum((partial_goal_mask[:,None]*(mu_x_sample - goal_pos))**2) / (batch_size)
        loss_non_anchor = torch.sum(partial_non_goal_mask[:,None]*((mu_x_sample - goal_pos)**2)) / (batch_size)
        loss_rec_pos_opt = loss_anchor + self.rec_gain * loss_non_anchor
        loss_rec_pos = torch.sum((mu_x_sample - goal_pos)**2) / (batch_size)
        stats["rec_pos_l"] = loss_rec_pos.item()

        # Distance loss
        # src, dst = res["edge_index_partial"]
        # # src, dst = res["edge_index_full"]    
        # dist_samples = ((mu_x_sample[src] - mu_x_sample[dst])**2).sum(dim=-1).sqrt()
        # dist = ((goal_pos[src] - goal_pos[dst])**2).sum(dim=-1).sqrt()
        # loss_rec_dist = torch.sum((dist_samples - dist)**2) / (batch_size)
        # stats["rec_dist_l"] = loss_rec_dist.item()

        qz_xc = res["qz_xc"]
        pz_c = res["pz_c"]
        loss_kl = torch.sum(kl_divergence(qz_xc, pz_c)) / (batch_size)
        stats["kl_l"] = loss_kl.item()
        
        # Point loss
        loss = loss_rec_pos + loss_kl
        loss_opt = loss_rec_pos_opt + beta_kl * loss_kl# + loss_sc

        # Distance loss
        # loss = loss_rec_dist + loss_kl
        # loss_opt = self.rec_gain * loss_rec_dist + beta_kl * loss_kl

        stats["total_l"] = loss.item()
        return loss_opt, stats
