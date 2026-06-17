import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

from hydra import initialize, compose
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import src.utils
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from src.diffusion_model import LiftedDenoisingDiffusion
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion import diffusion_utils
from src.datasets.schenker_dataset import SchenkerGraphDataModule, SchenkerDatasetInfos, SchenkerDiffHeteroGraphData
from src.analysis.spectre_utils import PlanarSamplingMetrics
from src.analysis.visualization import NonMolecularVisualization
from src.schenker_gnn.config import DEVICE


def initialize_model():
    os.environ["MASTER_ADDR"] = "localhost"
    # Bind a free port rather than a hardcoded 29500 so repeated runs / ZeroGPU
    # per-call subprocesses don't collide (EADDRINUSE).  Tolerate failure: the
    # process group is a training-era leftover and isn't needed for inference.
    import socket as _socket
    _s = _socket.socket()
    _s.bind(("", 0))
    os.environ["MASTER_PORT"] = str(_s.getsockname()[1])
    _s.close()

    warnings.filterwarnings("ignore", category=PossibleUserWarning)
    torch.set_float32_matmul_precision('medium')
    # NB: no torch.cuda.empty_cache() here — under ZeroGPU's emulated main process
    # cuda.is_available() is True but any real CUDA call triggers a forbidden init.

    if not dist.is_initialized():
        try:
            dist.init_process_group(backend="gloo", init_method="env://", rank=0, world_size=1)
        except Exception:
            pass

    with initialize(config_path="../SchenkerDiff/configs", version_base="1.3"):
        cfg = compose(config_name="config")

    dataset_config = cfg["dataset"]
    datamodule = SchenkerGraphDataModule(cfg)
    sampling_metrics = PlanarSamplingMetrics(datamodule)
    dataset_infos = SchenkerDatasetInfos(datamodule, dataset_config)
    train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
    visualization_tools = NonMolecularVisualization()

    if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    loaded_model = DiscreteDenoisingDiffusion.load_from_checkpoint(checkpoint_path="last-v1.ckpt", **model_kwargs)
    return loaded_model


def main():
    batch_size = 1
    keep_chain = 10
    number_chain_steps = 100

    loaded_model = initialize_model()

    E, r, names, n_nodes_list = sample_r_E(batch_size)
    print(E.shape)
    num_nodes = torch.tensor([int(x) for x in n_nodes_list]).to(loaded_model.device)
    if num_nodes is None:
        n_nodes = loaded_model.node_dist.sample_n(batch_size, loaded_model.device)
    elif type(num_nodes) == int:
        n_nodes = num_nodes * torch.ones(batch_size, device=loaded_model.device, dtype=torch.int)
    else:
        assert isinstance(num_nodes, torch.Tensor)
        n_nodes = num_nodes
    n_max = torch.max(n_nodes).item()

    arange = torch.arange(n_max, device=loaded_model.device).unsqueeze(0).expand(batch_size, -1)
    node_mask = arange < n_nodes.unsqueeze(1)

    z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=loaded_model.limit_dist, node_mask=node_mask)
    X, _, y = z_T.X, z_T.E, z_T.y

    E_transpose = E.permute(0, 2, 1, 3)
    E = torch.maximum(E, E_transpose).to(DEVICE)
    r = r.to(DEVICE)

    assert (E == torch.transpose(E, 1, 2)).all()
    assert number_chain_steps < loaded_model.T
    chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
    chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

    chain_X = torch.zeros(chain_X_size)
    chain_E = torch.zeros(chain_E_size)

    for s_int in reversed(range(0, loaded_model.T)):
        s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
        t_array = s_array + 1
        s_norm = s_array / loaded_model.T
        t_norm = t_array / loaded_model.T

        sampled_s, discrete_sampled_s = loaded_model.sample_p_zs_given_zt(s_norm, t_norm, X, E, r, y, node_mask)
        X, _, y = sampled_s.X, sampled_s.E, sampled_s.y

        discrete_sampled_s_E, _ = loaded_model.apply_node_mask_E_r(E, r, node_mask)

        write_index = (s_int * number_chain_steps) // loaded_model.T
        chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
        chain_E[write_index] = discrete_sampled_s_E[:keep_chain]

    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, _, y = sampled_s.X, sampled_s.E, sampled_s.y
    E, _ = loaded_model.apply_node_mask_E_r(E, r, node_mask)

    if keep_chain > 0:
        chain_X[0] = X[:keep_chain]
        chain_E[0] = E[:keep_chain]

        chain_X = diffusion_utils.reverse_tensor(chain_X)
        chain_E = diffusion_utils.reverse_tensor(chain_E)

        chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
        chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
        assert chain_X.size(0) == (number_chain_steps + 10)

    molecule_list = []
    for i in range(batch_size):
        n = n_nodes[i]
        atom_types = X[i, :n].cpu()
        edge_types = E[i, :n, :n].cpu()
        rhythm_types = r[i, :n, :].cpu()
        molecule_list.append([atom_types, edge_types, rhythm_types, names[i]])


def sample_r_E(batch_size):
    """
    Load `batch_size` processed score graphs and return batched edge and rhythm tensors.

    Returns:
        E_tensor: (batch_size, n_nodes, n_nodes, 10)
        r_tensor: (batch_size, n_nodes, dr)
        name_list: list of piece names
        node_sizes: list of node counts per sample
    """
    E_list = []
    r_list = []
    name_list = []
    node_sizes = []

    for _ in range(batch_size):
        idx = 1
        file_path = f"data/schenker/processed/heterdatacleaned/processed/{idx}_processed.pt"
        data_dict = torch.load(file_path)
        data = SchenkerDiffHeteroGraphData.hetero_to_data(data_dict)

        m = data.x.shape[0]
        E_sample = torch.zeros((m, m, 10))
        for i in range(data.edge_index.shape[1]):
            u = data.edge_index[0, i].item()
            v = data.edge_index[1, i].item()
            if u < m and v < m:
                E_sample[u, v, :] = data.edge_attr[i, :]

        dr = data.r.shape[1]
        r_sample = torch.zeros((m, dr))
        r_sample[:m, :] = data.r[:m, :]

        E_list.append(E_sample)
        r_list.append(r_sample)
        name_list.append(data_dict['name'])
        node_sizes.append(m)

    max_nodes = max(t.shape[0] for t in r_list)

    E_padded = [F.pad(e, (0, 0, 0, max_nodes - e.shape[0], 0, max_nodes - e.shape[0])) for e in E_list]
    r_padded = [F.pad(r, (0, 0, 0, max_nodes - r.shape[0])) for r in r_list]

    return torch.stack(E_padded, dim=0), torch.stack(r_padded, dim=0), name_list, node_sizes


if __name__ == "__main__":
    main()
