import os
import numpy as np
import torch as th
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

from typing import Union


def copy_save_dataset(dataset, save_dir, force=False):
    save_name = f"{dataset.name}_"
    if all([os.path.exists(os.path.join(save_dir, save_name + f"{x}.npy")) for x in ["joined", "masks", "terminals"]]):
        if not force:
            print("Dataset already exists, skipping")
            return

    joined = []
    masks = []
    terminals = []
    for i in tqdm(range(len(dataset))):
        X, Y, mask, terminal = dataset[i]
        if isinstance(X, th.Tensor):
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()

        traj = np.zeros((X.shape[0] + 1, *X.shape[1:]))
        traj[:-1] = X
        traj[-1] = Y[-1]
        joined.append(traj)

        masks.append(mask)
        terminals.append(terminal)

    joined = np.stack(joined)
    masks = np.stack(masks)
    terminals = np.stack(terminals)

    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, save_name + "joined.npy"), joined)
    np.save(os.path.join(save_dir, save_name + "masks.npy"), masks)
    np.save(os.path.join(save_dir, save_name + "terminals.npy"), terminals)

    infos = {
        'obs_std': dataset.obs_std,
        'obs_mean': dataset.obs_mean,
        'act_std': dataset.act_std,
        'act_mean': dataset.act_mean,
        'reward_std': dataset.reward_std,
        'reward_mean': dataset.reward_mean,
        'value_std': dataset.value_std,
        'value_mean': dataset.value_mean,
    }

    with open(os.path.join(save_dir, save_name + "infos.pkl"), "wb") as f:
        pickle.dump(infos, f)

    print("Saved dataset to", save_dir)


class CopiedDataset(Dataset):
    def __init__(self, save_dir: str, name: str):
        self.name = name
        self.joined = np.load(os.path.join(save_dir, f"{name}_joined.npy"))
        self.masks = np.load(os.path.join(save_dir, f"{name}_masks.npy"))
        self.terminals = np.load(os.path.join(save_dir, f"{name}_terminals.npy"))

        infos = pickle.load(open(os.path.join(save_dir, f"{name}_infos.pkl"), "rb"))
        self.obs_std = infos['obs_std']
        self.obs_mean = infos['obs_mean']
        self.act_std = infos['act_std']
        self.act_mean = infos['act_mean']
        self.reward_std = infos['reward_std']
        self.reward_mean = infos['reward_mean']
        self.value_std = infos['value_std']
        self.value_mean = infos['value_mean']

    def __len__(self):
        return self.joined.shape[0]

    def __getitem__(self, idx):
        X = self.joined[idx, :-1]
        Y = self.joined[idx, 1:]
        mask = self.masks[idx]
        terminal = self.terminals[idx]
        return X, Y, mask, terminal

    def normalize_joined_single(self, joined):
        joined_std = np.concatenate([self.obs_std[0], self.act_std[0], self.reward_std[None], self.value_std[None]])
        joined_mean = np.concatenate([self.obs_mean[0], self.act_mean[0], self.reward_mean[None], self.value_mean[None]])
        return (joined-joined_mean) / joined_std


def to_array(x, vec: Union[np.ndarray, th.Tensor]):
    if th.is_tensor(vec):
        return th.tensor(x).to(vec.device)
    return np.squeeze(x)


def normalize(x, mean, std):
    mean = to_array(x, mean)
    std = to_array(x, std)
    return (x - mean) / std


def denormalize(x, mean, std):
    mean = to_array(x, mean)
    std = to_array(x, std)
    return x * std + mean


def denormalize_joined(x, mean, std):
    denormalized = denormalize(x[..., :-1], mean, std)
    if th.is_tensor(x):
        return th.cat([denormalized, x[..., -1:]], dim=-1)
    return np.concatenate([denormalized, x[..., -1:]], axis=-1)