import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.functional as F
from torch.utils import data

from latent_planner.train_utils.timer import Timer
from latent_planner.config import TransformerConfig, TrainerConfig
from latent_planner.models.autoencoders import VQContinuousVAE, TransformerPrior
from latent_planner.train_utils.scheduler import CosineAnnealingWarmupRestarts


def to(xs, device):
    return [x.to(device) for x in xs]


def get_optimizer(config: TrainerConfig, model: VQContinuousVAE) -> th.optim.Optimizer:
    return model.configure_optimizers(config)


def train_repr_model(model: VQContinuousVAE,
                     config: TrainerConfig,
                     train_loader: data.DataLoader,
                     train_sampler: data.DistributedSampler,
                     optimizer: th.optim.Optimizer,
                     device: th.device,
                     log_freq: int = 100,
                     cur_epoch: int = 0):

    model.train(True)

    if train_sampler is not None:
        train_sampler.set_epoch(cur_epoch)

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=len(train_loader) // 5,
                                              cycle_mult=2.0,
                                              max_lr=config.learning_rate,
                                              min_lr=config.learning_rate / 100,
                                              warmup_steps=int(len(train_loader) // 10),
                                              gamma=0.5,
                                              last_epoch=-1)

    losses = []
    lrs = []
    timer = Timer()
    for i, batch in enumerate(train_loader):
        traj, next_traj, mask, terminals = to(batch, device)

        traj_recon, recon_loss, vq_loss, commit_loss = model(traj, next_traj, mask, terminals)
        loss = (recon_loss + vq_loss + commit_loss).mean()
        losses.append(loss.item())

        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
        optimizer.step()
        scheduler.step()

        lrs.append(scheduler.get_lr()[0])

        if i % log_freq == 0:
            print(f"Epoch ({cur_epoch}): [{i:4d} | {len(train_loader):4d}] | "
                  f"losses: {losses[-1]:.5f} | "
                  f"recon: {recon_loss.mean().item():.5f} | vq: {vq_loss.mean().item():.5f} | commit: {commit_loss.mean().item():.5f} | "
                  f"td: {timer():.2f} | "
                  f"lr: {scheduler.get_lr()[0]:.5e}")

            th.cuda.empty_cache()

    return losses, lrs


def train_prior_model(repr_model: VQContinuousVAE,
                      dyna_model: TransformerPrior,
                      config: TrainerConfig,
                      train_loader: data.DataLoader,
                      train_sampler: data.DistributedSampler,
                      optimizer: th.optim.Optimizer,
                      device: th.device,
                      log_freq: int = 100,
                      cur_epoch: int = 0,
                      **kwargs):

    repr_model.train(False)
    dyna_model.train(True)

    if train_sampler is not None:
        train_sampler.set_epoch(cur_epoch)

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=len(train_loader) // 5,
                                              cycle_mult=2.0,
                                              max_lr=config.learning_rate,
                                              min_lr=config.learning_rate / 100,
                                              warmup_steps=int(len(train_loader) // 10),
                                              gamma=0.5,
                                              last_epoch=-1)

    losses = []
    lrs = []
    timer = Timer()

    for i, batch in enumerate(train_loader):
        traj, next_traj, mask, terminals = to(batch, device)

        with th.no_grad():
            obs = traj[:, 0, :kwargs['obs_dim']]
            indices = repr_model(traj, terminals=terminals)

        logits, loss = dyna_model(indices[..., :-1], condition=obs, targets=indices)
        losses.append(loss.item())

        dyna_model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(dyna_model.parameters(), config.grad_norm_clip)
        optimizer.step()
        scheduler.step()

        lrs.append(scheduler.get_lr()[0])

        if i % log_freq == 0:
            print(f"Epoch ({cur_epoch}): [{i:4d} | {len(train_loader):4d}] | "
                  f"losses: {losses[-1]:.5f} | "
                  f"td: {timer():.2f} | "
                  f"lr: {scheduler.get_lr()[0]:.5e}")

    return losses, lrs
