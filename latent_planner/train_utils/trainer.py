import math
# import wandb
import numpy as np
import torch as th
from torch.utils import data

from latent_planner.config import TrainerConfig
from latent_planner.train_utils.timer import Timer


def to(xs, device):
    return [x.to(device) for x in xs]


def get_optimizer(config: TrainerConfig, model) -> th.optim.Optimizer:
    return model.configure_optimizers(config)


def train(model,
          config: TrainerConfig,
          train_loader: data.DataLoader,
          train_sampler: data.DistributedSampler,
          optimizer: th.optim.Optimizer,
          log_freq: int = 100,
          test_portion: float = 0.0,
          test_set = None,
          n_epochs: int = 0):

    model.train(True)
    train_sampler.set_epoch(n_epochs)

    n_tokens = 0
    losses = []
    timer = Timer()
    for it, batch_numpy in enumerate(train_loader):
        batch = to(batch_numpy, config.device)
        n_tokens += np.prod(batch[-2].shape)
        if n_tokens < config.warmup_tokens:
            lr_mult = float(n_tokens) / float(max(1, config.warmup_tokens))
        else:
            progress = float(n_tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        if config.lr_decay:
            lr = config.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = config.learning_rate

        with th.set_grad_enabled(True):
            *_, recon_loss, vq_loss, commit_loss = model(*batch)
            loss = (recon_loss + vq_loss + commit_loss).mean()
            losses.append(loss.item())

        model.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
        optimizer.step()

        if it % log_freq == 0:
            if test_portion == 0:
                summary = dict(recontruction_loss=recon_loss.item(),
                               commit_loss=commit_loss.item(),
                               lr=lr,
                               lr_mulr=lr_mult,
                               )
                print(
                    f'[ utils/training ] epoch {n_epochs} [ {it:4d} / {len(train_loader):4d} ] ',
                    f'train reconstruction loss {recon_loss.item():.5f} | '
                    f' train commit loss {commit_loss.item():.5f} |'
                    f' | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                    f't: {timer():.2f}')
            else:
                th.cuda.empty_cache()
                model.eval()
                with th.set_grad_enabled(False):
                    _, t_recon_loss, t_vq_loss, t_commit_loss = model(*to(test_set, config.device))
                model.train()
                summary = dict(recontruction_loss=recon_loss.item(),
                               commit_loss=commit_loss.item(),
                               test_reconstruction_loss=t_recon_loss.item(),
                               lr=lr,
                               lr_mulr=lr_mult,
                               )
                print(
                    f'[ utils/training ] epoch {n_epochs} [ {it:4d} / {len(train_loader):4d} ] ',
                    f'train reconstruction loss {recon_loss.item():.5f} |'
                    f' train commit loss {commit_loss.item():.5f} |'
                    f' test reconstruction loss {t_recon_loss.item():.5f} |'
                    f' | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                    f't: {timer():.2f}')
            # wandb.log(summary, step=self.n_epochs*len(loader)+it)
        if test_portion >= 0:
            th.cuda.empty_cache()

    return losses


def train_prior(repr_model,
                dyna_model,
                config: TrainerConfig,
                train_loader: data.DataLoader,
                train_sampler: data.DistributedSampler,
                optimizer: th.optim.Optimizer,
                obs_dim: int,
                log_freq: int = 100,
                n_epochs: int = 0):

    repr_model.train(False)
    dyna_model.train(True)
    if train_sampler is not None:
        train_sampler.set_epoch(n_epochs)

    n_tokens = 0
    losses = []
    timer = Timer()

    for it, batch_numpy in enumerate(train_loader):
        batch = to(batch_numpy, config.device)
        n_tokens += np.prod(batch[1].shape)
        if n_tokens < config.warmup_tokens:
            lr_mult = float(n_tokens) / float(max(1, config.warmup_tokens))
        else:
            progress = float(n_tokens - config.warmup_tokens) / float(
                max(1, config.final_tokens - config.warmup_tokens))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        if config.lr_decay:
            lr = config.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = config.learning_rate

        obs = batch[0][:, 0, :obs_dim]
        # indices = repr_model.encode(batch[0], terminals=batch[-1])
        indices = repr_model(batch[0], terminals=batch[-1])

        with th.set_grad_enabled(True):
            logits, loss = dyna_model(indices[:, :-1], condition=obs, targets=indices)
            losses.append(loss.item())

        dyna_model.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(dyna_model.parameters(), config.grad_norm_clip)
        optimizer.step()

        if it % log_freq == 0:
            summary = dict(loss=loss.item(),
                           lr=lr,
                           lr_mulr=lr_mult, )
            print(
                f'[ utils/training ] epoch {n_epochs} [ {it:4d} / {len(train_loader):4d} ] ',
                f' train loss {loss.item():.5f} |'
                f' | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                f't: {timer():.2f}')
            # wandb.log(summary, step=self.n_epochs * len(loader) + it)

    return losses