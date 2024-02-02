import os
import collections
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from torch.utils import data
import torch.distributed as dist
import torch.multiprocessing as mp

from latent_planner.train_utils.train_methods import train_prior_model, get_optimizer
from latent_planner.models.autoencoders import VQContinuousVAE, TransformerPrior, VQContinuousVAEEncWrap
from latent_planner.datasets.seq2 import SeqDataset
from latent_planner.config import (DefaultConfig, TransformerConfig, TrainerConfig,
                                   load_configs, get_recent)

from easydict import EasyDict


def prepare(env_name, model_name: str, dataset=None):
    config = DefaultConfig()
    # env_name = "maze2d-medium-v1"
    config.env_name = env_name
    config.dataset = dataset

    save_dir = get_recent(config.save_dir)
    dataset_config, model_config, trainer_config = load_configs(save_dir)

    # ============= Dataset =============
    dataset = SeqDataset(dataset_config)

    # ============= Model =============
    model = VQContinuousVAE(model_config)
    state_dict = th.load(os.path.join(save_dir, model_name))
    state_dict = collections.OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict)
    print(f'Loaded model from {os.path.join(save_dir, model_name)}')
    model.to(config.device)
    if config.normalize_state:
        padding_vector = np.zeros(model_config.transition_dim - 1)
        padding_vector = (padding_vector - dataset.mean) / dataset.std
        model.padding_vector = th.from_numpy(padding_vector).to(model.padding_vector)

    block_size = config.subsampled_seq_len // config.latent_step
    prior_config = TransformerConfig(
        save_dir=(save_dir, 'prior_model_config.pkl'),
        n_tokens=config.n_tokens,
        block_size=block_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        emb_dim=config.emb_dim * config.n_heads,
        latent_step=config.latent_step,

        observation_dim=dataset.observation_dim,
        # obs_shape=config.obs_shape,

        emb_dropout_rate=config.embd_pdrop,
        attn_dropout_rate=config.attn_pdrop,
        resid_dropout_rate=config.resid_pdrop,

        action_dim=model_config.action_dim,
        transition_dim=model_config.transition_dim,
        traj_emb_dim=model_config.traj_emb_dim,
        action_weight=model_config.action_weight,
        reward_weight=model_config.reward_weight,
        value_weight=model_config.value_weight,
        pos_weight=model_config.pos_weight,
        first_action_weight=model_config.first_action_weight,
        sum_reward_weight=model_config.sum_reward_weight,
        last_value_weight=model_config.last_value_weight,
        vocab_size=model_config.vocab_size,
        state_conditional=model_config.state_conditional,
    ).save()

    prior = TransformerPrior(prior_config)
    prior.to(config.device)

    # ============= Trainer =============

    warmup_tokens = len(dataset) * block_size
    final_tokens = 20 * warmup_tokens

    trainer_config = TrainerConfig(
        save_dir=(save_dir, 'prior_trainer_config.pkl'),
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        betas=(0.9, 0.95),
        grad_norm_clip=1.0,
        weight_decay=0.1,
        lr_decay=config.lr_decay,
        warmup_tokens=warmup_tokens,
        kl_warmup_tokens=warmup_tokens * 10,
        final_tokens=final_tokens,
        num_workers=0,
        device=config.device
    ).save()

    configs = {'dataset': dataset_config,
               'model': model_config,
               'prior': prior_config,
               'trainer': trainer_config,
               'all': config,
               'save_dir': save_dir}

    return dataset, model, prior, configs


def init_distributed_mode(rank, args):
    args.rank = rank
    args.gpu = args.rank % th.cuda.device_count()
    local_gpu_id = int(args.gpu_ids[args.rank])
    th.cuda.set_device(local_gpu_id)

    if args.rank is not None:
        print(f"Use GPU: {local_gpu_id} for training")

    dist.init_process_group(backend='nccl',
                            init_method=f"tcp://127.0.0.1:{args.port}",
                            world_size=args.ngpus_per_node,
                            rank=args.rank)

    dist.barrier()
    setup_for_distributed(args.rank == 0)
    print(f"Initialized the process group: {args.rank} / {args.ngpus_per_node}")


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def main(rank, args):
    init_distributed_mode(rank, args)
    local_gpu_id = args.gpu

    dataset, model, prior, configs = prepare(args['env_name'], args['model_name'], args['dataset'])
    config = configs['all']
    dataset_config = configs['dataset']
    prior_config = configs['prior']
    trainer_config = configs['trainer']

    trainer_config.batch_size = args.batch_size // args.ngpus_per_node
    trainer_config.num_workers = args.num_workers // args.ngpus_per_node

    train_sampler = data.DistributedSampler(dataset, shuffle=True)
    batch_sampler = data.BatchSampler(train_sampler, trainer_config.batch_size, drop_last=True)
    train_loader = data.DataLoader(dataset, batch_sampler=batch_sampler,
                                   num_workers=trainer_config.num_workers)

    model = VQContinuousVAEEncWrap(model)
    model = model.cuda(local_gpu_id)
    model = th.nn.parallel.DistributedDataParallel(model, device_ids=[local_gpu_id])

    prior = prior.cuda(local_gpu_id)
    prior = th.nn.parallel.DistributedDataParallel(prior, device_ids=[local_gpu_id])

    optimizer = get_optimizer(trainer_config, prior.module)

    # n_epochs = int(1e6 / len(dataset) * config.n_epochs_ref)
    n_epochs = 51
    save_freq = int(n_epochs // config.n_saves)

    loss_record = []
    lrs_record = []
    for epoch in range(n_epochs):
        losses, lrs = train_prior_model(model, prior, trainer_config, train_loader, train_sampler, optimizer,
                                        device=config.device, log_freq=100, cur_epoch=epoch,
                                        obs_dim=prior_config.observation_dim,)

        loss_record.append(losses)
        lrs_record.append(lrs)

        if local_gpu_id == 0:
            save_epoch = (epoch + 1) // save_freq * save_freq
            model_save_name = os.path.join(configs['save_dir'], f'prior_{save_epoch}.pt')
            print(f"Epoch[{epoch}/{n_epochs}] Saving model to {model_save_name}")

            state = prior.state_dict()
            th.save(state, model_save_name)

            fig, ax1 = plt.subplots()
            ax1.set_xlabel('iteration')
            ax1.set_ylabel('loss')
            ax2 = ax1.twinx()
            ax2.set_ylabel('lr', color='r')
            ax1.plot(np.array(loss_record).flatten())
            ax2.plot(np.array(lrs_record).flatten(), 'r', alpha=0.5)
            fig.tight_layout()
            plt.show()


if __name__ == '__main__':
    args = EasyDict(epoch=3,
                    batch_size=2512,
                    port=23456,
                    local_rank=0)

    args['ngpus_per_node'] = th.cuda.device_count()
    args['gpu_ids'] = list(range(args.ngpus_per_node))
    args['num_workers'] = args.ngpus_per_node * 4

    args['env_name'] = "antmaze-large-play-v2"
    args['model_name'] = 'state_18.pt'
    args['dataset'] = None

    mp.spawn(main, args=(args,), nprocs=args.ngpus_per_node, join=True)
    # main(0, args)
