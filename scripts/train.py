import os
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from torch.utils import data
import torch.distributed as dist
import torch.multiprocessing as mp

# import latent_planner.utils as utils
from latent_planner.train_utils.train_methods import train_repr_model, get_optimizer
from latent_planner.datasets.seq2 import SeqDataset
from latent_planner.datasets.d4rl_utils import load_environment
from latent_planner.models.autoencoders import VQContinuousVAE
from latent_planner.config import (DefaultConfig, DatasetConfig, TransformerConfig, TrainerConfig,
                                   get_recent, get_recent_model_name, load_config)
from latent_planner.datasets.dataset_utils import copy_save_dataset, CopiedDataset

from easydict import EasyDict


def prepare(env_name, dataset=None, pretrained=False):
    config = DefaultConfig()

    # ============= Dataset =============

    # env_name = "halfcheetah-medium-expert-v0"
    # env_name = "maze2d-medium-v1"
    config.env_name = env_name
    # config.dataset = 'maze2d-medium-sparse-v1'
    config.dataset = dataset

    seq_len = config.subsampled_seq_len * config.step
    config.log_dir = os.path.expanduser(config.log_dir)
    config.save_dir = os.path.expanduser(config.save_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir, exist_ok=True)

    save_dir = None

    if pretrained:
        save_dir = get_recent(config.save_dir)
        dataset_config = load_config(os.path.join(save_dir, 'data_config.pkl'), DatasetConfig)
    else:
        dataset_config = DatasetConfig(
            save_dir=os.path.join(config.save_dir, config.now, 'data_config.pkl'),
            env_name=config.env_name,
            dataset=config.dataset,
            termination_penalty=config.termination_penalty,
            seq_len=seq_len,
            step=config.step,
            discount=config.discount,
            disable_goal=config.disable_goal,
            normalize_sa=config.normalize_state,
            normalize_reward=config.normalize_reward,
            max_path_length=int(config.max_path_length),
            min_path_length=int(config.latent_step),
            device=config.device
        ).save()

    dataset = SeqDataset(dataset_config)
    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    transition_dim = act_dim + 3
    transition_dim += obs_dim if config.task_type == "locomotion" else 128

    # ============= Model =============

    # Total dimension of the input to the transformer
    block_size = config.subsampled_seq_len * transition_dim

    print(f'Dataset size: {len(dataset)} | '
          f'Joined dim: {transition_dim} '
          f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}')

    if pretrained:
        model_config = load_config(os.path.join(save_dir, 'model_config.pkl'), TransformerConfig)
    else:
        model_config = TransformerConfig(
            save_dir=os.path.join(config.save_dir, config.now, 'model_config.pkl'),
            vocab_size=config.vocab_size,
            block_size=block_size,

            n_tokens=config.n_tokens,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            emb_dim=config.emb_dim * config.n_heads,

            observation_dim=obs_dim,
            action_dim=act_dim,
            transition_dim=transition_dim,

            traj_emb_dim=config.traj_emb_dim,
            latent_step=config.latent_step,
            ma_update=config.ma_update,
            residual=config.residual,
            # obs_shape=config.obs_shape,
            # model=config.model,

            action_weight=config.action_weight,
            reward_weight=config.reward_weight,
            value_weight=config.value_weight,
            pos_weight=config.position_weight,
            first_action_weight=config.first_action_weight,
            sum_reward_weight=config.sum_reward_weight,
            last_value_weight=config.last_value_weight,

            emb_dropout_rate=config.embd_pdrop,
            attn_dropout_rate=config.attn_pdrop,
            resid_dropout_rate=config.resid_pdrop,

            bottleneck=config.bottleneck,
            masking=config.masking,
            state_conditional=config.state_conditional
        ).save()

    model = VQContinuousVAE(model_config)
    if pretrained:
        model_name = get_recent_model_name(save_dir, 'state')
        state_dict = th.load(os.path.join(save_dir, model_name))
        state_dict = collections.OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
        model.load_state_dict(state_dict)
        print(f'Loaded model from {os.path.join(save_dir, model_name)}')

    model.to(config.device)
    if config.normalize_state:
        padding_vector = np.zeros(model_config.transition_dim - 1)
        padding_vector = (padding_vector - dataset.mean) / dataset.std
        model.padding_vector = th.from_numpy(padding_vector).to(model.padding_vector)

    # ============= Training =============

    warmup_tokens = len(dataset) * block_size  # number of tokens seen per epoch
    final_tokens = 20 * warmup_tokens  # total number of tokens in dataset

    if pretrained:
        trainer_config = load_config(os.path.join(save_dir, 'trainer_config.pkl'), TrainerConfig)
    else:
        trainer_config = TrainerConfig(
            save_dir=os.path.join(config.save_dir, config.now, 'trainer_config.pkl'),
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            betas=(0.9, 0.95),
            grad_norm_clip=1.0,
            weight_decay=0.1,
            lr_decay=config.lr_decay,
            warmup_tokens=warmup_tokens,
            kl_warmup_tokens=warmup_tokens * 10,
            final_tokens=final_tokens,
            num_workers=8,
            device=config.device
        ).save()

    configs = {'dataset': dataset_config,
               'model': model_config,
               'trainer': trainer_config,
               'all': config}

    return dataset, model, configs


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


# def main_worker(gpu, ngpus_per_node, args):
#     dataset, env, model, configs = prepare()
#
#     args.gpu = gpu
#     th.cuda.set_device(args.gpu)
#     print(f"Use GPU: {args.gpu} for training")
#
#     args.rank = args.rank * ngpus_per_node + gpu
#     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                             world_size=args.world_size, rank=args.rank)
#
#     config = configs['all']
#     trainer_config = configs['trainer']
#
#     trainer_config.batch_size = int(trainer_config.batch_size / ngpus_per_node)
#     trainer_config.num_workers = int((trainer_config.num_workers + ngpus_per_node - 1) / ngpus_per_node)
#
#     model = th.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
#     sampler = data.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)
#
#     n_epochs = int(1e6 / len(dataset) * config.n_epochs_ref)
#     # n_epochs = 1
#     save_freq = int(n_epochs // config.n_saves)
#
#     optimizer = get_optimizer(trainer_config, model)
#     loader = data.DataLoader(dataset, shuffle=False, pin_memory=True,
#                              batch_size=trainer_config.batch_size,
#                              num_workers=trainer_config.num_workers,
#                              sampler=sampler)
#
#     for epoch in range(n_epochs):
#         print(f"\nEpoch {epoch}/{n_epochs} | {config.dataset} | {config.exp_name}")
#
#         train(model, trainer_config, loader, optimizer,
#               test_portion=dataset.test_portion,
#               test_set=dataset.get_test(),
#               n_epochs=epoch,
#               log_freq=50)
#
#         save_epoch = (epoch + 1) // save_freq * save_freq
#         state_path = os.path.join(config.save_dir, f"state_{save_epoch}.pt")
#         print(f"Saving model to {state_path}")
#
#         state = model.state_dict()
#         th.save(state, state_path)

def main(rank, args):
    init_distributed_mode(rank, args)
    local_gpu_id = args.gpu

    dataset, model, configs = prepare(env_name=args['env_name'],
                                      dataset=args['dataset'],
                                      pretrained=args['pretrained'])
    config = configs['all']
    trainer_config = configs['trainer']

    trainer_config.batch_size = args.batch_size // args.ngpus_per_node
    trainer_config.num_workers = args.num_workers // args.ngpus_per_node

    train_sampler = data.DistributedSampler(dataset, shuffle=True)
    batch_sampler = data.BatchSampler(train_sampler, trainer_config.batch_size, drop_last=True)
    train_loader = data.DataLoader(dataset, batch_sampler=batch_sampler,
                                   num_workers=trainer_config.num_workers)

    model = model.cuda(local_gpu_id)
    model = th.nn.parallel.DistributedDataParallel(model, device_ids=[local_gpu_id])

    optimizer = get_optimizer(trainer_config, model.module)

    # n_epochs = int(1e6 / len(dataset) * config.n_epochs_ref)
    n_epochs = 20
    save_freq = int(n_epochs // config.n_saves)

    loss_record = []
    lrs_record = []
    for epoch in range(n_epochs):
        losses, lrs = train_repr_model(model, trainer_config, train_loader, train_sampler, optimizer,
                                       device=config.device,
                                       log_freq=100,
                                       cur_epoch=epoch)

        loss_record.append(losses)
        lrs_record.append(lrs)

        if local_gpu_id == 0:
            save_epoch = (epoch + 1) // save_freq * save_freq
            state_path = os.path.join(config.save_dir, config.now,  f"state_{save_epoch}.pt")
            print(f"Epoch[{epoch}/{n_epochs}] Saving model to {state_path}")

            state = model.state_dict()
            th.save(state, state_path)

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

    # config.env_name = "maze2d-medium-v1"
    args['env_name'] = 'antmaze-large-play-v2'
    args['dataset'] = None
    args['pretrained'] = False

    mp.spawn(main, args=(args,), nprocs=args.ngpus_per_node, join=True)
    # main(None, args)
