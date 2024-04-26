import os
import functools
from typing import Tuple, Dict
import tqdm
import random
import numpy as np
import time
import datetime
import torch.multiprocessing as mp
import torch
import csv
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    StateDictType,
)

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from custom_dataset import FineTuneDataset

from configs.finetune import config as cfg

from utils import (
    CosineDecayWithWarmupLRScheduler,
    Memory_Maximizer,
    StatsTracker,
    load_full_state_model_checkpoint,
    load_full_state_optimizer_checkpoint,
    save_full_state_model_checkpoint,
    save_full_state_optimizer_checkpoint,
    format_to_gb,
)


def setup(rank, world_size):
    # initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print('in init_process_group, rank={0}'.format(rank))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f'clearing cache for rank {rank}')
    torch.cuda.empty_cache()


# FSDP activation checkpointing
non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    # check_fn = lambda submodule: isinstance(submodule, EncoderBlock)  # noqa: E731
    checkpoint_tformer_layers_policy = lambda submodule: isinstance(submodule, GPT2Block)

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=checkpoint_tformer_layers_policy)


def create_trace_profiler(tb_trace_dir):
    torch_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_trace_dir),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )

    return torch_profiler


def compute_finetune_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert len(logits.shape) == 3  # [B, max_seq_length, vocab_size]
    assert len(targets.shape) == len(mask.shape) == 2  # [B, max_seq_length]
    assert logits.shape[0] == targets.shape[0] == mask.shape[0]

    B, T, *_ = logits.shape

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')

    assert not torch.any(torch.isnan(loss))

    loss = loss.view(B, T)

    assert loss.shape == mask.shape

    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # note here prompt is less important than completion
    weights = mask.float().masked_fill(mask == -1, cfg.prompt_loss_weight).masked_fill(mask == 1, cfg.completion_loss_weight)
    loss *= weights

    loss = torch.mean(loss)

    return loss


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Tuple[int, float]:
    assert len(logits.shape) == 3  # [B, max_seq_length, vocab_size]
    assert len(targets.shape) == 2  # [B, max_seq_length]
    assert targets.shape == mask.shape  # [B, max_seq_length]
    assert logits.shape[0] == targets.shape[0]

    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # only include completion when compute accuracy
    weights = mask.float().masked_fill(mask == -1, 0)

    # get the index of the max log-probability
    pred = torch.softmax(logits, dim=-1).argmax(dim=-1)

    correct = pred.eq(targets.view_as(pred)).float()

    # only consider completion when compute metrics
    correct *= weights
    num_accurate = correct.sum().item()
    num_samples = weights.bool().sum().item()

    return (num_accurate, num_samples)


def update_step(
    model: GPT2LMHeadModel,
    optimizer: torch.optim.AdamW,
    scheduler: CosineDecayWithWarmupLRScheduler,
    grad_clip: float,
    scaler: ShardedGradScaler = None,
) -> None:
    """Run a single parameter update step"""
    if grad_clip > 0.0:
        if scaler is not None:  # when using float16
            scaler.unscale_(optimizer)  # unscale before clip gradients

        # FSDP needs to use this method to clip gradient norm instead of torch.nn.utils.clip_grad_norm_
        # if 0:
        #    model.clip_grad_norm_(grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if scaler is not None:  # when using float16
        scaler.step(optimizer)
        scaler.update()  # adjust scaling for next batch
    else:
        optimizer.step(set_to_none=True)

    # prepare for next update
    optimizer.zero_grad()
    scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch


@torch.no_grad()
def run_evaluation_steps(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    steps: int,
    local_rank: int,
    tracker: StatsTracker,
) -> None:
    """Run M evaluation steps"""
    model.eval()  # set model in evaluation mode

    inner_pbar = None
    if local_rank == 0:
        inner_pbar = tqdm.tqdm(range(cfg.eval_iters), colour='green', desc='Evaluation steps')

    for i, (x, y, attn_mask, loss_mask) in enumerate(loader):
        x, y, loss_mask = (
            x.to(local_rank, non_blocking=True),
            y.to(local_rank, non_blocking=True),
            loss_mask.to(local_rank, non_blocking=True),
        )
        # use pre-computed attention mask to handle padded sequence,
        # this is very slow (2-3x slower), also requires huge amount of GPU RAM...
        if attn_mask is not None:
            attn_mask = attn_mask.to(local_rank, non_blocking=True)

        output = model(x, attn_mask)

        loss = compute_finetune_loss(output, y, loss_mask)

        num_acc, num_samples = compute_metrics(output.detach(), y.detach())
        tracker.update(loss.detach(), num_acc, num_samples)

        if inner_pbar is not None:
            inner_pbar.update(1)

        if i >= steps:
            break

        if inner_pbar is not None:
            inner_pbar.update(1)

    if inner_pbar is not None:
        inner_pbar.close()


def rank0_logger(msg: str, rank, is_log_on_rank0):
    if is_log_on_rank0:
        if rank == 0:
            print(f'GPU {rank}: ' + msg)
    else:
        print(f'GPU {rank}: ' + msg)


def _collate_fn(batch):
    """
    Custom collate function to pad the sequence to maximum length in the batch,
    which is much faster than pad the sequence to some global max sequence length.

    In addition, it will compute the attention mask and loss mask for the batch.
    """

    batch_size = len(batch)
    if batch_size == 1:  # not need to pad, just process the samples
        prompt, completion = batch[0]
        completion_len = len(completion)
        assert completion_len < cfg.max_seq_length

        sequence = torch.concat((prompt, completion), dim=0)
        seq_length = len(sequence)

        if seq_length > cfg.max_seq_length:
            sequence = sequence[-(cfg.max_seq_length) :]

        x = sequence[:-1]  # [seq_length - 1], exclude the last token for x, so x, y pair have the same length
        y = sequence[1:]  # [seq_length - 1],  shift target one step to the right

        # create a loss mask for the positions for the prompts, completion
        # where -1s are prompt tokens, 1s are completion tokens
        loss_mask = torch.ones_like(y).to(dtype=torch.int)
        loss_mask[:-completion_len] = -1  # prompt tokens

        # add batch dimension
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        loss_mask = loss_mask.unsqueeze(0)

        # the (Scaled Dot Product Attention) SDPA module with casual attention is much faster if we don't pre-compute the mask
        attn_mask = None

        return x, y, attn_mask, loss_mask

    batch_seq_lengths = [len(item[0]) + len(item[1]) for item in batch]

    max_batch_seq_length = max(batch_seq_lengths)

    assert max_batch_seq_length <= cfg.max_seq_length

    # concatenate prompt, completion together
    batch_sequences = torch.full((batch_size, max_batch_seq_length), cfg.pad_id, dtype=torch.long)

    # where -1s are prompt tokens, 1s are completion tokens, and 0s are padding tokens
    loss_mask = torch.full((batch_size, max_batch_seq_length), 0, dtype=torch.long)

    for i, (prompt, completion) in enumerate(batch):
        # need prompt, completion lengths to compute loss mask
        prompt_len, completion_len = len(prompt), len(completion)

        # enforce check sequence length, since trunk sequence is not the ideal solution since we might lost some very important context
        seq_len = prompt_len + completion_len
        assert seq_len <= max_batch_seq_length

        seq = torch.concat((prompt, completion), dim=0)

        # right padding, a simplified example where 0s are pad id: [1, 2, 3] -> [1, 2, 3, 0, 0]
        batch_sequences[i, :seq_len] = seq
        loss_mask[i, :prompt_len] = -1  # prompt tokens
        loss_mask[i, prompt_len : prompt_len + completion_len] = 1  # completion tokens

    x = batch_sequences[:, :-1]  # [batch_size, max_batch_seq_length - 1]
    y = batch_sequences[:, 1:]  # [batch_size, max_batch_seq_length - 1]

    # shift to right to align with y
    loss_mask = loss_mask[:, 1:]

    # create attention mask
    # BUG in SDPA module when use -inf or bool mask will cause NaNs
    # attn_mask = torch.full((batch_size, 1, max_batch_seq_length - 1, max_batch_seq_length - 1), float(-1e10))

    # attn_mask = torch.triu(attn_mask, diagonal=1)
    attn_mask = None
    # for i, seq_len in enumerate(batch_seq_lengths):
    #     # example of what a casual mask with special logic to handle pad ids looks like:
    #     # [0, -inf, -inf, -inf, -inf]
    #     # [0, 0, -inf, -inf, -inf]
    #     # [0, 0, 0, -inf, -inf]
    #     # [-inf, -inf, -inf, -inf, -inf]
    #     attn_mask[i, :, :seq_len, :seq_len] = torch.triu(attn_mask[i, :, :seq_len, :seq_len], diagonal=1)

    return x, y, attn_mask, loss_mask


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if             p.requires_grad)


# def fsdp_main():
def fsdp_main(local_rank, world_size, args):
    assert cfg.train_batch_size >= 1
    assert cfg.gradient_accum_steps >= 1

    if not os.path.exists(cfg.pretrain_ckpt_file):
        raise ValueError(f'Invalid pretrained checkpoint "{cfg.pretrain_ckpt_file}", aborting...')

    rank = local_rank

    setup(rank, world_size)


    logger = functools.partial(rank0_logger, rank=rank, is_log_on_rank0=cfg.is_log_on_rank0)

    # --------------- Load datasets ---------------

    logger('\nLoading datasets ...')

    train_dataset = FineTuneDataset(data_sources=cfg.train_datasources)
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, drop_last=True, shuffle=False)
    cuda_kwargs = {
        'collate_fn': _collate_fn,
        'num_workers': cfg.dataloader_workers,
        'batch_size': cfg.train_batch_size,
        'pin_memory': True,
        # 'shuffle': True, # results in "ValueError: sampler option is mutually exclusive with shuffle"
        'shuffle': False,
        'drop_last': True
    }
    train_kwargs = {'sampler': train_sampler}
    train_kwargs.update(cuda_kwargs)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    logger(f'--> Train dataset metadata:\n{train_dataset.get_metadata()}')

    print(f"rank = {rank} train_dataset sample count = {len(train_dataset)}")
    # print(f"rank = {rank} data_loader batch count = {len(list(train_loader))}")

    # create evaluation dataset on demand
    eval_loader = None
    if cfg.eval_interval > 0:
        eval_dataset = FineTuneDataset(data_sources=cfg.eval_datasources)
        eval_loader = DataLoader(eval_dataset, **cuda_kwargs)
        logger(f'--> Evaluation dataset metadata:\n{eval_dataset.get_metadata()}')

    batch_size = int(cfg.train_batch_size * cfg.gradient_accum_steps)
    steps_per_epoch = len(train_loader) // cfg.gradient_accum_steps
    max_train_steps = steps_per_epoch * cfg.num_epochs

    # --------------- Setup model and optimizer ---------------

    logger('\nInitialize model and FSDP ...')

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    # tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_type)
    model = GPT2LMHeadModel.from_pretrained(cfg.model_type)
    # Both of the below are needed
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    logger(f'--> number of parameters: {get_num_params(model) / 1e6:.2f} million')

    scaler = None
    mixed_precision_policy = None  # defaults to fp32

    if cfg.mixed_precision:
        bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported() and dist.is_nccl_available()
        if bf16_ready:
            logger('--> bFloat16 enabled for mixed precision ...')
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                # Gradient communication precision.
                reduce_dtype=torch.bfloat16,
                # Buffer precision.
                buffer_dtype=torch.bfloat16,
            )
        else:
            logger('--> float16 enabled for mixed precision ...')
            # requires grad scaler
            scaler = ShardedGradScaler()
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                # Gradient communication precision.
                reduce_dtype=torch.float16,
                # Buffer precision.
                buffer_dtype=torch.float16,
            )
    else:
        logger('--> fallback to float32 ...')

    # _auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={EncoderBlock})

    if 0:
        model = FSDP(
            model,
            auto_wrap_policy=_auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            backward_prefetch=cfg.backward_prefetch,
            forward_prefetch=cfg.forward_prefetch,
            cpu_offload=CPUOffload(offload_params=False),
            sharding_strategy=cfg.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=cfg.limit_all_gathers,
            use_orig_params=True,  # need this since we're only use weight decay for some of the params
        )


    if cfg.fsdp_activation_checkpointing:
        logger('--> applying FSDP activation checkpointing ...')
        apply_fsdp_checkpointing(model)

    logger(f'--> FSDP model:\n{model}')

    if cfg.compile_model:
        logger('--> compile model using torch.compile() ...')
        model = torch.compile(model)

    logger('\nInitialize optimizer ...')

    optimizer = AdamW(model.parameters(), lr=cfg.init_lr)
    total_steps = len(train_loader) * cfg.num_epochs/cfg.gradient_accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)


    # --------------- Start Training ---------------

    logger(f'\nStarting to run {cfg.max_train_iters} training iterations ...')

    torch_profiler = None
    # Careful as the logs will grow very fast
    if cfg.use_profiler:
        torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

    tb_writer = None
    csv_file_name = None
    memmax = None
    mem_alloc_tracker = None
    inner_pbar = None
    train_stats = None
    eval_stats = None
    train_steps = 0

    if rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        if cfg.use_tensorboard:
            tb_writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.model_type))
            # get current date and time
            current_datetime = datetime.datetime.now().strftime("%H-%M-%S")
            # convert datetime obj to string
            str_current_datetime = str(current_datetime)

            # create a file object along with extension
            csv_file_name = os.path.join(cfg.log_dir, cfg.model_type, str_current_datetime + ".csv")

        if cfg.track_gpu_mem_usage:
            memmax = Memory_Maximizer()
            mem_alloc_tracker = []
            memmax.start()


    train_tracker = StatsTracker(True, local_rank)
    val_tracker = StatsTracker(True, local_rank)

    start = time.time()
    for epoch in range(1, cfg.num_epochs + 1):  # for each epoch
        # logger.info(f'Start epoch {epoch}')
        logger(f'Start epoch {epoch} on GPU {local_rank}')
        model.train()
        train_tracker.reset()
        val_tracker.reset()
        with tqdm.tqdm(train_loader, unit="batch") as tepoch:
            for iter, batch in enumerate(tepoch):  # for each batch in current epoch
            # Runs the forward pass with autocasting.
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    x, y, attn_mask, loss_mask = batch

                    x, y, loss_mask = (
                        x.to(local_rank, non_blocking=True),
                        y.to(local_rank, non_blocking=True),
                        loss_mask.to(local_rank, non_blocking=True),
                    )
                    output = model(x)
                    logits = output['logits']
                    loss = compute_finetune_loss(logits, y, loss_mask)
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.

                # scale the loss to account for gradient accumulation
                scaled_loss = loss / cfg.gradient_accum_steps

                if scaler is not None:  # when using float16
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                num_acc, num_samples = compute_metrics(logits.detach(), y.detach(), loss_mask.detach())
                train_tracker.update(loss.detach(), num_acc, num_samples)

                if (iter+1) % cfg.gradient_accum_steps == 0:
                    update_step(model, optimizer, scheduler, cfg.grad_clip, scaler)
                    train_steps += 1

                    if torch_profiler is not None:
                        torch_profiler.step()

                    # logging training statistics
                    if train_steps % cfg.log_interval == 0:
                        train_stats = train_tracker.get_dict()
                        train_stats['learning_rate'] = optimizer.param_groups[0]['lr']
                        time_elapsed = time.time() - start
                        num_batches = len(train_loader) # this is adjusted for batch size and number of GPUs in the
                        # world
                        percent_complete = iter/num_batches
                        if percent_complete > 0.9:
                            # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

                            # If we have a distributed model, save only the encapsulated model
                            # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
                            model_to_save = model.module if hasattr(model, 'module') else model

                            # If we save using the predefined names, we can load using `from_pretrained`
                            output_model_file = os.path.join(cfg.ckpt_dir, cfg.saved_model_name)
                            output_config_file = os.path.join(cfg.ckpt_dir, cfg.config_name)

                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)

                        train_stats['percent_complete'] = percent_complete
                        # Note: tb_writer only exists for rank 0 GPU
                        log_statistics(logger, tb_writer, csv_file_name, train_steps, time_elapsed, train_stats, True)
                        train_tracker.reset()

                        if cfg.track_gpu_mem_usage and rank == 0:
                            memmax.update()
                            mem_alloc_tracker.append(format_to_gb(torch.cuda.memory_allocated()))


    if rank == 0:
        # training is done...show some training stats.
        if cfg.track_gpu_mem_usage:
            memmax.stop()
            logger(f'Total memory allocated: {mem_alloc_tracker}')
            logger(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    dist.barrier()
    cleanup()


def log_statistics(logger, tb_writer: SummaryWriter, csv_file_name, train_steps: int, time_elapsed, stats: Dict, is_training: bool) -> None:
    logger(f'Training steps {train_steps}, is status for validation: {not is_training}')
    logger(str(stats))

    if tb_writer is not None:
        tb_tag = 'train' if is_training else 'val'
        for k, v in stats.items():
            # plot percent_complete against time_elapsed, everything else against train steps
            if k == 'percent_complete':
                tb_writer.add_scalar(f'{tb_tag}/{k}', v, time_elapsed)
            else:
                tb_writer.add_scalar(f'{tb_tag}/{k}', v, train_steps)
    # write percent_complete against time elapsed to a csv file
        with open(csv_file_name, 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([time_elapsed, stats.get('percent_complete')])


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    # torch.backends.cuda.enable_flash_sdp(True)
    # torch.backends.cuda.enable_mem_efficient_sdp(True)
    # torch.backends.cuda.enable_math_sdp(True)

    # fsdp_main()
    WORLD_SIZE = 2
    mp.spawn(fsdp_main,
             args=(WORLD_SIZE, cfg),
             nprocs=WORLD_SIZE,
             join=True)
