# coding=utf-8
# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import pickle
import random
import sys
from io import open
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.cuda.amp import GradScaler

from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import csv
from squad_example import SquadExample
from input_features import InputFeatures
from torch.utils.tensorboard import SummaryWriter
from schedulers import LinearWarmUpScheduler
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import datetime
import modeling
from optimization import BertAdam, warmup_linear
from tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)
from utils import is_main_process, format_step
import dllogger, time
from torch.nn.parallel import DistributedDataParallel as DDP
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def fsdp_main(local_rank, world_size, args):

    rank = local_rank
    if local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        n_gpu = 1

    if is_main_process():
        Path(os.path.dirname(args.json_summary)).mkdir(parents=True, exist_ok=True)
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])

    dllogger.metadata("e2e_train_time", {"unit": "s"})
    dllogger.metadata("training_sequences_per_second", {"unit": "sequences/s"})
    dllogger.metadata("final_loss", {"unit": None})
    dllogger.metadata("e2e_inference_time", {"unit": "s"})
    dllogger.metadata("inference_sequences_per_second", {"unit": "sequences/s"})
    dllogger.metadata("exact_match", {"unit": None})
    dllogger.metadata("F1", {"unit": None})

    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.world_size > 1), args.fp16))

    dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dllogger.log(step="PARAMETER", data={"SEED": args.seed})

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and os.listdir(
            args.output_dir) != ['logfile.txt']:
        print("WARNING: Output directory {} already exists and is not empty.".format(args.output_dir),
              os.listdir(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512)  # for bert large
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    # Get the paths for the files containing squad examples and features that we already created
    base_dir = os.path.dirname(args.train_file)
    input_filename = os.path.basename(args.train_file)
    squad_file_name = os.path.join(base_dir, input_filename + '_squad_' +
                                   '{0}_{1}_{2}'.format(str(args.max_seq_length), str(args.doc_stride),
                                                        str(args.max_query_length)) + '.pkl')
    feature_file_name = os.path.join(base_dir, input_filename + '_features_' +
                                     '{0}_{1}_{2}'.format(str(args.max_seq_length), str(args.doc_stride),
                                                          str(args.max_query_length)) + '.pkl')
    if args.do_train:
        check_file1 = os.path.isfile(squad_file_name)
        check_file2 = os.path.isfile(feature_file_name)
        train_features = None
        check_file = check_file1 and check_file2


        if check_file:
            with open(squad_file_name, 'rb') as f:
                train_examples = pickle.load(f)
            with open(feature_file_name, "rb") as reader:
                train_features = pickle.load(reader)
        else:
            return

        # original NVIDIA code calculates the num_train_optimization_steps using the number of squadexamples, which
        # is incorrect I think, because the dataloader is created using train_features, which can be different from the
        # number of squad examples
        # num_train_optimization_steps = int(
        #    len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare model
        config = modeling.BertConfig.from_json_file(args.config_file)
        # Padding for divisibility by 8
        if config.vocab_size % 8 != 0:
            config.vocab_size += 8 - (config.vocab_size % 8)

        model = modeling.BertForQuestionAnswering(config)
        # model = modeling.BertForQuestionAnswering.from_pretrained(args.bert_model,
        # cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))
        dllogger.log(step="PARAMETER", data={"loading_checkpoint": True})
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
        checkpoint = checkpoint["model"] if "model" in checkpoint.keys() else checkpoint
        model.load_state_dict(checkpoint, strict=False)
        dllogger.log(step="PARAMETER", data={"loaded_checkpoint": True})
        model.to(device)
        num_weights = sum([p.numel() for p in model.parameters() if p.requires_grad])
        dllogger.log(step="PARAMETER", data={"model_weights_num": num_weights})

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        scaler = None
        if args.fp16:
            scaler = GradScaler()

        if world_size > 1:
            model = DDP(model, find_unused_parameters=True)

        global_step = 0

        dllogger.log(step="PARAMETER", data={"train_start": True})
        dllogger.log(step="PARAMETER", data={"training_samples": len(train_examples)})
        dllogger.log(step="PARAMETER", data={"training_features": len(train_features)})
        dllogger.log(step="PARAMETER", data={"train_batch_size": args.train_batch_size})
        dllogger.log(step="PARAMETER", data={"steps": num_train_optimization_steps})
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        if world_size == 1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data, rank=rank, num_replicas=world_size, drop_last=True, shuffle=False)
        # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size * n_gpu)
        # don't think we should be scaling the batch size by number of GPUs
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)
        print(f"rank = {rank} train_dataset sample count = {len(train_data)}")
        print(f"rank = {rank} num_batches = {len(train_dataloader)}")

        # tf summary
        tb_writer = None
        if rank == 0:
            os.makedirs(args.log_dir, exist_ok=True)

            if args.use_tensorboard:
                tb_writer = SummaryWriter(os.path.join(args.log_dir, args.model_type))
                # get current date and time
                current_datetime = datetime.datetime.now().strftime("%H-%M-%S")
                # convert datetime obj to string
                str_current_datetime = str(current_datetime)

                # create a file object along with extension
                csv_file_name = os.path.join(args.log_dir, args.model_type, str_current_datetime + ".csv")

        model.train()
        # gradClipper = GradientClipper(max_grad_norm=1.0)
        final_loss = None
        # this is adjusted for batch size and number of GPUs in the
        # world
        num_batches = len(train_dataloader)
        train_start = time.time()
        for epoch in range(int(args.num_train_epochs)):
            train_iter = tqdm(train_dataloader, desc="Iteration",
                              disable=args.disable_progress_bar) if is_main_process() else train_dataloader
            for step, batch in enumerate(train_iter):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # Terminate early for benchmarking

                    if args.max_steps > 0 and global_step > args.max_steps:
                        break

                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    start_logits, end_logits = model(input_ids, segment_ids, input_mask)
                    # If we are on multi-GPU, split add a dimension
                    if len(start_positions.size()) > 1:
                        start_positions = start_positions.squeeze(-1)
                    if len(end_positions.size()) > 1:
                        end_positions = end_positions.squeeze(-1)
                    # sometimes the start/end positions are outside our model inputs, we ignore these terms
                    ignored_index = start_logits.size(1)
                    start_positions.clamp_(0, ignored_index)
                    end_positions.clamp_(0, ignored_index)

                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                    loss = (start_loss + end_loss) / 2

                if args.gradient_accumulation_steps > 1: # This division should happen outside auto_cast
                    loss = loss / args.gradient_accumulation_steps
                if scaler is not None:  # when using float16
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # gradient clipping
                # gradClipper.step(amp.master_params(optimizer))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # unscale the gradients
                    if scaler is not None:  # when using float16
                        scaler.unscale_(optimizer)
                        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                        scaler.step(optimizer) # will also step optimizer
                        scaler.update()  # adjust scaling for next batch
                    else:
                        optimizer.step() # step the optimizer manually
                    # must zero gradients for the next step
                    optimizer.zero_grad()
                    global_step += 1

                final_loss = loss.item()
                if rank == 0 and global_step % args.log_interval == 0:
                    train_stats = {}
                    train_stats['learning_rate'] = optimizer.get_lr()[0]
                    time_elapsed = time.time() - train_start
                    percent_complete = step / num_batches
                    train_stats['percent_complete'] = percent_complete
                    train_stats['loss'] = final_loss
                    log_statistics(tb_writer, csv_file_name, global_step, time_elapsed, train_stats, True)

                if step % args.log_freq == 0:
                    dllogger.log(step=(epoch, global_step,), data={"step_loss": final_loss,
                                                                   "learning_rate": optimizer.param_groups[0]['lr']})
        time_to_train = time.time() - train_start

    if args.do_train and is_main_process() and not args.skip_checkpoint:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, modeling.WEIGHTS_NAME)
        torch.save({"model": model_to_save.state_dict()}, output_model_file)
        output_config_file = os.path.join(args.output_dir, modeling.CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())


def log_statistics(tb_writer: SummaryWriter, csv_file_name, train_steps: int, time_elapsed, stats: Dict, is_training: bool) -> None:
    # logger(f'Training steps {train_steps}, is status for validation: {not is_training}')
    # logger(str(stats))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1.0, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="world_size (usually number of GPUs) for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--vocab_file',
                        type=str, default=None, required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument('--log_freq',
                        type=int, default=50,
                        help='frequency of logging loss.')
    parser.add_argument('--json-summary', type=str, default="results/dllogger.json",
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument("--eval_script",
                        help="Script to evaluate squad predictions",
                        default="evaluate.py",
                        type=str)
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to use evaluate accuracy of predictions")
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--disable-progress-bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument("--skip_cache",
                        default=False,
                        action='store_true',
                        help="Whether to cache train features")
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        help="Location to cache train feaures. Will default to the dataset directory")
    parser.add_argument("--profile",
                        default=False,
                        action='store_true',
                        help="Whether to profile model.")
    parser.add_argument("--log_dir",
                        default='results',
                        help="Directory for storing training summary.")
    parser.add_argument("--log_interval", type=int,
                        default=5,
                        help="training iteration stride when logs are written")
    parser.add_argument("--model_type",
                        default='bert_small',
                        help="model type")
    parser.add_argument("--use_tensorboard",
                        default=False,
                        help="whether to log or not")

    cfg = parser.parse_args()
    mp.spawn(fsdp_main,
             args=(cfg.world_size, cfg),
             nprocs=cfg.world_size,
             join=True)
    dllogger.flush()
