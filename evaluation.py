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
import random
import sys
from io import open
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from squad_example import SquadExample
from input_features import InputFeatures
from tqdm import tqdm, trange
import modeling
from tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)
from utils import is_main_process, format_step
import dllogger, time

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


def get_answers(examples, features, results, args):
    predictions = collections.defaultdict(list)  # it is possible that one example corresponds to multiple features
    Prediction = collections.namedtuple('Prediction', ['text', 'start_logit', 'end_logit'])

    if args.version_2_with_negative:
        null_vals = collections.defaultdict(lambda: (float("inf"), 0, 0))
    for ex, feat, result in match_results(examples, features, results):
        start_indices = _get_best_indices(result.start_logits, args.n_best_size)
        end_indices = _get_best_indices(result.end_logits, args.n_best_size)
        prelim_predictions = get_valid_prelim_predictions(start_indices, end_indices, feat, result, args)
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)
        if args.version_2_with_negative:
            score = result.start_logits[0] + result.end_logits[0]
            if score < null_vals[ex.qas_id][0]:
                null_vals[ex.qas_id] = (score, result.start_logits[0], result.end_logits[0])

        curr_predictions = []
        seen_predictions = []
        for pred in prelim_predictions:
            if len(curr_predictions) == args.n_best_size:
                break
            if pred.start_index > 0:  # this is a non-null prediction TODO: this probably is irrelevant
                final_text = get_answer_text(ex, feat, pred, args)
                if final_text in seen_predictions:
                    continue
            else:
                final_text = ""

            seen_predictions.append(final_text)
            curr_predictions.append(Prediction(final_text, pred.start_logit, pred.end_logit))
        predictions[ex.qas_id] += curr_predictions

    # Add empty prediction
    if args.version_2_with_negative:
        for qas_id in predictions.keys():
            predictions[qas_id].append(Prediction('',
                                                  null_vals[ex.qas_id][1],
                                                  null_vals[ex.qas_id][2]))

    nbest_answers = collections.defaultdict(list)
    answers = {}
    for qas_id, preds in predictions.items():
        nbest = sorted(
            preds,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)[:args.n_best_size]

        # In very rare edge cases we could only have single null prediction.
        # So we just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(Prediction(text="empty", start_logit=0.0, end_logit=0.0))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry and entry.text:
                best_non_null_entry = entry
        probs = _compute_softmax(total_scores)
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_answers[qas_id].append(output)
        if args.version_2_with_negative:
            if best_non_null_entry: # this can be null, not sure why..
                score_diff = null_vals[qas_id][0] - best_non_null_entry.start_logit - best_non_null_entry.end_logit
                if score_diff > args.null_score_diff_threshold:
                    answers[qas_id] = ""
                else:
                    answers[qas_id] = best_non_null_entry.text
            else:
                answers[qas_id] = ""
        else:
            answers[qas_id] = nbest_answers[qas_id][0]['text']

    return answers, nbest_answers


def get_answer_text(example, feature, pred, args):
    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
    orig_doc_start = feature.token_to_orig_map[pred.start_index]
    orig_doc_end = feature.token_to_orig_map[pred.end_index]
    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    orig_text = " ".join(orig_tokens)

    final_text = get_final_text(tok_text, orig_text, args.do_lower_case, args.verbose_logging)
    return final_text


def get_valid_prelim_predictions(start_indices, end_indices, feature, result, args):
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["start_index", "end_index", "start_logit", "end_logit"])
    prelim_predictions = []
    for start_index in start_indices:
        for end_index in end_indices:
            if start_index >= len(feature.tokens):
                continue
            if end_index >= len(feature.tokens):
                continue
            if start_index not in feature.token_to_orig_map:
                continue
            if end_index not in feature.token_to_orig_map:
                continue
            if not feature.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > args.max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index]))
    return prelim_predictions


def match_results(examples, features, results):
    unique_f_ids = set([f.unique_id for f in features])
    unique_r_ids = set([r.unique_id for r in results])
    matching_ids = unique_f_ids & unique_r_ids
    features = [f for f in features if f.unique_id in matching_ids]
    results = [r for r in results if r.unique_id in matching_ids]
    features.sort(key=lambda x: x.unique_id)
    results.sort(key=lambda x: x.unique_id)

    for f, r in zip(features, results):  # original code assumes strict ordering of examples. TODO: rewrite this
        yield examples[f.example_index], f, r


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indices(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indices = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indices.append(index_and_score[i][0])
    return best_indices


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
    if args.no_cuda:
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dllogger.log(step="PARAMETER", data={"SEED": args.seed})

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512)  # for bert large

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = modeling.BertForQuestionAnswering(config)
    dllogger.log(step="PARAMETER", data={"loading_checkpoint": True})
    checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
    checkpoint = checkpoint["model"] if "model" in checkpoint.keys() else checkpoint
    model.load_state_dict(checkpoint, strict=False)
    dllogger.log(step="PARAMETER", data={"loaded_checkpoint": True})
    model.to(device)
    num_weights = sum([p.numel() for p in model.parameters() if p.requires_grad])
    dllogger.log(step="PARAMETER", data={"model_weights_num": num_weights})

    if args.do_predict and is_main_process():

        if args.fp16:
            model.half()

        output_dir = os.path.dirname(args.predict_file)
        input_filename = os.path.basename(args.predict_file)
        squad_file_name = os.path.join(output_dir, input_filename + '_squad_' +
                                       '{0}_{1}_{2}'.format(str(args.max_seq_length), str(args.doc_stride),
                                                            str(args.max_query_length)) + '.pkl')
        feature_file_name = os.path.join(output_dir, input_filename + '_features_' +
                                         '{0}_{1}_{2}'.format(str(args.max_seq_length), str(args.doc_stride),
                                                              str(args.max_query_length)) + '.pkl')
        # These files must exist.
        with open(squad_file_name, "rb") as reader:
            eval_examples = pickle.load(reader)
        with open(feature_file_name, "rb") as reader:
            eval_features = pickle.load(reader)
        if eval_examples is None or eval_features is None:
            print(f"{squad_file_name} and/or {feature_file_name} are not found, exiting")
            exit(0)

        dllogger.log(step="PARAMETER", data={"infer_start": True})
        dllogger.log(step="PARAMETER", data={"eval_samples": len(eval_examples)})
        dllogger.log(step="PARAMETER", data={"eval_features": len(eval_features)})
        dllogger.log(step="PARAMETER", data={"predict_batch_size": args.predict_batch_size})

        # can use this to lower the number of samples we are predicting for to run the evaluation faster
        max_num_samples = (int)(len(eval_features)/3)
        eval_features = eval_features[0:max_num_samples]
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        infer_start = time.time()
        model.eval()
        all_results = []
        dllogger.log(step="PARAMETER", data={"eval_start": True})
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
            if len(all_results) % 1000 == 0:
                dllogger.log(step="PARAMETER", data={"sample_number": len(all_results)})
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))

        time_to_infer = time.time() - infer_start
        output_prediction_file = os.path.join(output_dir, "predictions.json")
        output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")

        answers, nbest_answers = get_answers(eval_examples, eval_features, all_results, args)
        with open(output_prediction_file, "w") as f:
            f.write(json.dumps(answers, indent=4) + "\n")
        with open(output_nbest_file, "w") as f:
            f.write(json.dumps(nbest_answers, indent=4) + "\n")

        # output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
        # write_predictions(eval_examples, eval_features, all_results,
        #                   args.n_best_size, args.max_answer_length,
        #                   args.do_lower_case, output_prediction_file,
        #                   output_nbest_file, output_null_log_odds_file, args.verbose_logging,
        #                   args.version_2_with_negative, args.null_score_diff_threshold)

        if args.do_eval and is_main_process():
            import sys
            import subprocess
            eval_out = subprocess.check_output([sys.executable, args.eval_script,
                                                args.predict_file, output_dir + "/predictions.json"])
            scores = str(eval_out).strip()
            exact_match = float(scores.split(":")[1].split(",")[0])
            f1 = float(scores.split(":")[2].split("}")[0])


    if args.do_predict and is_main_process():
        dllogger.log(step=tuple(), data={"e2e_inference_time": time_to_infer,
                                         "inference_sequences_per_second": len(eval_features) / time_to_infer})
    if args.do_eval and is_main_process():
        dllogger.log(step=tuple(), data={"exact_match": exact_match, "F1": f1})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
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

    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
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
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
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


    cfg = parser.parse_args()
    WORLD_SIZE = 1
    mp.spawn(fsdp_main,
             args=(WORLD_SIZE, cfg),
             nprocs=WORLD_SIZE,
             join=True)