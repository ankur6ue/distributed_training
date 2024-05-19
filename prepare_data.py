
import argparse
import collections
import json
import logging
import math
import os
import sys
from io import open

from multiprocessing import Pool, Queue
import functools
from tqdm import tqdm, trange
import time
import multiprocessing as mp
import multiprocessing as mp
from squad_example import SquadExample
from input_features import InputFeatures
from tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

pid = -1
tokenizer = None
shared_cnt = None
features = []


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def create_squad_example(version_2_with_negative, is_training, entry_):

    idx, entry = entry_
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    examples = []

    for paragraph in entry["paragraphs"]:
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        for qa in paragraph["qas"]:
            qas_id = qa["id"]
            question_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible = False
            if is_training:
                if version_2_with_negative:
                    is_impossible = qa["is_impossible"]
                if (len(qa["answers"]) != 1) and (not is_impossible):
                    raise ValueError(
                        "For training, each question should have exactly 1 answer.")
                if not is_impossible:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning("Could not find answer: '%s' vs. '%s'",
                                       actual_text, cleaned_answer_text)
                        continue
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""

            example = SquadExample(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                is_impossible=is_impossible)
            examples.append(example)
    return examples, os.getgid()


def convert_examples_to_features_impl(features, max_seq_length,
                                 doc_stride, max_query_length, is_training, example_idx, example):

    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None
        if is_training and not example.is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        if is_training and example.is_impossible:
            start_position = 0
            end_position = 0
        # increment count in a process-safe way
        global shared_cnt
        local_val = 0
        with shared_cnt.get_lock():
            local_val = shared_cnt.value
            shared_cnt.value += 1

        features.append(InputFeatures(
            unique_id=1000000000 + local_val,  # unique across all features
            example_index=example_idx,  # example index
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            is_impossible=example.is_impossible))
    return features


def convert_examples_to_features(max_seq_length,
                                 doc_stride, max_query_length, is_training, do_merge, example_):
    """Loads a data file into a list of `InputBatch`s."""
    example_idx, example = example_
    if do_merge:
        _features = []
        return convert_examples_to_features_impl(_features, max_seq_length,
                                                 doc_stride, max_query_length, is_training, example_idx, example)
    else:
        global features # global declaration not really required because without explicit global declaration,
        # features will automatically refer to the global features variable
        # no need to return the features here, because each process will modify its own features global variable
        convert_examples_to_features_impl(features, max_seq_length,
                                          doc_stride, max_query_length, is_training, example_idx, example)


def save(args):
    '''
        :param tuple containing a directory name and process index. Saves features to directory/features{i}.pkl
        :return: None
        '''
    pid, dirname = args
    output_file = os.path.join(dirname, f"features{pid}.pkl")
    global features
    with open(output_file, "wb") as writer:
        pickle.dump(features, writer)


def stream(input_data):
    '''
    :param iterable to iterate through
    :return: a single item from the input
    '''
    for idx, entry in enumerate(tqdm(input_data)):
        yield idx, entry


def worker_init(q, shared_cnt_):
    global pid
    global shared_cnt
    pid = q.get()
    shared_cnt = shared_cnt_
    # print(f"process id: {pid}")


def squad2features(args):
    input_file = args.input_file
    version_2_with_negative = args.version_2_with_negative
    is_training = args.do_train
    cpus = mp.cpu_count()
    # ignore user provided num_cpus on a low CPU system to avoid crashes/system slowdowns
    if cpus < 3:
        nproc = 1
    else: # don't hog all the cpus on the system
        nproc = min(args.nproc, cpus - 5)

    # output directory and filename for the features file
    output_dir = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    feature_file_name = os.path.join(output_dir, input_filename + '_features_' +
                                     '{0}_{1}_{2}'.format(str(args.max_seq_length), str(args.doc_stride),
                                                          str(args.max_query_length)) + '.pkl')
    squad_file_name = os.path.join(output_dir, input_filename + '_squad_' +
                                     '{0}_{1}_{2}'.format(str(args.max_seq_length), str(args.doc_stride),
                                                          str(args.max_query_length)) + '.pkl')

    """Read a SQuAD json file into a list of SquadExample."""
    examples = []
    # create a shared process-safe count variable
    shared_cnt = mp.Value('i', 0)

    # arguments that don't change for various processes can be included in a partial function
    create_squad_example_partial = functools.partial(create_squad_example, version_2_with_negative, is_training)
    # mechanism to give each process an id starting from 0 to nproc. We put numbers from 0-9 on a queue.
    # Worker processes read from this queue and use each number as their id
    q = Queue(nproc)
    for i in range(nproc):
        q.put(i)
    with Pool(nproc, initializer=worker_init, initargs=(q, shared_cnt)) as pool:
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
            for example, pid in pool.imap(create_squad_example_partial, stream(input_data)):
                if example is None:
                    continue
                else:
                    examples.extend(example)
        # save squad examples
        with open(squad_file_name, "wb") as writer:
            pickle.dump(examples, writer)
        # Now convert each squad example into "features" which is training data for BERT. We tokenize each example, add
        # the special tokens to separate context from question, add padding tokens to create spans of length = BERT's
        # context length, mapping between words and tokens etc. Each example can be processed in parallel, so this is
        # an embarrassingly parallel use-case, perfect for applying multi-proceesing techniques.

        unique_id = 1000000000 # Incremented and added as a property of each feature vector
        cnt = 0
        # See below for an explanation
        do_merge = args.do_merge

        convert_e2f_partial = functools.partial(convert_examples_to_features, args.max_seq_length,
                                                args.doc_stride,
                                                args.max_query_length,
                                                is_training, do_merge)

        features = []
        start = time.time()
        # As before, our approach to parallelize feature creation is to stream examples to the worker process pool.
        # Each process receives a example, and converts it into a list of features. There are two options from hereon:
        # Option 1 (do_merge=True): Each worker process passes the feature list back to the main process, where
        # the features can be concatenated into a master list of features. The downside of this approach is that
        # the features created by each process must be passed to the main process (using a process-safe Queue created
        # automatically by the map functions). Because this is a multiple producer, single consumer system with
        # multiple worker processes writing to the queue and single main process consuming from the queue, this can
        # cause a slowdown if the queue is full. This leads us to option #2

        # Option 2 (do_merge=False): To avoid the slowdown mentioned above, each worker process writes the features
        # it generates to its local (to each procesS) list of features (which is declared as a global variable, and
        # because each process gets its own copy of global variables, is different for each process).
        # Once all the examples are processed into features, we call a save function on each process, that writes
        # the features generated by that process to a file on disk.
        # With this option, we need to create a common count object that will be shared across all processes to
        # to increment the feature count.

        if do_merge: # option 1
            # returns a list of features
            for features_ in pool.imap(convert_e2f_partial, stream(examples)):
                if features_ is None:
                    continue
                else:
                    for feature in features_:
                    # no need to do this, because counts are incremented using shared variables
                    #     feature.example_index = cnt
                    #     feature.unique_id = unique_id
                         features.append(feature)
                    #     unique_id = unique_id + 1
                    #     cnt = cnt + 1
            with open(feature_file_name, "wb") as writer:
                pickle.dump(features, writer)

        else: # Option 2
            ret = pool.imap(convert_e2f_partial, stream(examples))
            list(ret)  # imap executes asynchronously, so dereference the return value to wait for imap execution
            # to finish
            # Each process maintains its own list of features. Now save to separate files. Pass the file index to
            # each process. Note each process could also use the index we had provided earlier during pool creation
            # but pass the indices as an argument to show a different way of achieving the same outcome
            pool.map(save, [(i, output_dir) for i in range(nproc)])
            # To do a fair comparison with Option 1, we must read back the files each process wrote and consolidate
            # the results
            for i in range(0, nproc):  # we'll have as many files as nproc
                output_file = os.path.join(output_dir, f"features{i}.pkl")
                with open(output_file, "rb") as reader:
                    features_ = pickle.load(reader)
                    features.extend(features_)
                    os.remove(output_file)
                    # remove temporary file

            # write the consolidated features back to disk
            with open(feature_file_name, "wb") as writer:
                pickle.dump(features, writer)
        time_elapsed = time.time() - start
        print(f"time to generate features using {nproc} processes is {time_elapsed:.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument('--vocab_file',
                        type=str, default=None, required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--do_merge',
                        action='store_true',
                        help='If true, features produced by each process are concatenated into one list in the \
                        main process and saved to a file. If false, each process maintains its own list of features \
                        and those features are saved to separate files. The intuition behind this is to avoid sending \
                        large data back from worker processes to the main process which could prevent linear scaling')
    parser.add_argument('--nproc', type=int, default=1, required=False, help='number of worker processes')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    cfg = parser.parse_args()

    tokenizer = BertTokenizer(cfg.vocab_file, do_lower_case=cfg.do_lower_case, max_len=512)  # for bert large
    squad2features(cfg)
    print('done')
