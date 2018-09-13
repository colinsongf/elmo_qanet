import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open
import sys, os, time
import tensorflow as tf
import os
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''
datadir = "/data/elmo_experiment_20180906/20180906_model"
vocab_file = os.path.join(datadir, 'vocab-2016-09-10.txt')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'weights.hdf5')
nlp = spacy.blank("en")

def convert_to_features(config, data, word2idx_dict, char2idx_dict):

    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = config.test_para_limit
    ques_limit = config.test_ques_limit
    ans_limit = 100
    char_limit = config.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
    y1 = np.zeros([para_limit], dtype=np.float32)
    y2 = np.zeros([para_limit], dtype=np.float32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


# def process_file(filename, data_type, word_counter, char_counter):
def process_file(config, filename, data_type, word_counter, char_counter, is_test=False):

    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit

    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace("''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)

                # discard context tokens longer than para_limit
                if len(context_tokens) > para_limit:
                    context_tokens = context_tokens[:para_limit]

                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                max_span = spans[-1][1]

                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    ques = qa["question"].replace("''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)

                    # discard query tokens longer than ques_limit
                    if len(ques_tokens) > ques_limit: ques_tokens = ques_tokens[:ques_limit]

                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1

                    y1s, y2s, answer_texts = [], [], []
                    candidates, answer_position, cand_chars= None, None, None
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        # answer_text = ' '.join(word_tokenize(answer_text))
                        answer_start = answer["answer_start"]
                        answer_end = answer_start + len(answer_text)

                        candidates = answer["candidates"]
                        candidates = [' '.join(word_tokenize(x)) for x in candidates]
                        cand_chars = [list(token) for token in candidates]
                        answer_position = answer["answer_position"]
                        answer_texts.append(answer_text)

                        # if answers not within #para_limit words, y1 and y2 should be empty, however it does not
                        # matter if simply selecting answer from the candidate list
                        if answer_start <= max_span and answer_end <= max_span:
                            answer_span = []
                            for idx, span in enumerate(spans):
                                if not (answer_end <= span[0] or answer_start >= span[1]):
                                    answer_span.append(idx)
                            y1, y2 = answer_span[0], answer_span[-1]
                            y1s.append(y1)
                            y2s.append(y2)

                    # include sample only if answer exists
                    # if answer_texts and y1s and y2s:
                    # total += 1 # only increment counter if example is included
                    total += 1
                    example = {"context_tokens": context_tokens, "context_chars": context_chars,
                               "ques_tokens": ques_tokens,"ques_chars": ques_chars, "y1s": y1s, "y2s": y2s,
                               "candidates": candidates, "cand_chars": cand_chars,
                               "answer_position": answer_position, "id": total}
                    examples.append(example)

                    # context is still complete but spans and context_tokens etc are clipped
                    eval_examples[str(total)] = {"context": context, "spans": spans, "candidates": candidates,
                                             "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))

    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have {} embeddings".format(len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
        print("{} tokens have embedding".format(len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)} # start the index from 2
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    '''
        embedding_dict = {token:emb}
        token2idx_dict = {token:idx}
        idx2emb_dict   = {idx:emb}
        emb_mat = [emb] 
    '''
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict

def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):

    """
        Converting tokens to np.arrays of indices
    """
    def get_cand_position(candidates, context_tokens):
        cand_positions = np.zeros([cand_limit, para_limit], dtype=np.float32)

        context = ' '.join(context_tokens)
        for i, token in enumerate(candidates):
            token = token+' '
            char_start = context.find(token)
            if char_start > -1:
                l = len(token.split())
                pretext = context[:char_start].strip()
                token_start = len(pretext.split())
                for j in range(token_start, token_start+l):
                    if j < para_limit:
                        cand_positions[i][j] = 1.0
        # DEBUG
        # for i, (_, c) in enumerate(zip(cand_positions, candidates)):
        #     print c
        #     for j, t in enumerate(cand_positions[i]):
        #         if cand_positions[i][j] > 0:
        #             print context_tokens[j]

        return cand_positions

    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    cand_limit = config.cand_limit

    #TODO: change hard-coded number to config parameter
    ans_limit = 100 if is_test else config.ans_limit
    char_limit = config.char_limit

    # def filter_func(example, is_test=False):
    #     return len(example["context_tokens"]) > para_limit or \
    #            len(example["ques_tokens"]) > ques_limit or \
    #            (example["y2s"][0] - example["y1s"][0]) > ans_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    batcher = Batcher(vocab_file, 50)
    for example in tqdm(examples):
        '''
            example = {"context_tokens": context_tokens, "context_chars": context_chars,
                                   "ques_tokens": ques_tokens,"ques_chars": ques_chars,
                                   "y1s": y1s, "y2s": y2s, "candidates": candidates,
                                   "answer_position": answer_position, "id": total}
        '''
        total_ += 1

        # TODO: comment this out as we do not need filtering
        # if filter_func(example, is_test): continue

        total += 1

        line_zeros = np.zeros([50], dtype=np.int32)
        #construct context ids list about elmo       
        new_context_tokens = []
        new_list_list_context_tokens = []        
        new_context_elmo_ids = []
        if len(example["context_tokens"]) > para_limit:
            new_context_tokens = example["context_tokens"][:(para_limit)]
        #if len(example["context_tokens"]) > para_limit - 2:
        #    new_context_tokens = example["context_tokens"][:(para_limit - 2)]
            new_list_list_context_tokens.append(new_context_tokens)
            context_elmo_ids = batcher.batch_sentences(new_list_list_context_tokens)
            new_context_elmo_ids = context_elmo_ids[0]
        else:
            new_context_tokens = example["context_tokens"]
            new_list_list_context_tokens.append(new_context_tokens)
            context_elmo_ids = batcher.batch_sentences(new_list_list_context_tokens)
            new_context_elmo_ids = context_elmo_ids[0]
            remain_length = para_limit - len(example["context_tokens"])
            #remain_length = para_limit - len(example["context_tokens"])
            for i in range(remain_length):
                new_context_elmo_ids = np.row_stack((new_context_elmo_ids, line_zeros))
                
        #construct questions ids list about elmo
        new_ques_tokens = []
        new_list_list_ques_tokens = []        
        new_question_elmo_ids = []
        if len(example["ques_tokens"]) >= ques_limit:
            new_ques_tokens = example["ques_tokens"][:(ques_limit)]
        #if len(example["ques_tokens"]) >= ques_limit - 2:
        #    new_ques_tokens = example["ques_tokens"][:(ques_limit - 2)]
            new_list_list_ques_tokens.append(new_ques_tokens)
            question_elmo_ids = batcher.batch_sentences(new_list_list_ques_tokens)
            new_question_elmo_ids = question_elmo_ids[0]
        else:
            new_ques_tokens = example["ques_tokens"]
            new_list_list_ques_tokens.append(new_ques_tokens)
            question_elmo_ids = batcher.batch_sentences(new_list_list_ques_tokens)
            new_question_elmo_ids = question_elmo_ids[0]
            remain_length = ques_limit - len(example["ques_tokens"])
            #remain_length = ques_limit - len(example["ques_tokens"])
            for i in range(remain_length):
                new_question_elmo_ids = np.row_stack((new_question_elmo_ids, line_zeros))
        
        #construct candidates ids list about elmo     
        new_candidate_tokens = []
        new_list_list_candidate_tokens = []   
        new_candidate_elmo_ids = []
        if len(example["candidates"]) > cand_limit:
            new_candidate_tokens = example["candidates"][:(cand_limit)]
        #if len(example["candidates"]) > cand_limit - 2:
        #    new_candidate_tokens = example["candidates"][:(cand_limit - 2)]
            new_list_list_candidate_tokens.append(new_candidate_tokens)
            candidate_elmo_ids = batcher.batch_sentences(new_list_list_candidate_tokens)
            new_candidate_elmo_ids = candidate_elmo_ids[0]
        else:
            new_candidate_tokens = example["candidates"]
            new_list_list_candidate_tokens.append(new_candidate_tokens)
            candidate_elmo_ids = batcher.batch_sentences(new_list_list_candidate_tokens)
            new_candidate_elmo_ids = candidate_elmo_ids[0]
            remain_length = cand_limit - len(example["candidates"])
            #remain_length = cand_limit - len(example["candidates"])
            for i in range(remain_length):
                new_candidate_elmo_ids = np.row_stack((new_candidate_elmo_ids, line_zeros))
        

        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        cand_idxs = np.zeros([cand_limit], dtype=np.int32)
        cand_char_idxs = np.zeros([cand_limit, char_limit], dtype=np.int32)
        cand_label = np.zeros([cand_limit], dtype=np.float32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word_idx(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1 # OOV

        def _get_char_idx(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1 # OOV

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word_idx(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word_idx(token)

        for i, token in enumerate(example["candidates"]):
            cand_idxs[i] = _get_word_idx(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break # discard chars beyond limit
                context_char_idxs[i, j] = _get_char_idx(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char_idx(char)

        for i, token in enumerate(example["cand_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                cand_char_idxs[i, j] = _get_char_idx(char)

        if len(example["y1s"]) > 0:
            start = example["y1s"][-1]
            y1[start] = 1.0

        if len(example["y2s"]) > 0:
            end = example["y2s"][-1]
            y2[end] = 1.0

        cand_label[example["answer_position"]] = 1.0
        cand_positions = get_cand_position(example["candidates"], example["context_tokens"])

        '''
            tf.train.Example is not a Python class, but a protocol buffer for structuring a TFRecord. 
                An tf.train.Example stores features in a single attribute features of type tf.train.Features.

            tf.train.Features is a collection of named features.
            
            tf.train.Feature wraps a list of data of a specific type: tf.train.BytesList (attribute name bytes_list),
                tf.train.FloatList (attribute name float_list), or tf.train.Int64List (attribute name int64_list).
            
            tf.python_io.TFRecordWriter.write() accepts a string as parameter and writes it to disk, 
                meaning that structured data must be serialized first --> tf.train.Example.SerializeToString()
        '''

        record = tf.train.Example(features=tf.train.Features(feature={
            "context_elmo_idxs":      tf.train.Feature(bytes_list=tf.train.BytesList(value=[new_context_elmo_ids.tostring()])),
            "question_elmo_idxs":      tf.train.Feature(bytes_list=tf.train.BytesList(value=[new_question_elmo_ids.tostring()])),
            "candidate_elmo_idxs":      tf.train.Feature(bytes_list=tf.train.BytesList(value=[new_candidate_elmo_ids.tostring()])),
            "context_idxs":      tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
            "ques_idxs":         tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
            "cand_idxs":         tf.train.Feature(bytes_list=tf.train.BytesList(value=[cand_idxs.tostring()])),
            "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
            "ques_char_idxs":    tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
            "cand_char_idxs":    tf.train.Feature(bytes_list=tf.train.BytesList(value=[cand_char_idxs.tostring()])),
            "cand_label":        tf.train.Feature(bytes_list=tf.train.BytesList(value=[cand_label.tostring()])),
            "cand_positions":    tf.train.Feature(bytes_list=tf.train.BytesList(value=[cand_positions.tostring()])),
            "y1":                tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
            "y2":                tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
            "id":                tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))}))

        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)

def prepro(config):
    word_counter, char_counter = Counter(), Counter()

    # process_file
    # train
    train_examples, train_eval = process_file(config, config.train_file, "train", word_counter, char_counter)
    save(config.train_eval_file, train_eval, message="train eval")
    # dev
    dev_examples, dev_eval = process_file(config, config.dev_file, "dev", word_counter, char_counter, is_test=True)
    save(config.dev_eval_file, dev_eval, message="dev eval")
    # test
    test_examples, test_eval = process_file(config, config.test_file, "test", word_counter, char_counter, is_test=True)
    save(config.test_eval_file, test_eval, message="test eval")

    # get_embedding
    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    if os.path.isfile(config.word_emb_file) and os.path.isfile(config.word_dictionary):
        print("word embedding pre-processed files exist, loading...")
        with open(config.word_emb_file, "r") as fw, open(config.word_dictionary, "r") as fd:
            word_emb_mat = json.load(fw)
            word2idx_dict = json.load(fd)
    else:
        word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", emb_file=word_emb_file,
                                                    size=config.glove_word_size, vec_size=config.glove_dim)
        save(config.word_emb_file, word_emb_mat, message="word embedding")
        save(config.word_dictionary, word2idx_dict, message="word dictionary")

    if os.path.isfile(config.char_emb_file) and os.path.isfile(config.char_dictionary):
        print("char embedding pre-processed files exist, loading...")
        with open(config.char_emb_file, "r") as fc, open(config.char_dictionary, "r") as fd:
            char_emb_mat = json.load(fc)
            char2idx_dict = json.load(fd)
    else:
        char_emb_mat, char2idx_dict = get_embedding(char_counter, "char", emb_file=char_emb_file,
                                                    size=char_emb_size, vec_size=char_emb_dim)
        save(config.char_emb_file, char_emb_mat, message="char embedding")
        save(config.char_dictionary, char2idx_dict, message="char dictionary")

    # build_features
    # train
    build_features(config, train_examples, "train", config.train_record_file, word2idx_dict, char2idx_dict)
    # dev
    dev_meta = build_features(config, dev_examples, "dev", config.dev_record_file, word2idx_dict, char2idx_dict,
                              is_test=True)
    save(config.dev_meta, dev_meta, message="dev meta")
    # test
    test_meta = build_features(config, test_examples, "test", config.test_record_file, word2idx_dict, char2idx_dict,
                               is_test=True)
    save(config.test_meta, test_meta, message="test meta")

