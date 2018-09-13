import tensorflow as tf
import re
from collections import Counter
import string

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


def get_record_parser(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit
        cand_limit = config.cand_limit

        '''
            tf.parse_single_example: Parses a single Example proto
            tf.decode_raw: Reinterpret the bytes of a string as a vector of numbers.
        '''
        features = tf.parse_single_example(example,
                                           features={
                                               "context_elmo_idxs":      tf.FixedLenFeature([], tf.string),
                                               "question_elmo_idxs":         tf.FixedLenFeature([], tf.string),
                                               "candidate_elmo_idxs":         tf.FixedLenFeature([], tf.string),
                                               "context_idxs":      tf.FixedLenFeature([], tf.string),
                                               "ques_idxs":         tf.FixedLenFeature([], tf.string),
                                               "cand_idxs":         tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs":    tf.FixedLenFeature([], tf.string),
                                               "cand_char_idxs":    tf.FixedLenFeature([], tf.string),
                                               "cand_label":        tf.FixedLenFeature([], tf.string),
                                               "cand_positions":    tf.FixedLenFeature([], tf.string),
                                               "y1":                tf.FixedLenFeature([], tf.string),
                                               "y2":                tf.FixedLenFeature([], tf.string),
                                               "id":                tf.FixedLenFeature([], tf.int64)
                                           })
        context_elmo_idxs =      tf.reshape(tf.decode_raw(features["context_elmo_idxs"],      tf.int32), [2*(para_limit + 2), 50])
        question_elmo_idxs =         tf.reshape(tf.decode_raw(features["question_elmo_idxs"],         tf.int32), [2*(ques_limit + 2), 50])
        candidate_elmo_idxs =         tf.reshape(tf.decode_raw(features["candidate_elmo_idxs"],         tf.int32), [2*(cand_limit + 2), 50])        
        context_idxs =      tf.reshape(tf.decode_raw(features["context_idxs"],      tf.int32), [para_limit])
        ques_idxs =         tf.reshape(tf.decode_raw(features["ques_idxs"],         tf.int32), [ques_limit])
        cand_idxs =         tf.reshape(tf.decode_raw(features["cand_idxs"],         tf.int32), [cand_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_char_idxs =    tf.reshape(tf.decode_raw(features["ques_char_idxs"],    tf.int32), [ques_limit, char_limit])
        cand_char_idxs =    tf.reshape(tf.decode_raw(features["cand_char_idxs"],    tf.int32), [cand_limit, char_limit])
        cand_label =        tf.reshape(tf.decode_raw(features["cand_label"],        tf.float32), [cand_limit])
        cand_positions =    tf.reshape(tf.decode_raw(features["cand_positions"], tf.float32), [cand_limit, para_limit])
        y1 =                tf.reshape(tf.decode_raw(features["y1"],                tf.float32), [para_limit])
        y2 =                tf.reshape(tf.decode_raw(features["y2"],                tf.float32), [para_limit])
        qa_id =             features["id"]
        
        return context_elmo_idxs, question_elmo_idxs, candidate_elmo_idxs, \
               context_idxs, ques_idxs, cand_idxs, \
               context_char_idxs, ques_char_idxs, cand_char_idxs, \
               cand_label, cand_positions, y1, y2, qa_id
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    '''
        tf.data.TFRecordDataset: A Dataset comprising records from one or more TFRecord files.
        range(start, stop[, step])
        tf.data.TFRecordDataset. batch(): Combines consecutive elements of this dataset into batches.
             NOTE: If the number of elements (N) in this dataset is not an exact multiple of batch_size, 
             the final batch contain smaller tensors with shape N % batch_size in the batch dimension. 
             If your program needs batches having the same shape, use tf.contrib.data.batch_and_drop_remainder
        tf.data.TFRecordDataset.repeat(): Repeats this dataset count times. None or -1: repeated indefinitely
        tf.data.TFRecordDataset.apply(): Apply a transformation function to this dataset.
        tf.data.TFRecordDataset.shuffle()--buffer_size: A tf.int64 scalar tf.Tensor, 
            representing the number of elements from this dataset from which the new dataset will sample. 
            (see https://stackoverflow.com/questions/46444018/
            meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle/48096625#48096625)
        tf.clip_by_value: Clips tensor values to a specified min and max.
        tf.data.Dataset.shuffle(): handle datasets that are too large to fit in memory
    '''
    dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads)\
        .shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_elmo_idxs, question_elmo_idxs, candidate_elmo_idxs,
                     context_idxs, ques_idxs, cand_idxs,
                     context_char_idxs, ques_char_idxs, cand_char_idxs,
                     cand_label, cand_positions, y1, y2, qa_id):
            c_len = tf.reduce_sum(tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5*config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads)\
        .repeat().batch(config.batch_size)
    return dataset


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def convert_tokens_cand(eval_file, qa_id, pred, gold):
    answer_dict = {}
    for qid, p, g in zip(qa_id, pred, gold):
        # context = eval_file[str(qid)]["context"]
        # spans = eval_file[str(qid)]["spans"]
        # uuid = eval_file[str(qid)]["uuid"]
        candidates = eval_file[str(qid)]["candidates"]
        gold_cand = candidates[g]
        pred_cand = candidates[p]
        answer = eval_file[str(qid)]["answers"]
        # start_idx = spans[p1][0]
        # end_idx = spans[p2][1]
        # answer_dict[str(qid)] = context[start_idx: end_idx]
        # remapped_dict[uuid] = context[start_idx: end_idx]
        answer_dict[str(qid)] = (gold_cand, pred_cand, answer)
    return answer_dict

def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
