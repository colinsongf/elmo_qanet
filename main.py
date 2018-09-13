import tensorflow as tf
import json as json
import numpy as np
from tqdm import tqdm
import os, sys

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

from model import Model
from demo import Demo
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset, convert_tokens_cand


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    '''
        Iterator: Represents the state of iterating through a Dataset.
        
        from_string_handle(): https://www.tensorflow.org/api_docs/python/tf/data/Iterator
            This method allows you to define a "feedable" iterator where you can choose between concrete iterators 
            by feeding a value in a tf.Session.run call. In that case, string_handle would a tf.placeholder, 
            and you would feed it with the value of tf.data.Iterator.string_handle in each step.
        
        make_one_shot_iterator():Creates an Iterator for enumerating the elements of this dataset. 
            The returned iterator will be initialized automatically. 
            A "one-shot" iterator does not currently support re-initialization.
    '''

    dev_total = meta["total"]

    print("get recorded feature parser...")
    parser = get_record_parser(config)

    graph = tf.Graph()
    with graph.as_default() as g:

        print("get dataset batch iterator...")
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_dataset = get_dataset(config.dev_record_file, parser, config)
        dev_iterator = dev_dataset.make_one_shot_iterator()

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

        print("Building model...")
        model = Model(config, iterator, word_mat, char_mat, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        patience = 0
        best_em = 0.0
        best_f1 = 0.0
        best_acc = 0.0

        with tf.Session(config=sess_config) as sess:

            sess.run(tf.global_variables_initializer())

            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())

            saver = tf.train.Saver(max_to_keep=100)
            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))

            writer = tf.summary.FileWriter(config.log_dir)

            global_step = max(sess.run(model.global_step), 1)
            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1

                # train on one batch
                loss, train_op = sess.run([model.loss, model.train_op],
                                          feed_dict={handle: train_handle, model.dropout: config.dropout})
                # DEBUG
                # temp = sess.run(model.debug_ops, feed_dict={handle: train_handle, model.dropout: config.dropout})
                # for t in temp:
                #     # print(t)
                #     print(t.shape)
                # sys.exit(0)

                # save batch loss
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)

                if global_step % config.checkpoint == 0:
                    # evaluate on train
                    print('Evaluating on Train...')
                    num_bt = config.val_num_batches
                    # _, summ = evaluate_batch(model, num_bt, train_eval_file, sess, "train", handle, train_handle)
                    _, summ = evaluate_batch_cand(model, num_bt, train_eval_file, sess, "train", handle, train_handle)
                    # write to summary
                    for s in summ: writer.add_summary(s, global_step)

                    # evaluate on dev
                    print('Evaluating on Dev...')
                    num_bt = dev_total // config.batch_size + 1
                    # metrics, summ = evaluate_batch(model, num_bt, dev_eval_file, sess, "dev", handle, dev_handle)
                    metrics, summ = evaluate_batch_cand(model, num_bt, dev_eval_file, sess, "dev", handle, dev_handle)
                    # write to summary
                    for s in summ: writer.add_summary(s, global_step)
                    writer.flush()
                    # save checkpoint model
                    #filename = os.path.join(config.save_dir, "model_{}.ckpt".format(global_step))
                    #saver.save(sess, filename)

                    # early stop
                    # dev_f1 = metrics["f1"]
                    # dev_em = metrics["exact_match"]
                    dev_acc = metrics["acc"]
                    if dev_acc < best_acc:
                        patience += 1
                        if patience > config.early_stop:
                            print('>>>>>> WARNING !!! <<<<<<< Early_stop reached!!!')
                    # save best model
                    else:
                        patience = 0

                        best_acc = dev_acc
                        #filename = os.path.join(config.save_dir, "model_{}.best".format(global_step))
                        filename = os.path.join(config.save_dir, "model_{}.ckpt".format(global_step))
                        saver.save(sess, filename)
                            # if dev_f1 >= best_f1:
                            #     best_f1 = dev_f1
                            #     filename = os.path.join(config.save_dir, "model_{}.bestf1".format(global_step))
                            #     saver.save(sess, filename)


def evaluate_batch(model, num_batches, eval_file, sess, data_set, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2 = sess.run([model.qa_id, model.loss, model.yp1, model.yp2],feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_set), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/f1".format(data_set), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/em".format(data_set), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]

def evaluate_batch_cand(model, num_batches, eval_file, sess, data_set, handle, str_handle):
    pred = []
    gold = []
    losses = []
    metrics = {}
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, logits, yx= sess.run([model.qa_id, model.loss, model.cand_logits, model.yx],
                                          feed_dict={handle: str_handle})
        pred.extend(np.argmax(logits, axis=1))
        gold.extend(np.argmax(yx, axis=1))
        losses.append(loss)
    loss = np.mean(losses)
    correct = [i for i, j in zip(gold, pred) if i == j]
    acc = 100.0*len(correct)/len(gold)
    metrics["loss"] = loss
    metrics["acc"] = acc
    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_set), simple_value=metrics["loss"]), ])
    acc_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/accuracy".format(data_set), simple_value=metrics["acc"]), ])
    # em_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/em".format(data_set), simple_value=metrics["exact_match"]), ])
    print("\naccuracy = {:.3f}%".format(acc))
    return metrics, [loss_sum, acc_sum]


def demo(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    model = Model(config, None, word_mat, char_mat, trainable=False, demo = True)
    demo = Demo(model, config)


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]

    graph = tf.Graph()
    with graph.as_default() as g:
        test_batch = get_dataset(config.test_record_file, get_record_parser(config, is_test=True),
                                 config).make_one_shot_iterator()

        model = Model(config, test_batch, word_mat, char_mat, trainable=False, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # model_path = os.path.join(config.save_dir, 'model_79000.bestckpt')
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            print("Model Loaded from --> {}".format(tf.train.latest_checkpoint(config.save_dir)))
            # saver.restore(sess, model_path)
            # print("Model Loaded from --> {}".format(model_path))
            if config.decay < 1.0:
                sess.run(model.assign_vars)
            losses = []
            pred = []
            gold = []
            answer_dict = {}
            metrics = {}
            remapped_dict = {}
            for step in tqdm(range(total // config.batch_size + 1)):
                qa_id, loss, logits, yx = sess.run([model.qa_id, model.loss, model.cand_logits, model.yx])
                pred_idx = np.argmax(logits, axis=1)
                pred.extend(pred_idx)
                gold_idx = np.argmax(yx, axis=1)
                gold.extend(gold_idx)
                answer_dict_ = convert_tokens_cand(eval_file, qa_id.tolist(), pred_idx.tolist(), gold_idx.tolist())

                answer_dict.update(answer_dict_)
                losses.append(loss)
            loss = np.mean(losses)
            correct = [i for i, j in zip(gold, pred) if i == j]
            acc = 100.0 * len(correct) / len(gold)
            metrics["loss"] = loss
            metrics["acc"] = acc
            # metrics = evaluate(eval_file, answer_dict)
            with open(config.answer_file, "w") as fh:
                json.dump(answer_dict, fh)
            print("Loss: {}, Accuracy: {}".format(metrics['loss'], metrics['acc']))
