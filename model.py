import tensorflow as tf
import sys, copy
import os
import numpy as np
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from layers import initializer, regularizer, residual_block, highway, conv, \
    mask_logits, trilinear, total_params, optimized_trilinear_for_attention

class Model(object):
    def __init__(self, config, batch,
                 word_mat=None, char_mat=None, trainable=True, opt=True, demo = False, graph = None):

        self.config = config
        self.demo = demo
        self.debug_ops = []

        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            if self.demo:
                self.c_elmo = tf.placeholder(tf.int32, [None, config.test_para_limit + 2, 50], "context_elmo_idxs")
                self.q_elmo = tf.placeholder(tf.int32, [None, config.test_ques_limit + 2, 50], "question_elmo_idxs")
                self.x_elmo = tf.placeholder(tf.int32, [None, config.cand_limit, 50 + 2], "candidates_elmo_idxs")
                self.c = tf.placeholder(tf.int32, [None, config.test_para_limit], "context")
                self.q = tf.placeholder(tf.int32, [None, config.test_ques_limit], "question")
                self.x = tf.placeholder(tf.int32, [None, config.cand_limit], "candidates")
                self.ch = tf.placeholder(tf.int32, [None, config.test_para_limit, config.char_limit], "context_char")
                self.qh = tf.placeholder(tf.int32, [None, config.test_ques_limit, config.char_limit],"question_char")
                self.xh = tf.placeholder(tf.int32, [None, config.cand_limit, config.char_limit], "candidate_char")
                self.y1 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index1")
                self.y2 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index2")
            else:
                '''
                    get_next(): Returns a nested structure of tf.Tensors representing the next element.
                    In graph mode, you should typically call this method once and use its result as the input to 
                    another computation. A typical loop will call tf.Session.run on the result of that computation
                    
                    features = tf.parse_single_example(example,
                                       features={
                                           "context_idxs": tf.FixedLenFeature([], tf.string),
                                           "ques_idxs": tf.FixedLenFeature([], tf.string),
                                           "cand_idxs": tf.FixedLenFeature([], tf.string),
                                           "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                           "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                           "cand_char_idxs": tf.FixedLenFeature([], tf.string),
                                           "cand_label": tf.FixedLenFeature([], tf.string),
                                           "y1": tf.FixedLenFeature([], tf.string),
                                           "y2": tf.FixedLenFeature([], tf.string),
                                           "id": tf.FixedLenFeature([], tf.int64)
                                       })
                
                    c:     Tensor("IteratorGetNext:0", shape=(N, 500), dtype=int32)
                    q:     Tensor("IteratorGetNext:1", shape=(N, 50), dtype=int32)
                    x:     Tensor("IteratorGetNext:2", shape=(N, 50), dtype=int32)
                    ch:    Tensor("IteratorGetNext:3", shape=(N, 500, 16), dtype=int32)
                    qh:    Tensor("IteratorGetNext:3", shape=(N, 50, 16), dtype=int32)
                    xh:    Tensor("IteratorGetNext:4", shape=(N, 50, 16), dtype=int32)
                    yx:    Tensor("IteratorGetNext:6", shape=(N, 100), dtype=float32)
                    y1:    Tensor("IteratorGetNext:5", shape=(N, 500), dtype=float32)
                    y2:    Tensor("IteratorGetNext:6", shape=(N, 500), dtype=float32)
                    qa_id: Tensor("IteratorGetNext:7", shape=(N,), dtype=int64)
                    
                '''
                # batch: train_dataset iterator
                self.c_elmo, self.q_elmo, self.x_elmo, \
                self.c, self.q, self.x, \
                self.ch, self.qh, self.xh, \
                self.yx, self.xp, self.y1, self.y2, self.qa_id = batch.get_next()

            if self.config.max_margin:
                self.yx_inv = 1 - self.yx

            # TODO


            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                            trainable=False)
            self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

            # all initialized to the max_length matrices with zeros --> 1's cover actual lengths
            self.c_mask = tf.cast(self.c, tf.bool) # Tensor("Cast:0", shape=(N, 500), dtype=bool)
            self.q_mask = tf.cast(self.q, tf.bool) # Tensor("Cast_1:0", shape=(N, 50), dtype=bool)
            self.x_mask = tf.cast(self.x, tf.bool) # Tensor("Cast_2:0", shape=(N, 100), dtype=bool)

            # Tensor("Sum:0", shape=(N,), dtype=int32)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            # Tensor("Sum:0", shape=(N,), dtype=int32)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
            # Tensor("Sum:0", shape=(N,), dtype=int32)
            self.x_len = tf.reduce_sum(tf.cast(self.x_mask, tf.int32), axis=1)

            '''
                tf.slice(input_, begin, size, name=None): extracts a slice of size from a tensor input 
                    starting at the location specified by begin. The slice size is represented as a tensor shape, 
                    where size[i] is the number of elements of the 'i'th dimension of input that you want to slice. 
                    The (begin) for the slice is represented as an offset in each dimension of input.
                    In other words, begin[i] is the offset into the 'i'th dim of input that you want to slice from.
            '''

            # Memory space optimization
            if opt:
                N = config.batch_size if not self.demo else 1
                CL = config.char_limit
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)
                self.x_maxlen = tf.reduce_max(self.x_len)
                self.c_elmo = tf.slice(self.c_elmo, [0, 0, 0], [N, self.c_maxlen + 2, 50])                     # shape=(N, PL, 50)
                self.q_elmo = tf.slice(self.q_elmo, [0, 0, 0], [N, self.q_maxlen + 2, 50])                     # shape=(N, QL, 50)
                self.x_elmo = tf.slice(self.x_elmo, [0, 0, 0], [N, self.x_maxlen + 2, 50])                     # shape=(N, XL, 50)
                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])                     # shape=(N, PL)
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])                     # shape=(N, QL)
                self.x = tf.slice(self.x, [0, 0], [N, self.x_maxlen])                     # shape=(N, XL)
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])           # shape=(N, PL)
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])           # shape=(N, QL)
                self.x_mask = tf.slice(self.x_mask, [0, 0], [N, self.x_maxlen])           # shape=(N, XL)
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])            # shape=(N, PL, 16)
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])            # shape=(N, QL, 16)
                self.xh = tf.slice(self.xh, [0, 0, 0], [N, self.x_maxlen, CL])            # shape=(N, XL, 16)
                self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])                   # shape=(N, PL)
                self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])                   # shape=(N, PL)
                self.yx = tf.slice(self.yx, [0, 0], [N, self.x_maxlen])                   # shape=(N, XL)

                if self.config.cand_condense_vector:
                    self.xp = tf.slice(self.xp, [0, 0, 0], [N, self.x_maxlen, self.c_maxlen]) # shape=(N, XL, PL)

                if self.config.max_margin:
                    self.yx_inv = tf.slice(self.yx_inv, [0, 0], [N, self.x_maxlen])       # shape=(N, x_maxlen)
            else:
                self.c_maxlen, self.q_maxlen, self.x_maxlen = config.para_limit, config.ques_limit, config.cand_limit

            # DEBUG
            self.debug_ops.extend([self.xp, self.yx, self.y1])

            # shape=(N * c_maxlen)
            self.ch_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            # shape=(N * q_maxlen)
            self.qh_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

            self.forward()
            total_params()

            if trainable:
                self.lr = tf.minimum(config.learning_rate, 0.001 / tf.log(999.)
                                     * tf.log(tf.cast(self.global_step, tf.float32) + 1))
                self.opt = tf.train.AdamOptimizer(learning_rate = self.lr,beta1 = 0.8,beta2 = 0.999,epsilon = 1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        N = config.batch_size if not self.demo else 1
        PL = self.c_maxlen
        QL = self.q_maxlen
        XL = self.x_maxlen

        # DEBUG
        self.debug_ops.extend([PL, QL, XL])

        CL = config.char_limit  # 16
        d = config.hidden       # 96
        dc = config.char_dim    # 64
        nh = config.num_heads   # 1

        with tf.variable_scope("Input_Embedding_Layer"):
            '''
                self.ch : (N, c_maxlen, 16)
                self.qh : (N, q_maxlen, 16)
                self.xh : (N, x_maxlen, 16)
            '''
            ######################################
            #get elmo embeddings
            ######################################
            datadir = "/data/elmo_experiment_20180906/20180906_model"
            vocab_file = os.path.join(datadir, 'vocab-2016-09-10.txt')
            options_file = os.path.join(datadir, 'options.json')
            weight_file = os.path.join(datadir, 'weights.hdf5')
            print(vocab_file)
            print(options_file)
            print(weight_file)
            
            # Create a Batcher to map text to character ids.
            batcher = Batcher(vocab_file, 50)
            
            # Input placeholders to the biLM.
            #context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
            #question_character_ids = tf.placeholder('int32', shape=(None, None, 50))
            
            # Build the biLM graph.
            bilm = BidirectionalLanguageModel(options_file, weight_file)
            
            # Get ops to compute the LM embeddings.
            print(self.c)
            print(self.c.shape)
            #print(self.ch)
            #print(self.ch.shape)
            print(self.c_elmo)
            print(self.c_elmo.shape)
            print(self.q_elmo)
            print(self.q_elmo.shape)
            print(self.x_elmo)
            print(self.x_elmo.shape)
             
            context_embeddings_op = bilm(self.c_elmo)
            question_embeddings_op = bilm(self.q_elmo)
            candidate_embeddings_op = bilm(self.x_elmo)
            
            # Get an op to compute ELMo (weighted average of the internal biLM layers)
            # Our SQuAD model includes ELMo at both the input and output layers
            # of the task GRU, so we need 4x ELMo representations for the question
            # and context at each of the input and output.
            # We use the same ELMo weights for both the question and context
            # at each of the input and output.
            #context elmo
            elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
            with tf.variable_scope('', reuse=True):
                # the reuse=True scope reuses weights from the context for the question
                elmo_question_input = weight_layers(
                    'input', question_embeddings_op, l2_coef=0.0
                )
                elmo_candidate_input = weight_layers(
                    'input', candidate_embeddings_op, l2_coef=0.0
                )
            
            elmo_context_output = weight_layers(
                'output', context_embeddings_op, l2_coef=0.0
            )
            with tf.variable_scope('', reuse=True):
                # the reuse=True scope reuses weights from the context for the question
                elmo_question_output = weight_layers(
                    'output', question_embeddings_op, l2_coef=0.0
                )
                elmo_candidate_output = weight_layers(
                    'output', candidate_embeddings_op, l2_coef=0.0
                )
            
            ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch), [N * PL, CL, dc]) #(N*PL,16,64)
            qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh), [N * QL, CL, dc]) #(N*QL,16,64)
            xh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.xh), [N * XL, CL, dc]) #(N*XL,16,64)

            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)
            xh_emb = tf.nn.dropout(xh_emb, 1.0 - 0.5 * self.dropout)

            # BiDAF style conv-highway encoder: conv over chars in each word in a batch of passages
            ch_emb = conv(ch_emb, d, bias = True, activation = tf.nn.relu, kernel_size = 5,
                          name = "char_conv", reuse = None) # (N*c_maxlen, 16-5+1, 96)
            qh_emb = conv(qh_emb, d, bias = True, activation = tf.nn.relu, kernel_size = 5,
                          name = "char_conv", reuse = True) # (N*q_maxlen, 16-5+1, 96)
            xh_emb = conv(xh_emb, d, bias = True, activation = tf.nn.relu, kernel_size = 5,
                          name="char_conv", reuse=True)  # (N*x_maxlen, 16-5+1, 96)

            # Max Pooling
            ch_emb = tf.reduce_max(ch_emb, axis = 1) # (N*c_maxlen, 96)
            qh_emb = tf.reduce_max(qh_emb, axis = 1) # (N*q_maxlen, 96)
            xh_emb = tf.reduce_max(xh_emb, axis = 1) # (N*x_maxlen, 96)

            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]]) # (N, c_maxlen, 96)
            qh_emb = tf.reshape(qh_emb, [N, QL, qh_emb.shape[-1]]) # (N, q_maxlen, 96)
            xh_emb = tf.reshape(xh_emb, [N, XL, xh_emb.shape[-1]]) # (N, x_maxlen, 96)

            '''
                self.c : (N, c_maxlen)
                self.q : (N, q_maxlen)
                self.x : (N, x_maxlen)
            '''
            #print(self.c)
            #print(self.q)
            #print(self.x)
            
            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)#(N,c_maxlen,300)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)#(N,q_maxlen,300)
            x_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.x), 1.0 - self.dropout)#(N,x_maxlen,300)

            #c_emb_elmo = 
            #q_emb_elmo = 
            #x_emb_elmo = 

            c_emb = tf.concat([c_emb, ch_emb], axis=2) # (N, c_maxlen, 396)
            q_emb = tf.concat([q_emb, qh_emb], axis=2) # (N, q_maxlen, 396)
            x_emb = tf.concat([x_emb, xh_emb], axis=2) # (N, x_maxlen, 396)
            
            print(c_emb)
            print(c_emb.shape)
            
            c_emb = tf.concat([elmo_context_output['weighted_op'], c_emb], axis=2) # (N, c_maxlen, 1024 + 396)
            q_emb = tf.concat([elmo_question_output['weighted_op'], q_emb], axis=2) # (N, q_maxlen, 1024 + 396)
            x_emb = tf.concat([elmo_candidate_output['weighted_op'], x_emb], axis=2) # (N, x_maxlen, 1024 + 396)
            
            print(c_emb)
            print(c_emb.shape)

            c_emb = highway(c_emb, size = d, scope = "highway", dropout = self.dropout, reuse = None)#(N,c_maxlen,96)
            q_emb = highway(q_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)#(N,q_maxlen,96)
            x_emb = highway(x_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)#(N,x_maxlen,96)

        with tf.variable_scope("Embedding_Encoder_Layer"):
            '''
                -> positional encoding 
                -> layer_normalization 
                -> depth-wise separable convolution 
                -> self attention 
                -> feed forward network
                In the paper: The total number of encoder blocks is 1
            '''
            # (N, c_maxlen, 96)
            c = residual_block(c_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.c_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.c_len,
                scope = "Encoder_Residual_Block",
                bias = False,
                dropout = self.dropout)
            # (N, q_maxlen, 96)
            q = residual_block(q_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.q_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.q_len,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)

        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            '''
                tf.tile(input, multiples, name=None): creates a new tensor by replicating input multiples times. 
                    The output tensor's i'th dimension has input.dims(i) * multiples[i] elements, 
                    and the values of input are replicated multiples[i] times along the 'i'th dimension.
                Paper: The layer parameters are the same as the Embedding Encoder Layer 
                       except that convolution layer number is 2 within a block 
                       and the total number of blocks is 7
            '''
            '''
                c:        (N, c_maxlen, d)
                q:        (N, q_maxlen, d)
                ch_emb:   (N, c_maxlen, d)
                qh_emb:   (N, q_maxlen, d)
                C:        (N, c_maxlen, q_maxlen, d)
                Q:        (N, c_maxlen, q_maxlen, d)
                S:        (N, c_maxlen, q_maxlen)
                mask_q:   (N, 1, q_maxlen)
                mask_c:   (N, c_maxlen, 1)
                S_:       (N, c_maxlen, q_maxlen)
                S_T:      (N, q_maxlen, c_maxlen)
                self.c2q: (N, c_maxlen, d) = tf.matmul(S_, q)
                self.q2c: (N, c_maxlen, d) = tf.matmul(tf.matmul(S_, S_T), c)
            '''
            # change from commit f0c79cc93dc1dfdad2bc8abb712a53d078814a56 by Min on 27 Apr 18
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)

            # optimization from jasonwbw
            S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen,
                                                  input_keep_prob=1.0 - self.dropout)

            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), axis = 1),(0,2,1))

            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)

            # change from commit f0c79cc93dc1dfdad2bc8abb712a53d078814a56 by Min on 27 Apr 18
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]
            # if config.q2c:
            #     attention_outputs.append(c * self.q2c)

        # with tf.variable_scope("Model_Encoder_Layer"):
        #     inputs = tf.concat(attention_outputs, axis = -1)
        #
        #     # same as a dxd MLP layer
        #     self.enc = [conv(inputs, d, name = "input_projection")] # d=hidden=96
        #
        #     for i in range(3):
        #         if i % 2 == 0: # dropout every 2 blocks
        #             self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
        #         self.enc.append(
        #             residual_block(self.enc[i],
        #                 num_blocks = 7,
        #                 num_conv_layers = 2,
        #                 kernel_size = 5,
        #                 mask = self.c_mask,
        #                 num_filters = d,
        #                 num_heads = nh,
        #                 seq_len = self.c_len,
        #                 scope = "Model_Encoder",
        #                 bias = False,
        #                 reuse = True if i > 0 else None,
        #                 dropout = self.dropout)
        #             )

            # DEBUG
            # self.debug_ops.append(inputs)
            # self.debug_ops.extend(self.enc)

        with tf.variable_scope("Output_Layer"):
            '''
                broadcasting:dimensions with size 1 are stretched or "copied" to match the other
            '''
            '''
                x_emb:              (N, x_maxlen, d)
                inputs:             (N, c_maxlen, 4*d)
                mask_x:             (N, x_maxlen, 1)
                c_proj:             (N, c_maxlen, d)
                S_xc/S_xc_:         (N, x_maxlen, c_maxlen)
                x2c:                (N, x_maxlen, d)
                xp_exp:             (N, x_maxlen, c_maxlen, 1)
                c_proj_exp:         (N, 1, c_maxlen, d)
                cand_context:       (N, x_maxlen, c_maxlen, d)
                cand_context_pool:  (N, x_maxlen, d)
                cand_condense:      (N, x_maxlen, d*2)
                self.cand_condense: (N, x_maxlen, d)
                self.cand_logits:   (N, x_maxlen, 1)
            '''
            inputs = tf.concat(attention_outputs, axis = -1)

            # masking candidate embedding
            mask_x = tf.expand_dims(self.x_mask, 2)
            c_proj = conv(inputs, d, name="context_projection")

            S_xc = optimized_trilinear_for_attention([x_emb, c_proj], self.x_maxlen, self.c_maxlen,
                                                  input_keep_prob=1.0 - self.dropout)
            S_xc_ = tf.nn.softmax(mask_logits(S_xc, mask = mask_x))

            self.x2c = tf.matmul(S_xc_, c_proj)

            self.cand_condense = self.x2c

            if self.config.cand_condense_vector:
                xp_exp = tf.expand_dims(self.xp, axis=-1)
                c_proj_exp = tf.expand_dims(c_proj, axis=1)
                cand_context = tf.multiply(c_proj_exp, xp_exp)

                if self.config.cand_condense_conv:
                    cand_context = tf.reshape(cand_context, [N*XL, PL, d])
                    cand_context = conv(cand_context, d, bias=True, activation=tf.nn.relu,
                                        kernel_size=3, name="candidate_from_context")
                    cand_context = tf.reshape(cand_context, [N, XL, -1, d])

                if self.config.cand_condense_pool:
                    cand_context_pool = tf.reduce_max(cand_context, axis=-2)
                else:
                    cand_context_pool = tf.reduce_mean(cand_context, axis=-2)

                cand_condense = tf.concat([self.x2c, cand_context_pool], axis = -1)
                self.cand_condense = conv(cand_condense, d, name="candidate_projection")

                if self.config.cand_fuse_vector:
                    raise NotImplementedError

                # DEBUG
                self.debug_ops.extend([xp_exp, c_proj_exp, cand_context, cand_context_pool,
                                       cand_condense, self.cand_condense])

            if not config.max_margin:
                cand_logits = tf.squeeze(conv(self.cand_condense, 1, bias=False, name="candidate_logits_1"), -1)
                self.cand_logits = mask_logits(cand_logits, mask=self.x_mask)
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.cand_logits, labels=self.yx)
                # DEBUG
                self.debug_ops.extend([loss, x_emb, c_proj, S_xc, S_xc_, self.x2c,
                                       self.x_mask, self.cand_logits, self.yx])
            else:
                cand_logits = conv(self.cand_condense, 1, bias=False, name="candidate_logits_1")
                cand_logits = tf.tanh(cand_logits)
                cand_logits = tf.squeeze(conv(cand_logits, 1, bias=False, name="candidate_logits_2"), -1)
                self.cand_logits = tf.sigmoid(cand_logits)
                pos = tf.multiply(self.cand_logits, self.yx)
                pos = tf.reduce_max(pos, axis=-1)
                negs = tf.multiply(self.cand_logits, self.yx_inv)
                neg = tf.reduce_max(negs, axis=-1)
                loss = tf.maximum(tf.add(tf.subtract(neg, pos), config.margin), 0.0)
                # DEBUG
                self.debug_ops.extend([loss, x_emb, c_proj, S_xc, S_xc_, self.x2c,
                                       self.x_mask, self.cand_logits, self.yx,
                                       pos, negs, neg, self.yx, self.yx_inv])

            self.loss = tf.reduce_mean(loss)

        # with tf.variable_scope("Output_Layer"):
        #     '''
        #         tf.matrix_band_part: Copy a tensor setting everything outside a central band
        #                              in each innermost matrix to zero.
        #         self.enc[i]:  (N, c_maxlen, d)
        #         start_logits: (N, c_maxlen)
        #         end_logits:   (N, c_maxlen)
        #         logits1:      (N, c_maxlen)
        #         logits2:      (N, c_maxlen)
        #         outer:        (N, c_maxlen, c_maxlen)
        #         self.c_mask:  (N, c_maxlen)
        #         yp1, yp2, losses, losses2: (N,)
        #     '''
        #
        #     # map vectors to scalars
        #     start_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[2]],axis = -1),1,
        #                                    bias = False, name = "start_pointer"),-1)
        #     end_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[3]],axis = -1),1,
        #                                  bias = False, name = "end_pointer"), -1)
        #     self.logits = [mask_logits(start_logits, mask = self.c_mask), mask_logits(end_logits, mask = self.c_mask)]
        #
        #     logits1, logits2 = [l for l in self.logits]
        #
        #     losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=self.y1)
        #     losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=self.y2)
        #     self.loss = tf.reduce_mean(losses + losses2)
        #
        #     # find max-score span
        #     outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
        #                       tf.expand_dims(tf.nn.softmax(logits2), axis=1))
        #     # change from commit f0c79cc93dc1dfdad2bc8abb712a53d078814a56 by Min on 27 Apr 18
        #     outer = tf.matrix_band_part(outer, 0, config.ans_limit)
        #     self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        #     self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
        #
        #     # DEBUG
        #     self.debug_ops.extend([start_logits, end_logits, logits1, logits2,
        #                            outer, self.yp1, self.yp2, losses, losses2, self.loss])

        if config.l2_norm is not None:
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            self.loss += l2_loss

        if config.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                # change from commit f0c79cc93dc1dfdad2bc8abb712a53d078814a56 by Min on 27 Apr 18
                self.assign_vars = []
                # self.shadow_vars = []
                # self.global_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.assign_vars.append(tf.assign(var, v))
                        # self.shadow_vars.append(v)
                        # self.global_vars.append(var)
                # self.assign_vars = []
                # for g,v in zip(self.global_vars, self.shadow_vars):
                #     self.assign_vars.append(tf.assign(g,v))

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
