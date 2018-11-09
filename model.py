import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import GRUCell, LSTMCell, MultiRNNCell
import attention_decoder_fn
from tensorflow.contrib.seq2seq.python.ops.seq2seq import dynamic_rnn_decoder
from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from output_projection import output_projection_layer
from tensorflow.python.ops import variable_scope

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS', '_NAF_H', '_NAF_R', '_NAF_T']

class IEMSAModel(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            is_train,
            vocab=None,
            embed=None,
            learning_rate=0.1,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=30,
            use_lstm=True):
        
        self.posts_1 = tf.placeholder(tf.string, shape=(None, None))
        self.posts_2 = tf.placeholder(tf.string, shape=(None, None))
        self.posts_3 = tf.placeholder(tf.string, shape=(None, None))
        self.posts_4 = tf.placeholder(tf.string, shape=(None, None))

        self.entity_1 = tf.placeholder(tf.string, shape=(None,None,None,3))
        self.entity_2 = tf.placeholder(tf.string, shape=(None,None,None,3))
        self.entity_3 = tf.placeholder(tf.string, shape=(None,None,None,3))
        self.entity_4 = tf.placeholder(tf.string, shape=(None,None,None,3))

        self.entity_mask_1 = tf.placeholder(tf.float32, shape=(None, None, None))
        self.entity_mask_2 = tf.placeholder(tf.float32, shape=(None, None, None))
        self.entity_mask_3 = tf.placeholder(tf.float32, shape=(None, None, None))
        self.entity_mask_4 = tf.placeholder(tf.float32, shape=(None, None, None))

        self.posts_length_1 = tf.placeholder(tf.int32, shape=(None))
        self.posts_length_2 = tf.placeholder(tf.int32, shape=(None))
        self.posts_length_3 = tf.placeholder(tf.int32, shape=(None))
        self.posts_length_4 = tf.placeholder(tf.int32, shape=(None))

        self.responses = tf.placeholder(tf.string, shape=(None, None))
        self.responses_length = tf.placeholder(tf.int32, shape=(None))

        self.epoch = tf.Variable(0, trainable=False, name='epoch')
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)

        if is_train:
            self.symbols = tf.Variable(vocab, trainable=False, name="symbols")
        else:
            self.symbols = tf.Variable(np.array(['.']*num_symbols), name="symbols")
        self.symbol2index = HashTable(KeyValueTensorInitializer(self.symbols, 
            tf.Variable(np.array([i for i in range(num_symbols)], dtype=np.int32), False)), 
            default_value=UNK_ID, name="symbol2index")

        self.posts_input_1 = self.symbol2index.lookup(self.posts_1)

        self.posts_2_target = self.posts_2_embed = self.symbol2index.lookup(self.posts_2)
        self.posts_3_target = self.posts_3_embed = self.symbol2index.lookup(self.posts_3)
        self.posts_4_target = self.posts_4_embed = self.symbol2index.lookup(self.posts_4)

        self.responses_target = self.symbol2index.lookup(self.responses)

        batch_size, decoder_len = tf.shape(self.posts_1)[0], tf.shape(self.responses)[1]

        self.posts_input_2 = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32) * GO_ID,
            tf.split(self.posts_2_embed, [tf.shape(self.posts_2)[1]-1, 1], 1)[0]], 1)
        self.posts_input_3 = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32) * GO_ID,
            tf.split(self.posts_3_embed, [tf.shape(self.posts_3)[1]-1, 1], 1)[0]], 1)
        self.posts_input_4 = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32) * GO_ID,
            tf.split(self.posts_4_embed, [tf.shape(self.posts_4)[1]-1, 1], 1)[0]], 1)

        self.responses_target = self.symbol2index.lookup(self.responses)

        batch_size, decoder_len = tf.shape(self.posts_1)[0], tf.shape(self.responses)[1]

        self.responses_input = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32)*GO_ID,
            tf.split(self.responses_target, [decoder_len-1, 1], 1)[0]], 1)

        self.encoder_2_mask = tf.reshape(tf.cumsum(tf.one_hot(self.posts_length_2-1, 
            tf.shape(self.posts_2)[1]), reverse=True, axis=1), [-1, tf.shape(self.posts_2)[1]])
        self.encoder_3_mask = tf.reshape(tf.cumsum(tf.one_hot(self.posts_length_3-1, 
            tf.shape(self.posts_3)[1]), reverse=True, axis=1), [-1, tf.shape(self.posts_3)[1]])
        self.encoder_4_mask = tf.reshape(tf.cumsum(tf.one_hot(self.posts_length_4-1, 
            tf.shape(self.posts_4)[1]), reverse=True, axis=1), [-1, tf.shape(self.posts_4)[1]])

        self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1, 
            decoder_len), reverse=True, axis=1), [-1, decoder_len])

        if embed is None:
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
        else:
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
        
        self.encoder_input_1 = tf.nn.embedding_lookup(self.embed, self.posts_input_1)
        self.encoder_input_2 = tf.nn.embedding_lookup(self.embed, self.posts_input_2)
        self.encoder_input_3 = tf.nn.embedding_lookup(self.embed, self.posts_input_3)
        self.encoder_input_4 = tf.nn.embedding_lookup(self.embed, self.posts_input_4)

        self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input) 

        entity_embedding_1 = tf.reshape(tf.nn.embedding_lookup(self.embed, self.symbol2index.lookup(self.entity_1)), 
                            [batch_size, tf.shape(self.entity_1)[1], tf.shape(self.entity_1)[2], 3 * num_embed_units])
        entity_embedding_2 = tf.reshape(tf.nn.embedding_lookup(self.embed, self.symbol2index.lookup(self.entity_2)), 
                            [batch_size, tf.shape(self.entity_2)[1], tf.shape(self.entity_2)[2], 3 * num_embed_units])
        entity_embedding_3 = tf.reshape(tf.nn.embedding_lookup(self.embed, self.symbol2index.lookup(self.entity_3)), 
                            [batch_size, tf.shape(self.entity_3)[1], tf.shape(self.entity_3)[2], 3 * num_embed_units])
        entity_embedding_4 = tf.reshape(tf.nn.embedding_lookup(self.embed, self.symbol2index.lookup(self.entity_4)), 
                            [batch_size, tf.shape(self.entity_4)[1], tf.shape(self.entity_4)[2], 3 * num_embed_units])

        head_1, relation_1, tail_1 = tf.split(entity_embedding_1, [num_embed_units] * 3, axis=3)
        head_2, relation_2, tail_2 = tf.split(entity_embedding_2, [num_embed_units] * 3, axis=3)
        head_3, relation_3, tail_3 = tf.split(entity_embedding_3, [num_embed_units] * 3, axis=3)
        head_4, relation_4, tail_4 = tf.split(entity_embedding_4, [num_embed_units] * 3, axis=3)

        with tf.variable_scope('graph_attention'):
            #[batch_size, max_reponse_length, max_triple_num, 2*embed_units]            
            head_tail_1 = tf.concat([head_1, tail_1], axis=3)
            #[batch_size, max_reponse_length, max_triple_num, embed_units]            
            head_tail_transformed_1 = tf.layers.dense(head_tail_1, num_embed_units, activation=tf.tanh, name='head_tail_transform')
            #[batch_size, max_reponse_length, max_triple_num, embed_units]            
            relation_transformed_1 = tf.layers.dense(relation_1, num_embed_units, name='relation_transform')
            #[batch_size, max_reponse_length, max_triple_num]            
            e_weight_1 = tf.reduce_sum(relation_transformed_1 * head_tail_transformed_1, axis=3) 
            #[batch_size, max_reponse_length, max_triple_num]            
            alpha_weight_1 = tf.nn.softmax(e_weight_1)
            #[batch_size, max_reponse_length, embed_units]            
            graph_embed_1 = tf.reduce_sum(tf.expand_dims(alpha_weight_1, 3) * (tf.expand_dims(self.entity_mask_1, 3) * head_tail_1), axis=2)

        with tf.variable_scope('graph_attention', reuse=True):
            head_tail_2 = tf.concat([head_2, tail_2], axis=3)
            head_tail_transformed_2 = tf.layers.dense(head_tail_2, num_embed_units, activation=tf.tanh, name='head_tail_transform')
            relation_transformed_2 = tf.layers.dense(relation_2, num_embed_units, name='relation_transform')
            e_weight_2 = tf.reduce_sum(relation_transformed_2 * head_tail_transformed_2, axis=3) 
            alpha_weight_2 = tf.nn.softmax(e_weight_2)
            graph_embed_2 = tf.reduce_sum(tf.expand_dims(alpha_weight_2, 3) * (tf.expand_dims(self.entity_mask_2, 3) * head_tail_2), axis=2)

        with tf.variable_scope('graph_attention', reuse=True):
            head_tail_3 = tf.concat([head_3, tail_3], axis=3)
            head_tail_transformed_3 = tf.layers.dense(head_tail_3, num_embed_units, activation=tf.tanh, name='head_tail_transform')
            relation_transformed_3 = tf.layers.dense(relation_3, num_embed_units, name='relation_transform')
            e_weight_3 = tf.reduce_sum(relation_transformed_3 * head_tail_transformed_3, axis=3) 
            alpha_weight_3 = tf.nn.softmax(e_weight_3)
            graph_embed_3 = tf.reduce_sum(tf.expand_dims(alpha_weight_3, 3) * (tf.expand_dims(self.entity_mask_3, 3) * head_tail_3), axis=2)            

        with tf.variable_scope('graph_attention', reuse=True):
            head_tail_4 = tf.concat([head_4, tail_4], axis=3)
            head_tail_transformed_4 = tf.layers.dense(head_tail_4, num_embed_units, activation=tf.tanh, name='head_tail_transform')
            relation_transformed_4 = tf.layers.dense(relation_4, num_embed_units, name='relation_transform')
            e_weight_4 = tf.reduce_sum(relation_transformed_4 * head_tail_transformed_4, axis=3) 
            alpha_weight_4 = tf.nn.softmax(e_weight_4)
            graph_embed_4 = tf.reduce_sum(tf.expand_dims(alpha_weight_4, 3) * (tf.expand_dims(self.entity_mask_4, 3) * head_tail_4), axis=2)

        if use_lstm:
            cell = MultiRNNCell([LSTMCell(num_units)] * num_layers)
        else:
            cell = MultiRNNCell([GRUCell(num_units)] * num_layers)

        output_fn, sampled_sequence_loss = output_projection_layer(num_units,
                                                                   num_symbols, num_samples)

        encoder_output_1, encoder_state_1 = dynamic_rnn(cell, self.encoder_input_1, self.posts_length_1, dtype=tf.float32, scope="encoder")
 
        attention_keys_1, attention_values_1, attention_score_fn_1, attention_construct_fn_1 \
                = attention_decoder_fn.prepare_attention(graph_embed_1, encoder_output_1, 'luong', num_units)
        decoder_fn_train_1 = attention_decoder_fn.attention_decoder_fn_train(encoder_state_1, 
                attention_keys_1, attention_values_1, attention_score_fn_1, attention_construct_fn_1, max_length=tf.reduce_max(self.posts_length_2))
        encoder_output_2, encoder_state_2, alignments_ta_2 = dynamic_rnn_decoder(cell, decoder_fn_train_1,
                self.encoder_input_2, self.posts_length_2, scope="decoder")
        self.alignments_2 = tf.transpose(alignments_ta_2.stack(), perm=[1, 0, 2])

        self.decoder_loss_2 = sampled_sequence_loss(encoder_output_2, 
                self.posts_2_target, self.encoder_2_mask)

        with variable_scope.variable_scope('', reuse=True):
            attention_keys_2, attention_values_2, attention_score_fn_2, attention_construct_fn_2 \
                    = attention_decoder_fn.prepare_attention(graph_embed_2, encoder_output_2, 'luong', num_units)
            decoder_fn_train_2 = attention_decoder_fn.attention_decoder_fn_train(encoder_state_2, 
                    attention_keys_2, attention_values_2, attention_score_fn_2, attention_construct_fn_2, max_length=tf.reduce_max(self.posts_length_3))
            encoder_output_3, encoder_state_3, alignments_ta_3 = dynamic_rnn_decoder(cell, decoder_fn_train_2,
                self.encoder_input_3, self.posts_length_3, scope="decoder")
            self.alignments_3 = tf.transpose(alignments_ta_3.stack(), perm=[1, 0, 2])

            self.decoder_loss_3 = sampled_sequence_loss(encoder_output_3, 
                self.posts_3_target, self.encoder_3_mask)

            attention_keys_3, attention_values_3, attention_score_fn_3, attention_construct_fn_3 \
                    = attention_decoder_fn.prepare_attention(graph_embed_3, encoder_output_3, 'luong', num_units)
            decoder_fn_train_3 = attention_decoder_fn.attention_decoder_fn_train(encoder_state_3, 
                    attention_keys_3, attention_values_3, attention_score_fn_3, attention_construct_fn_3, max_length=tf.reduce_max(self.posts_length_4))
            encoder_output_4, encoder_state_4, alignments_ta_4 = dynamic_rnn_decoder(cell, decoder_fn_train_3,
                    self.encoder_input_4, self.posts_length_4, scope="decoder")
            self.alignments_4 = tf.transpose(alignments_ta_4.stack(), perm=[1, 0, 2])

            self.decoder_loss_4 = sampled_sequence_loss(encoder_output_4, 
                self.posts_4_target, self.encoder_4_mask)

            attention_keys, attention_values, attention_score_fn, attention_construct_fn \
                    = attention_decoder_fn.prepare_attention(graph_embed_4, encoder_output_4, 'luong', num_units)

        if is_train:
            with variable_scope.variable_scope('', reuse=True):
                decoder_fn_train = attention_decoder_fn.attention_decoder_fn_train(encoder_state_4, 
                    attention_keys, attention_values, attention_score_fn, attention_construct_fn, max_length=tf.reduce_max(self.responses_length))
                self.decoder_output, _, alignments_ta = dynamic_rnn_decoder(cell, decoder_fn_train,
                        self.decoder_input, self.responses_length, scope="decoder")
                self.alignments = tf.transpose(alignments_ta.stack(), perm=[1, 0, 2])

                self.decoder_loss = sampled_sequence_loss(self.decoder_output, 
                        self.responses_target, self.decoder_mask)
                
            self.params = tf.trainable_variables()
        
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, 
                    dtype=tf.float32)
            self.learning_rate_decay_op = self.learning_rate.assign(
                    self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)
            
            #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)

            gradients = tf.gradients(self.decoder_loss + self.decoder_loss_2 + self.decoder_loss_3 + self.decoder_loss_4, self.params)
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                    max_gradient_norm)
            self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                    global_step=self.global_step)

        else:
            with variable_scope.variable_scope('', reuse=True):
                decoder_fn_inference = attention_decoder_fn.attention_decoder_fn_inference(output_fn, 
                    encoder_state_4, attention_keys, attention_values, attention_score_fn, 
                    attention_construct_fn, self.embed, GO_ID, EOS_ID, max_length, num_symbols)                            
                self.decoder_distribution, _, alignments_ta = dynamic_rnn_decoder(cell, decoder_fn_inference,
                            scope="decoder")
                output_len = tf.shape(self.decoder_distribution)[1]
                self.alignments = tf.transpose(alignments_ta.gather(tf.range(output_len)), [1, 0, 2])

            self.generation_index = tf.argmax(tf.split(self.decoder_distribution, 
                [2, num_symbols-2], 2)[1], 2) + 2 # for removing UNK
            self.generation = tf.nn.embedding_lookup(self.symbols, self.generation_index, name="generation") 
            
            self.params = tf.trainable_variables()
        
        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def step_decoder(self, session, data, forward_only=False):
        input_feed = {self.posts_1: data['posts_1'],
                      self.posts_2: data['posts_2'],
                      self.posts_3: data['posts_3'],
                      self.posts_4: data['posts_4'],
                      self.entity_1: data['entity_1'],
                      self.entity_2: data['entity_2'],
                      self.entity_3: data['entity_3'],
                      self.entity_4: data['entity_4'],
                      self.entity_mask_1: data['entity_mask_1'],
                      self.entity_mask_2: data['entity_mask_2'],
                      self.entity_mask_3: data['entity_mask_3'],
                      self.entity_mask_4: data['entity_mask_4'],
                      self.posts_length_1: data['posts_length_1'],
                      self.posts_length_2: data['posts_length_2'],
                      self.posts_length_3: data['posts_length_3'],
                      self.posts_length_4: data['posts_length_4'],
                      self.responses: data['responses'],
                      self.responses_length: data['responses_length']}
        if forward_only:
            output_feed = [self.decoder_loss, self.alignments_2]
        else:
            output_feed = [self.decoder_loss, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)
