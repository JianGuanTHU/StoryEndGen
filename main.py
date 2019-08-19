import numpy as np
import tensorflow as tf
import sys
import time
import random
from pattern.en import lemma
random.seed(time.time())

from model import IEMSAModel, _START_VOCAB

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 10000, "vocabulary size.")
tf.app.flags.DEFINE_integer("embed_units", 200, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_integer("triple_num", 10, "max number of triple for each query")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("inference_path", "", "Set filename of inference, default isscreen")

FLAGS = tf.app.flags.FLAGS

def load_data(path, fname):    
    post = []
    with open('%s/%s.post' % (path, fname)) as f:
        for line in f:
            tmp = line.strip().split("\t")
            post.append([p.split() for p in tmp])

    with open('%s/%s.response' % (path, fname)) as f:
        response = [line.strip().split() for line in f.readlines()]
    data = []
    for p, r in zip(post, response):
        data.append({'post': p, 'response': r})
    return data

def load_relation(path):  
    file = open('%s/triples_shrink.txt' % (path), "r")
    
    relation = {}
    for line in file:
        tmp = line.strip().split()
        if tmp[0] in relation:
            if tmp[2] not in relation[tmp[0]]:
                relation[tmp[0]].append(tmp)
        else:
            relation[tmp[0]] = [tmp]

    for r in relation.keys():
        tmp_vocab = {}
        i = 0
        for re in relation[r]:
            if re[2] in vocab_dict.keys():
                tmp_vocab[i] = vocab_dict[re[2]]
            i += 1
        tmp_list = sorted(tmp_vocab, key=tmp_vocab.get)[:FLAGS.triple_num] if len(tmp_vocab) > FLAGS.triple_num else sorted(tmp_vocab, key=tmp_vocab.get)
        new_relation = []
        for i in tmp_list:
            new_relation.append(relation[r][i])
        relation[r] = new_relation

    return relation

def build_vocab(path, data):
    print("Creating vocabulary...")
 
    relation_vocab_list = []
    relation_file = open(path + "/relations.txt", "r")
    for line in relation_file:
        relation_vocab_list += line.strip().split()
   
    vocab = {}
    for i, pair in enumerate(data):
        if i % 100000 == 0:
            print("    processing line %d" % i)
        for token in [word for p in pair['post'] for word in p]+pair['response']:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    vocab_list = _START_VOCAB + relation_vocab_list + sorted(vocab, key=vocab.get, reverse=True)

    if len(vocab_list) > FLAGS.symbols:
        vocab_list = vocab_list[:FLAGS.symbols]

    print("Loading word vectors...")
    vectors = {}    
    with open(path + '/glove.6B.200d.txt', 'r') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector

    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = map(float,  vectors[word].split())
        else:
            vector = np.zeros((FLAGS.embed_units), dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
    return vocab_list, embed, vocab

def gen_batched_data(data):
    encoder_len = [max([len(item['post'][i]) for item in data]) + 1 for i in range(4)]
    decoder_len = max([len(item['response']) for item in data]) + 1    
    posts_1, posts_2, posts_3, posts_4, posts_length_1, posts_length_2, posts_length_3, posts_length_4, responses, responses_length = [], [], [], [], [], [], [], [], [], []

    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)
            
    for item in data:
        posts_1.append(padding(item['post'][0], encoder_len[0]))
        posts_2.append(padding(item['post'][1], encoder_len[1]))
        posts_3.append(padding(item['post'][2], encoder_len[2]))
        posts_4.append(padding(item['post'][3], encoder_len[3]))

        posts_length_1.append(len(item['post'][0]) + 1)
        posts_length_2.append(len(item['post'][1]) + 1)
        posts_length_3.append(len(item['post'][2]) + 1)
        posts_length_4.append(len(item['post'][3]) + 1)

        responses.append(padding(item['response'], decoder_len))
        responses_length.append(len(item['response']) + 1)

    entity = [[], [], [], []]
    for item in data:
        for i in range(4):
            entity[i].append([])
            for word in item['post'][i]:
                try:
                    w = lemma(word).encode("ascii")
                except UnicodeDecodeError, e:
                    w = word
                if w in relation:
                    entity[i][-1].append(relation[w])
                else:
                    entity[i][-1].append([['_NAF_H', '_NAF_R', '_NAF_T']])
    max_response_length = [0,0,0,0]
    max_triple_length = [0,0,0,0]
    for i in range(4):
        for item in entity[i]:
            if len(item) > max_response_length[i]:
                max_response_length[i] = len(item)
            for triple in item:
                if len(triple) > max_triple_length[i]:
                    max_triple_length[i] = len(triple)
    for i in range(4):
        for j in range(len(entity[i])):
            for k in range(len(entity[i][j])):
                if len(entity[i][j][k]) < max_triple_length[i]:
                    entity[i][j][k] = entity[i][j][k] + [['_NAF_H', '_NAF_R', '_NAF_T']] * (max_triple_length[i] - len(entity[i][j][k]))
            if len(entity[i][j]) < (max_response_length[i] + 1):
                entity[i][j] = entity[i][j] + [[['_NAF_H', '_NAF_R', '_NAF_T']] * max_triple_length[i]] * (max_response_length[i] + 1 - len(entity[i][j]))
    
    entity_0, entity_1, entity_2, entity_3 = entity[0], entity[1], entity[2], entity[3]
    entity_mask = [[], [], [], []]
    for i in range(4):
        for j in range(len(entity[i])):
            entity_mask[i].append([])
            for k in range(len(entity[i][j])):
                entity_mask[i][-1].append([])
                for r in entity[i][j][k]:
                    if r[0] == '_NAF_H':
                        entity_mask[i][-1][-1].append(0)
                    else:
                        entity_mask[i][-1][-1].append(1)

    entity_mask_0, entity_mask_1, entity_mask_2, entity_mask_3 = entity_mask[0], entity_mask[1], entity_mask[2], entity_mask[3]

    batched_data = {'posts_1': np.array(posts_1),
                    'posts_2': np.array(posts_2), 
                    'posts_3': np.array(posts_3), 
                    'posts_4': np.array(posts_4),       
                    'entity_1': np.array(entity_0),
                    'entity_2': np.array(entity_1),
                    'entity_3': np.array(entity_2),
                    'entity_4': np.array(entity_3),
                    'entity_mask_1': np.array(entity_mask_0),
                    'entity_mask_2': np.array(entity_mask_1),
                    'entity_mask_3': np.array(entity_mask_2),
                    'entity_mask_4': np.array(entity_mask_3),
                    'posts_length_1': posts_length_1, 
                    'posts_length_2': posts_length_2, 
                    'posts_length_3': posts_length_3, 
                    'posts_length_4': posts_length_4, 
                    'responses': np.array(responses),
                    'responses_length': responses_length}
    return batched_data

def train(model, sess, dataset):
    st, ed, loss = 0, 0, []
    while ed < len(dataset):
        print "epoch %d, training %.4f %%...\r" % (epoch, float(ed) / len(dataset) * 100),
        st, ed = ed, ed + FLAGS.batch_size if ed + \
            FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batched_data(dataset[st:ed])
        outputs = model.step_decoder(sess, batch_data)
        loss.append(outputs[0])

    sess.run(model.epoch_add_op)
    return np.mean(loss) 

def evaluate(model, sess, dataset):
    st, ed, loss = 0, 0, []
    while ed < len(dataset):
        print "epoch %d, evaluate %.4f %%...\r" % (epoch, float(ed) / len(dataset) * 100),
        st, ed = ed, ed + FLAGS.batch_size if ed + \
            FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batched_data(dataset[st:ed])
        outputs = model.step_decoder(sess, batch_data, forward_only=True)
        loss.append(outputs[0])
    return np.mean(loss)

def inference(model, sess, dataset):
    st, ed, posts, truth, generations, alignments_2, alignments_3, alignments_4, alignments = 0, 0, [], [], [], [], [], [], []
    while ed < len(dataset):
        st, ed = ed, ed + FLAGS.batch_size if ed + \
            FLAGS.batch_size < len(dataset) else len(dataset)
        data = gen_batched_data(dataset[st:ed])
        outputs = sess.run(['generation:0', model.alignments_2, model.alignments_3, model.alignments_4, model.alignments],
                           {model.posts_1: data['posts_1'],
                            model.posts_2: data['posts_2'],
                            model.posts_3: data['posts_3'],
                            model.posts_4: data['posts_4'],
                            model.entity_1: data['entity_1'],
                            model.entity_2: data['entity_2'],
                            model.entity_3: data['entity_3'],
                            model.entity_4: data['entity_4'],
                            model.entity_mask_1: data['entity_mask_1'],
                            model.entity_mask_2: data['entity_mask_2'],
                            model.entity_mask_3: data['entity_mask_3'],
                            model.entity_mask_4: data['entity_mask_4'],
                            model.posts_length_1: data['posts_length_1'],
                            model.posts_length_2: data['posts_length_2'],
                            model.posts_length_3: data['posts_length_3'],
                            model.posts_length_4: data['posts_length_4']})
        generations.append(outputs[0])
        alignments_2.append(outputs[1])
        alignments_3.append(outputs[2])
        alignments_4.append(outputs[3])
        alignments.append(outputs[4])

        posts.append([d['post'] for d in dataset[st:ed]])
        truth.append([d['response'] for d in dataset[st:ed]])

    output_file = open("./output_"+ str(FLAGS.inference_version) + ".txt", "w")
        
    for batch_generation in generations:
        for response in batch_generation:
            result = []
            for token in response:
                if token != '_EOS':
                    result.append(token)
                else:
                    break
            print >> output_file, ' '.join(result)
    return 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        data_train = load_data(FLAGS.data_dir, 'train')
        data_dev = load_data(FLAGS.data_dir, 'val')
        data_test = load_data(FLAGS.data_dir, 'test')
        vocab, embed, vocab_dict = build_vocab(FLAGS.data_dir, data_train)
        relation = load_relation(FLAGS.data_dir)

        model = IEMSAModel(
                FLAGS.symbols, 
                FLAGS.embed_units,
                FLAGS.units, 
                FLAGS.layers,
                is_train=True,
                vocab=vocab,
                embed=embed)

        if FLAGS.log_parameters:
            model.print_parameters()
        
        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
            model.symbol2index.init.run()
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            model.symbol2index.init.run()
        pre_losses = [1e18] * 3
        while True:
            epoch = model.epoch.eval()
            random.shuffle(data_train)
            start_time = time.time()
            loss = train(model, sess, data_train)
            model.saver.save(sess, '%s/checkpoint' %
                             FLAGS.train_dir, global_step=model.global_step)
            if loss > max(pre_losses):
                sess.run(model.learning_rate_decay_op)
            pre_losses = pre_losses[1:] + [loss]
            print "epoch %d learning rate %.4f epoch-time %.4f perplexity [%.8f]" \
                  % (epoch, model.learning_rate.eval(), time.time() - start_time, np.exp(loss))

            loss = evaluate(model, sess, data_dev)
            print "        val_set, perplexity [%.8f]" % np.exp(loss)
            loss = evaluate(model, sess, data_test)
            print "        test_set, perplexity [%.8f]" % np.exp(loss)

    else:
        model = IEMSAModel(
                FLAGS.symbols, 
                FLAGS.embed_units, 
                FLAGS.units, 
                FLAGS.layers, 
                is_train=False,
                vocab=None)

        if FLAGS.log_parameters:
            model.print_parameters()

        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (
                FLAGS.train_dir, FLAGS.inference_version)
        print 'restore from %s' % model_path
        model.saver.restore(sess, model_path)
        model.symbol2index.init.run()

        data_train = load_data(FLAGS.data_dir, 'train')
        data_dev = load_data(FLAGS.data_dir, 'val')
        data_test = load_data(FLAGS.data_dir, 'test')
        vocab, embed, vocab_dict = build_vocab(FLAGS.data_dir, data_train)
        relation = load_relation(FLAGS.data_dir)

        inference(model, sess, data_test)
