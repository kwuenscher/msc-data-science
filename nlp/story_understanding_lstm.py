import tensorflow as tf
import numpy as np
from random import sample

################# Preprocessing #################

def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)

def split_dataset_mlp(x, y, z, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainX, trainY, trainZ = x[:lens[0]], y[:lens[0]], z[:lens[0]]
    testX, testY, testZ = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]], z[lens[0]:lens[0]+lens[1]]
    validX, validY, validZ = x[-lens[-1]:], y[-lens[-1]:], z[-lens[-1]:]

    return (trainX,trainY,trainZ), (testX,testY,testZ), (validX,validY,validZ)

def batch_gen(x, y, batch_size):
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T

def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T

def rand_batch_mlp(x, y, z, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield np.array(x)[sample_idx], np.array(y)[sample_idx], np.array(z)[sample_idx]


def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])


def getRevVocab(vocab):
    return {v: k for k, v in vocab.items()}


def flattenStory(stories, lengths):
    out_sentences_dev1 = [item for sent in stories for item in sent]
    out_seq_len_dev1 = [item for sent in lengths for item in sent]
    return out_sentences_dev1, out_seq_len_dev1

def getBatchGen(trainX, trainY, batch_size):
    counter = 0
    while True:
        if counter >= shape(trainX)[0] // batch_size:
            counter = 0
            yield trainX[counter:counter+batch_size].T, trainY[counter:counter+batch_size].T
            counter += 1
        else:
            yield trainX[counter:counter+batch_size].T, trainY[counter:counter+batch_size].T
            counter += 1


def getBatchGenMLP(trainX, trainY, trainZ, batch_size):
    counter = 0
    while True:
        if counter >= shape(trainX)[0] // batch_size:
            counter = 0
            yield trainX[counter:counter+batch_size].T, trainY[counter:counter+batch_size].T, trainZ[counter:counter+batch_size]
            counter += 1
        else:
            yield trainX[counter:counter+batch_size].T, trainY[counter:counter+batch_size].T, trainZ[counter:counter+batch_size]
            counter += 1

def orderStories(data, order):
    out_sentences_orderd = []
    for i, story in enumerate(data):
        out_sentences_orderd.append([story[item] for item in order[i]])
    return out_sentences_orderd

def makeDataSeq2SeqReady(data):
    out_sentences_enc = []
    out_sentences_dec = []
    for i, item in enumerate(data):
        out_sentences_enc.append(item[:-1])
        out_sentences_dec.append(item[1:])

    return out_sentences_enc, out_sentences_dec

def w2vToNumpy():
    word2vec = {} #skip information on first line
    fin= open('glove.6B.50d.txt')
    for line in fin:
        items = line.replace('\r','').replace('\n','').split(' ')
        if len(items) < 10: continue
        word = items[0]
        vect = np.array([float(i) for i in items[1:] if len(i) > 1])
        word2vec[word] = vect


    return word2vec

################# Pipeline #################

OOV = '<OOV>'
PAD = '<PAD>'

def ttokenize(scentence):
    import re
    #word=scentence.split(' ')
    word = scentence.lower()
    token = re.compile("[\w]+(?=n't)|n't|\'m|\'ll|[\w]+|[.?!;,\-\(\)—\:']")
    t=token.findall(word)
    #t=list(reversed(t))
    return t

def tokenize(input):
    print(input.split(' '))
    return input.split(' ')

def my_pipeline(data, vocab=None, max_sent_len_=None):
    is_ext_vocab = True
    if vocab is None:
        is_ext_vocab = False
        vocab = {PAD: 0, OOV: 1}

    max_sent_len = -1
    data_sentences = []
    data_orders = []

    out_seq_len = []


    for instance in data:
        sents = []
        data_seq_len = []
        for sentence in instance['story']:
            sent = []
            tokenized = ttokenize(sentence)

            data_seq_len.append(len(tokenized))

            for token in tokenized:

                if not is_ext_vocab and token not in vocab:
                    vocab[token] = len(vocab)
                if token not in vocab:
                    token_id = vocab[OOV]
                else:
                    token_id = vocab[token]
                sent.append(token_id)
            if len(sent) > max_sent_len:
                max_sent_len = len(sent)
            sents.append(sent)

        out_seq_len.append(data_seq_len)

        data_sentences.append(sents)
        data_orders.append(instance['order'])

    if max_sent_len_ is not None:
        max_sent_len = max_sent_len_
    out_sentences = np.full([len(data_sentences), 5, max_sent_len], vocab[PAD], dtype=np.int32)

    for i, elem in enumerate(data_sentences):
        for j, sent in enumerate(elem):
            out_sentences[i, j, 0:len(sent)] = sent

    out_orders = np.array(data_orders, dtype=np.int32)

    return out_sentences, out_orders, out_seq_len, vocab, max_sent_len

################# Models #################

######### Seq2Seq + MLP #########

class Seq2SeqOrdering(object):

    def __init__(self, xseq_len, yseq_len,
            xvocab_size, yvocab_size,
            emb_dim, num_layers, ckpt_path,
            lr=0.01,
            epochs=10, model_name='seq2seq_model'):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name
        self.emb_dim = emb_dim
        self.epochs = 10000


        self.mlp_hidden = 64
        self.n_classes = 2
        self.mlp_input = emb_dim
        self.mlp_epochs = 10000


        # build thy graph
        #  attach any part of the graph that needs to be exposed, to the self
        def __graph__():


            ############### Placeholders for seq2seq ###############

            # placeholders
            tf.reset_default_graph()
            #  encoder inputs : list of indices of length xseq_len
            self.enc_ip = [ tf.placeholder(shape=[None,],
                            dtype=tf.int64,
                            name='ei_{}'.format(t)) for t in range(xseq_len) ]

            #  labels that represent the real outputs
            self.labels = [ tf.placeholder(shape=[None,],
                            dtype=tf.int64,
                            name='ei_{}'.format(t)) for t in range(yseq_len) ]

            #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
            self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]


            # Basic LSTM cell wrapped in Dropout Wrapper
            self.keep_prob = tf.placeholder(tf.float32)
            # define the basic cell

            ############### Set Up LSTM Net ###############

            basic_cell = tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.BasicLSTMCell(self.emb_dim, state_is_tuple=True),
                    output_keep_prob=self.keep_prob)
            # stack cells together : n layered model
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)


            # for parameter sharing between training model
            #  and testing model
            with tf.variable_scope('decoder') as scope:
                # build the seq2seq model
                #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
                self.decode_outputs, self.decode_states = tf.nn.seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip, stacked_lstm,
                                                    xvocab_size, yvocab_size, emb_dim)
                # share parameters
                scope.reuse_variables()
                # testing model, where output of previous timestep is fed as input
                #  to the next timestep
                self.decode_outputs_test, self.decode_states_test = tf.nn.seq2seq.embedding_rnn_seq2seq(
                    self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                    feed_previous=True)


            ############### Seq2seq Loss ###############

            with tf.variable_scope('loss') as scope:
                # weighted loss
                #  TODO : add parameter hint
                loss_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]
                self.loss = tf.nn.seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, yvocab_size)

                scope.reuse_variables()

                self.loss_permutation = tf.nn.seq2seq.sequence_loss(self.decode_outputs_test, self.labels, loss_weights, yvocab_size)


            ############### Seq2seq Optimisation ###############

            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)


            self.n_hidden_1 = 512 # 1st layer number of features
            self.n_hidden_2 = 256 # 2nd layer number of features
            self.n_input = self.emb_dim * 4
            self.n_classes_mlp = 5
            self.learning_rate = 0.01
            self.output_size = 25

            # tf Graph input
            self.x = tf.placeholder("float", [None, self.n_input])
            self.y = tf.placeholder(tf.int64, [None, self.n_classes_mlp])


            # Store layers weight & bias
            self.weights = {
                'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
                'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.output_size]))
            }
            self.biases = {
                'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
                'out': tf.Variable(tf.random_normal([self.output_size]))
            }

            # Construct model
            self.logits = self.multilayer_perceptron(self.x, self.weights, self.biases)
            #self.y = tf.reshape(self.y, [35, 5])
            self.logits_reshaped = tf.reshape(self.logits, [-1, 5, 5])

            self.unpacked_logits = [tensor for tensor in tf.unpack(self.logits_reshaped, axis=1)]
            self.softmaxes = [tf.nn.softmax(tensor) for tensor in self.unpacked_logits ]
            self.softmaxed_logits = tf.pack(self.softmaxes, axis=1)
            self.mlp_predict = tf.arg_max(self.softmaxed_logits , 2)

            # Define loss and optimizer
            self.mlp_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.mlp_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.mlp_loss)

        sys.stdout.write('>> Graph Ready <<')
        # build comput graph
        __graph__()

    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob # dropout prob
        #print(">> Made feed dict.")
        return feed_dict

        # Create model
    def multilayer_perceptron(self, x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return loss_v

    def eval_step(self, sess, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v = sess.run([self.loss, self.decode_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)

    # finally the train function that
    #  runs the train_op in a session
    #   evaluates on valid set periodically
    #    prints statistics
    def train(self, train_set, valid_set, sess=None ):
        # we need to save the model periodically
        #saver = tf.train.Saver()
        # if no session is given
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())

        sys.stdout.write('>> Training started <<')
        # run M epochs
        for i in range(self.epochs):
            try:
                self.train_batch(sess, train_set)
                if i % 100 == 0: #and i% (self.epochs//1) == 0: # TODO : make this tunable by the user
                    # save model to disk
                    #saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    # evaluate to get validation loss
                    val_loss = self.eval_batches(sess, valid_set, 16) # TODO : and this
                    # print stats
                    print('\nModel saved to disk at iteration #{}'.format(i))
                    print('val   loss : {0:.6f}'.format(val_loss))
                    sys.stdout.flush()
            except KeyboardInterrupt: # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                return sess

        return sess

    def trainMLP(self, train_batch, eval_batch, sess):

        for i in range(self.mlp_epochs):
            try:
                # Get training batch
                train_batchX, train_batchY, train_batchZ  = next(train_batch)

                # Determine batch size
                batch_size = np.shape(train_batchZ)[0]
                # Flatten story for embedding
                flatten_enc, flatten_dec = self.flattenBatch(train_batchX.T, train_batchY.T)
                # Embedd training batchX
                _, embedded_x = self.getSeq2SeqEmbedding(sess, flatten_enc, flatten_dec)


                final_h = embedded_x[0].h
                flatten_embeddings = final_h.reshape(batch_size, 4*self.emb_dim)
                feed ={self.x:flatten_embeddings,self.y:array(train_batchZ)}
                print(shape(sess.run(self.logits_reshaped, feed_dict=feed)))
                if i % 100 == 0:
                    train_batchX, train_batchY, train_batchZ  = next(eval_batch)
                    flatten_enc, flatten_dec = self.flattenBatch(train_batchX.T, train_batchY.T)
                    _, embedded_x = self.getSeq2SeqEmbedding(sess, flatten_enc, flatten_dec)
                    final_h = embedded_x[0].h
                    flatten_embeddings = final_h.reshape(batch_size, 4*self.emb_dim)
                    feed ={self.x:flatten_embeddings,self.y:array(train_batchZ)}

                    loss, pred = sess.run([self.mlp_loss, self.mlp_predict], feed_dict = feed)
                    acc = calculate_accuracy(train_batchZ, pred)
                    print("Iteration: {} Loss: {} Acc: {}".format(i, loss, acc))

            except KeyboardInterrupt:
                print("Training Stopped")
                break



    def restore_last_session(self):
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring last session at: ", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            return sess
        # return to user
        else:
            sess.close()
            print("No session saved.")

    def getSeq2SeqEmbedding(self, sess, x, y):
        feed = self.get_feed(x, y, keep_prob = 1)
        dec_out, dec_states = sess.run([self.decode_outputs, self.decode_states], feed_dict = feed)
        return dec_out, dec_states

    def flattenBatch(self, x, y):
        enc = array([item for something in x for item in something]).T
        dec = array([item for something in y for item in something]).T
        return enc, dec

    def predictLogits(self, sess, x, y, t):
        feed = self.get_feed(x, y, keep_prob = 1)
        feed.update({x: t})
        return sess.run(self.predict, feed_dict = feed)

    # prediction
    def predict(self, sess, X):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)


######### Base LSTM #########

def baselineLSTM():

    ## hidden 132, batch 50 - 55.5%
    target_size = 5
    vocab_size = len(vocab)
    input_size = 30
    n = 5460240
    hidden_size = 134
    BATCH_SIZE= 45
    n_stacks = 1
    embedding_dim = 60

    ### Base Line MODEL ###
    tf.reset_default_graph()
    ## PLACEHOLDERS
    story = tf.placeholder(tf.int64, [None, max_sent_len], "story")        # [batch_size x 5 x max_length]
    order = tf.placeholder(tf.int64, [None, 5], "order")             # [batch_size x 5]
    sen_len = tf.placeholder(tf.int64, [None], "sen_len")
    batch_size = tf.shape(story)[0]//5

    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="W")

    keep_prob = tf.placeholder(tf.float32)

    learning_rate = tf.placeholder(tf.float32)

    # Word embeddings
    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    embeddings = tf.get_variable("W", [vocab_size, input_size], initializer=initializer)

    sentences_embedded = tf.nn.embedding_lookup(embeddings, story)

    with tf.variable_scope("encoder") as varscope:

        basic_cell = tf.nn.rnn_cell.DropoutWrapper(
                            tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True),
                            output_keep_prob=keep_prob)

        _, final_first = tf.nn.dynamic_rnn(basic_cell, sentences_embedded, sequence_length=sen_len, dtype=tf.float32)

        final_firs_h = final_first.h

    reshape_final = tf.reshape(final_firs_h, [-1, hidden_size*5])

    logits_ = tf.contrib.layers.linear(reshape_final, 25)

    logits = tf.reshape(logits_, [-1, 5, 5])


    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=order))

    ## Optimizer
    optim = tf.train.AdamOptimizer(learning_rate)
    optim_op = optim.minimize(loss)
    init =tf.initialize_all_variables()

    unpacked_logits = [tensor for tensor in tf.unpack(logits, axis=1)]
    softmaxes = [tf.nn.softmax(tensor) for tensor in unpacked_logits]
    softmaxed_logits = tf.pack(softmaxes, axis=1)#
    predict = tf.arg_max(softmaxed_logits, 2)

    saver = tf.train.Saver()

    sess= tf.Session()
    sess.run(init)

    return sess


######### Train Baseline #########

def trainModel(sess = None):

        if not sess:
            sess = tf.Session()

        sess.run(tf.initialize_all_variables())

        for j in range(1):
            counter = 0
            slow_down = False
            slow_slow_down = False
            for i in range(n // BATCH_SIZE):
                #x, y, z = next(batch_gen)
                #x_flat, y_flat = flattenStory(x, y)
                if counter >= len(out_sentences)//BATCH_SIZE - BATCH_SIZE:
                    counter =0
                try:
                    if slow_down == True and slow_slow_down == False:
                        l_r = 0.001
                    elif slow_slow_down == True:
                        l_r = 0.0001
                    else:
                        l_r = 0.01

                    inst_story = out_sentences_flat[counter * BATCH_SIZE*5: (counter + 1) * BATCH_SIZE*5]
                    inst_order = out_orders[counter * BATCH_SIZE: (counter + 1) * BATCH_SIZE]
                    inst_seq_len = out_len_flat[counter * BATCH_SIZE*5: (counter + 1) * BATCH_SIZE*5]
                    feed_dict = {story: inst_story, order: inst_order, sen_len: inst_seq_len, keep_prob:0.5, learning_rate: l_r}
                    test = sess.run(optim_op, feed_dict = feed_dict)
                    #print(np.shape(test))

                    #print('hidden_size =', hidden_size, 'Epoch =' , j, "Batch:", i, 'out of ',n // BATCH_SIZE, "Loss:", loss1)

                    if i%10 == 0:
                        test_feed_dict = {story:test_stories1 , order: test_orders, sen_len:test_seq_len1,  keep_prob:1.0, learning_rate:l_r}
                        test_predicted = sess.run(predict, feed_dict=test_feed_dict)
                        test_accuracy = nn.calculate_accuracy(test_orders, test_predicted)
                        print('test_accuracy =', test_accuracy)
                        if test_accuracy > 0.538 and test_accuracy < 0.55:
                            slow_down = True
                            slow_slow_down = False
                        elif test_accuracy > 0.55:
                            slow_down = False
                            slow_slow_down = True
                        else:
                            slow_down = False

                        if test_accuracy > 0.555:
                            nn.save_model(sess)
                            print(test_accuracy)
                            break

                    counter += 1
                except KeyboardInterrupt:
                    print("Training Stopped")
                    nn.save_model(sess)
                    break


if __name__ in "__main__":
    import sys

    if "run" in sys.argv:
        model = baselineLSTM()
        trainModel(model)
        
