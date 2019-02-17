import tensorflow as tf
import numpy as np
import time
import os



def pick_top_n(preds, ouvcbsize, top_n=1):
    p = np.squeeze(preds)
    # print(p.shape, ouvcbsize) # p第三个维度应该是每个词的概率，整个函数不适用于批解码词概率向量序列
    # print(p, np.argsort(p))
    # print(np.argsort(p)[:-top_n])
    p[np.argsort(p)[:-top_n]] = 0
    #print(p)
    p = p / np.sum(p)
    c = np.random.choice(ouvcbsize, 1, p=p)[0]
    return c


class Seq2SeqModel:
    def __init__(self, inptvcb_size, ouptvcb_size, batch_size=50, embedding_size=256, maxlen=10,
                 gru_size=256, train_keep_prob=0.9, learning_rate=0.001, grad_clip=5):
        self.inptvcb_size = inptvcb_size
        self.ouptvcb_size = ouptvcb_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.maxlen = maxlen
        self.source_sequence_length = np.array([self.maxlen for l in range(batch_size)])
        # 应该每一句输入句子都要获取一个source_sequence_length，默认值先设为maxlen
        self.target_sentence_length = np.array([self.maxlen for l in range(batch_size)])
        self.gru_size = gru_size
        self.train_keep_prob = train_keep_prob
        # 未使用
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        tf.reset_default_graph()
        self.build_inputs()
        self.build_seq2seq()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.encoder_inputs = tf.placeholder(tf.int32, shape=(
                self.batch_size, self.maxlen), name='encoder_inputs')
            #为什么一般要用time-major的形状？ 即self.maxlen放前面
            # self.decoder_inputs = tf.placeholder(tf.int32, shape=(
            #     self.batch_size, self.maxlen), name='decoder_inputs')
            self.targets = tf.placeholder(tf.int32, shape=(
                self.batch_size, self.maxlen), name='targets')
            # for standard translated sentence
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            with tf.device("/gpu:0"):
                encoder_embedding = tf.get_variable('encoder_embedding', [self.inptvcb_size, self.embedding_size])
                self.encoder_emb_inp = tf.nn.embedding_lookup(encoder_embedding, self.encoder_inputs)
                decoder_embedding = tf.get_variable('decoder_embedding', [self.ouptvcb_size, self.embedding_size])
                self.decoder_emb_inp = tf.nn.embedding_lookup(decoder_embedding, self.targets)

    def build_seq2seq(self):
        encCell = tf.nn.rnn_cell.GRUCell(num_units=self.gru_size, name='encGRU')
        # 单层GRU-RNN，如果多层就要用tf.nn.rnn_cell.MultiRNNCell(for循环)获取
        self.initial_state = encCell.zero_state(self.batch_size, tf.float32)
        self.enc_outputs, self.enc_finalstate = tf.nn.dynamic_rnn(encCell, self.encoder_emb_inp,
                                                                  initial_state=self.initial_state,
                                                                  sequence_length=self.source_sequence_length,
                                                                  time_major=False)

        decCell = tf.nn.rnn_cell.GRUCell(num_units=self.gru_size, name='decGRU')

        # attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=1, memory=self.enc_outputs,
        #                                                         memory_sequence_length=self.source_sequence_length)
        # 设memory_sequence_length则句子不够长会补0到设定的长度（这里即是max_len=10）
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.gru_size, memory=self.enc_outputs)

        decCell = tf.contrib.seq2seq.AttentionWrapper(decCell, attention_mechanism)

        # print(self.enc_finalstate.shape, type(self.enc_finalstate))
        self.decoder_initial_state = decCell.zero_state(self.batch_size, tf.float32).clone(
          cell_state=self.enc_finalstate)
        # print(type(self.decoder_initial_state))  AttentionWrapperState!
        helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_inp, self.target_sentence_length, time_major=False)
        # self.attention_states = tf.Variable(tf.zeros([1, self.maxlen, self.gru_size], tf.float32))

        self.projection_layer = tf.layers.Dense(self.ouptvcb_size, use_bias=False)

        decoder = tf.contrib.seq2seq.BasicDecoder(decCell, helper, self.decoder_initial_state, output_layer=self.projection_layer)


        self.dec_outputs, self.dec_finalstate, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

        self.logits = self.dec_outputs.rnn_output
        self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')


    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.ouptvcb_size)
            print(y_one_hot, self.logits)
            # y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            target_weights = tf.sequence_mask(self.target_sentence_length, self.maxlen, dtype=tf.float32)
            #self.loss = tf.reduce_mean(loss)
            self.train_loss = (tf.reduce_sum(crossent * target_weights) / self.batch_size)


    def build_optimizer(self):
        # 使用clipping gradients
        params = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.train_loss, params), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, params))


    def train(self, pair_generator, val_arrset, n_iters, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            new_state = sess.run(self.initial_state)
            previous_eset_loss = 0
            for x, y in pair_generator:
                step += 1
                start = time.time()
                feed = {self.encoder_inputs: x,
                        self.targets: y,
                        self.initial_state: new_state}
                # You must feed a value for placeholder tensor 'inputs/decoder_inputs' with dtype int32 and shape [50,10]
                # print(type(x), type(y)) 都是<class 'numpy.ndarray'>
                # print(x.shape, y.shape) 都是(50, 10)
                snglpr_loss, _ = sess.run([self.train_loss, self.optimizer], feed_dict=feed)
                # snglpr_loss, new_state, _ = sess.run([self.train_loss,
                #                                      self.dec_finalstate,
                #                                      self.optimizer],
                #                                     feed_dict=feed)
                # snglpr_loss, new_state, encoder_fst, decoder_intst = sess.run([self.train_loss, self.dec_finalstate,
                #                                                     self.enc_finalstate, self.decoder_initial_state],
                #                                                    feed_dict=feed)
                end = time.time()
                # print(snglpr_loss)
                # print(encoder_fst.shape, decoder_intst.time)
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, n_iters),
                          'loss: {:.4f}... '.format(snglpr_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                    # print('step: {}/{}... '.format(step, n_iters),
                    #   '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                    # 验证集损失只用于终止条件
                    # xe, ye = val_arrset[:, 0], val_arrset[:, 1]
                    # efeed = {self.encoder_inputs: xe,
                    #             self.targets: ye,
                    #             self.initial_state: new_state}
                    eset_loss = self.evaluate(val_arrset)
                    val_loss_diff = eset_loss - previous_eset_loss
                    print('val_loss: {:.4f}...'.format(eset_loss))
                    if val_loss_diff > 0.1:
                        print('Early stop by val_loss_diff: {:.4f}...'.format(val_loss_diff))
                        break
                    previous_eset_loss = eset_loss
                if step >= n_iters:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)


    def train_restored(self, global_step, pair_generator, val_arrset, n_iters, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer()) # 这个会把训练出来的参数清零吗？
            step = int(global_step)
            new_state = sess.run(self.initial_state)
            previous_eset_loss = 999
            for x, y in pair_generator:
                step += 1
                start = time.time()
                feed = {self.encoder_inputs: x,
                        self.targets: y,
                        self.initial_state: new_state}
                snglpr_loss, _ = sess.run([self.train_loss, self.optimizer], feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, n_iters),
                          'loss: {:.4f}... '.format(snglpr_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                    eset_loss = self.evaluate(val_arrset)
                    val_loss_diff = previous_eset_loss - eset_loss
                    print('val_loss: {:.4f}...'.format(eset_loss))
                    if val_loss_diff < 0.01:
                        print('Early stop by val_loss_diff: {:.4f}...'.format(val_loss_diff))
                        break
                    previous_eset_loss = eset_loss
                if step >= n_iters:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)


    def evaluate(self, val_arrset ):
        # 不用另开一个独立的session，加载保存的模型到一个新的模型实例就可以了。
        val_arrset = np.array(val_arrset)[:50]
        # print(np.array(val_arrset).shape)
        sess = self.session
        new_state = sess.run(self.initial_state)
        xe, ye = val_arrset[:, 0], val_arrset[:, 1]
        # print(xe.shape, ye.shape)
        efeed = {self.encoder_inputs: xe,
                 self.targets: ye,
                 self.initial_state: new_state}
        eset_loss = sess.run(self.train_loss, feed_dict=efeed)
        return eset_loss
    # evaluate应该一次喂整个训练集才对的，目前的实现是喂训练集的前50对样本！

    def sample(self, insnt_arr, ouvcbsize):
        sess = self.session
        new_state = sess.run(self.initial_state)
        # preds = np.ones((ouvcbsize,))
        test_input = np.zeros((50, 10)) #把句子的词序填入第一行试试！
        reference = np.ones((50, 10))
        # print(insnt_arr, test_input[0])
        for index in range(len(insnt_arr)):
            test_input[0][index] = insnt_arr[index]

        feed = {self.encoder_inputs: test_input,
                self.targets: reference,
                self.initial_state: new_state}
        # sample时不计算损失，targets只是填饱，不影响proba_prediction计算。
        preds, new_state = sess.run([self.proba_prediction, self.dec_finalstate],
                                    feed_dict=feed)
        output_sentence = []
        # output_sentence = pick_top_n(preds[0][0], ouvcbsize)
        # print(preds[0][0], output_sentence, output_sentence.shape)
        # print(preds[0][3], pick_top_n(preds[0][3], ouvcbsize), output_sentence.shape)
        for i in range(self.maxlen):
            v = pick_top_n(preds[0][i], ouvcbsize)
            output_sentence.append(v)
        return np.array(output_sentence)


    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        # self.saver.restore(self.session, tf.train.latest_checkpoint('./')) 后面再看一下这种方法
        print('Restored from: {}'.format(checkpoint))


if __name__=="__main__":
    vacab_size = 5
    probabilities = [0.01, 0.05, 0.05, 0.5, 0.39]
    for i in range(5):
        c = pick_top_n(probabilities, vacab_size, 3)
        print(c)








