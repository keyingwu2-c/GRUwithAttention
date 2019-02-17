import tensorflow as tf
from ReadUtls import *
from Mdls import *
import os
import codecs

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_string('checkpoint_path', 'model\default', 'checkpoint path')
# tf.flags.DEFINE_string('input_file', 'data/poetry.txt', 'utf8 encoded text file')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_integer('batch_size', 50, 'batch size for each iter')
tf.flags.DEFINE_integer('n_iters', 100000, 'max iterations to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')

def readData(max_len=10):
    # pairs = readSougoCorpus()
    # saveData(pairs, 'sougoe2z')
    input_lang, output_lang, pairs = loadData('zho', 'eng', '../Translation/sougoe2z.npz')
    print(random.choice(pairs))
    print(random.choice(pairs))
    print(random.choice(pairs))
    return input_lang, output_lang, pairs


def train(input_lang, output_lang, pairs, from_restored=False):
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    # 要加上加载模型参数的功能！
    tpairs, vpairs = split_tra_val(pairs)
    # tpairs[1][0] = '一对 好人 。'
    # tpairs[0][1] = 'Nice couple.'
    # print(tpairs[1][0])
    # tpairs = [['一对 好人 。' 'Nice couple.']['一对 烂货 ！' 'Damaged goods.']['一些 英雄 ！' 'Some heroes!']]

    g = tbatch_generator(input_lang, output_lang, tpairs, FLAGS.batch_size, FLAGS.n_iters)
    vpairs = valarr_generator(input_lang, output_lang, vpairs)
    model = Seq2SeqModel(input_lang.n_words, output_lang.n_words)
    # model.load(FLAGS.checkpoint_path)
    # 使用load时要设置训练循环前step的初始值，且不能初始化参数，可能还要tf.get_default_graph()等，要另写一个model.train_restored吗？
    if from_restored:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        print(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        model.train_restored(global_step, g, vpairs,  FLAGS.n_iters, model_path, FLAGS.save_every_n, FLAGS.log_every_n)
    else:
        model.train(g, vpairs,  FLAGS.n_iters, model_path, FLAGS.save_every_n, FLAGS.log_every_n)
    return


def eval(input_lang, output_lang, vpairs):
    vpairs = valarr_generator(input_lang, output_lang, vpairs)
    print(np.array(vpairs).shape)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    model = Seq2SeqModel(input_lang.n_words, output_lang.n_words)
    model.load(FLAGS.checkpoint_path)
    model.evaluate(vpairs)
    return


def sample(input_lang, output_lang, sentence):
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    model = Seq2SeqModel(input_lang.n_words, output_lang.n_words)
    model.load(FLAGS.checkpoint_path)
    insnt_arr = np.array(indexesFromSentence(input_lang, sentence))
    arr = model.sample(insnt_arr, output_lang.n_words)
    print(sentenceFromArray(output_lang, arr))
    return

def main(_):
    print('Testing !')
    input_lang, output_lang, pairs = readData()

    train(input_lang, output_lang, pairs, from_restored=True)

    # word2index没有EOS,SOS
    # sample(input_lang, output_lang, '对不起 彻底 没戏 。')
    # 先改好训练和验证，训练轮数多验证没问题时再看sample结果怎样！

    # tpairs, vpairs = split_tra_val(pairs) # 仅可用于调试，这样是不对的。
    # eval(input_lang, output_lang, vpairs)

    # 按道理sample不应该必须先读入训练数据集的，想想有没得改
if __name__ == "__main__":
    tf.app.run()


