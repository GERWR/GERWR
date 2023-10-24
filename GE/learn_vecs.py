import argparse
from gensim.models import Word2Vec
from datetime import datetime

def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="Run learn embeddings.")
    parser.add_argument('--input', nargs='?', default=' ',
                        help='Input walks path')
    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')
    parser.add_argument('--model', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')
    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    return parser.parse_args()

class Sentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        try:
            file_loc = self.fname
            for line in open(file_loc, mode='r'):
                line = line.rstrip('\n')
                words = line.split(" ")
                yield words
        except Exception:
            print("Failed reading file:")
            print(self.fname)

def learn_embeddings(walks):  #walks是随机游走生成的多个节点序列，被当做文本输入，调用Word2Vec模型，生成向量

    # Learn embeddings by optimizing the Skipgram objective using SGD.
    sentences = Sentences(walks)   #sentences是训练所需预料，可通过该方式加载，此处训练集为英文文本或分好词的中文文本
    model = Word2Vec(sentences, vector_size=args.dimensions, window=args.window_size, min_count=0,
                     workers=args.workers, epochs=args.iter, negative=25, sg=1)   # 训练skip-gram模型;
    print("defined model using w2v")
    model.wv.save_word2vec_format(args.output)   #通过该方式保存的模型，能通过文本格式打开，也能通过设置binary是否保存为二进制文件。但该模型在保存时丢弃了树的保存形式（详情参加word2vec构建过程，以类似哈夫曼树的形式保存词），所以在后续不能对模型进行追加训练

    model.wv.save(args.model)
    print("saved model in word2vec format")
'''
    #计算一个词的最近似的词：
    print(model.most_similar("1422", topn=3) ) # 计算与该 词最近似的词，topn指定排名前n的词

    #计算两个词的相似度：
   print(model.similarity("22", "50"))

    #获取词向量（有了这个不就意味着可以进行相关词语的加减等运算，虽然我不是太懂）：
    print(model['1272'])

    return
    '''

#word2vec函数参数：
#sentences：可以是一个list，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
 # sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
 #size：是指特征向量的维度，默认为100。大的size需要更多的训练数据, 但是效果会更好.推荐值为几十到几百。
 # window：表示当前词与预测词在一个句子中的最大距离是多少。Harris在1954年提出的分布假说(distributional
#hypothesis)指出， 一个词的词义由其所在的上下文决定。所以word2vec的参数中，窗口设置一般是5，而且是左右随机1 - 5（小于窗口大小）的大小，是均匀分布, 随机的原因应该是比固定窗口效果好，增加了随机性，个人理解应该是某一个中心词可能与前后多个词相关，也有的词在一句话中可能只与少量词相关（如短文本可能只与其紧邻词相关）。
 ###·  alpha: 是学习速率
#  seed：用于随机数发生器。与初始化词向量有关。
 #min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。该模块在训练结束后可以通过调用model.most_similar('电影',topn=10)得到与电影最相似的前10个词。如果‘电影’未被训练得到，则会报错‘训练的向量集合中没有留下该词汇’。
 # max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
 #sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
 #workers参数控制训练的并行数。
 #hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
# negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
 # cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（defau·t）则采用均值。只有使用CBOW的时候才起作用。
 # hashfxn： hash函数来初始化权重。默认使用python的hash函数
 # iter： 迭代次数，默认为5
 # trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
 # sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
 # batch_words：每一批的传递给线程的单词的数量，默认为10000。这里我认为是在训练时，控制一条样本（一个句子）中词的个数为10000，大于10000的截断；训练时依次输入batch_words的2*window（实际上不一定是2*window，因为代码内部还对[0,window]取了随机数）的词汇去训练词向量；另外，训练时候外层还有batch_size控制对样本的循环


def learn_embedding(args):

    # Pipeline for representational learning for all nodes in a graph.
    start_time = datetime.now()
    print('---- input parameters -----')
    print('iterations=', args.iter)
    print('window size=', args.window_size)
    print('dimensions=', args.dimensions)
    print('workers=', args.workers)
    print('--- run program ---')
    learn_embeddings(args.input)
    end_time = datetime.now()
    print('run time: ', (end_time-start_time))

def emb_main(args):
    pre = '../data/004/'
    args.input = pre + 'walks.txt'
    args.output = pre + 'emb.txt'
    args.model = pre + 'emb.model'
    args.dimensions = 128
    args.iter = 5  #迭代次数
    args.window_size = 10   #上下文分别的窗口尺寸（10*2个概率最大的词选）
    args.workers = 6   #workers参数控制训练的并行数。
    learn_embedding(args)  #walks是随机游走生成的多个节点序列，被当做文本输入，调用Word2Vec模型，生成向量

if __name__ == '__main__':
    args = parse_args()
    emb_main(args)


#walks中每一行（40个结点）表示一篇文章