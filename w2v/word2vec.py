# from  keras_gcn import GraphConv
# from keras.callbacks import TensorBoard
import argparse
import gensim
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import DL.config as config


class Sentences(object):
    """
    生成gensim sentence需要的格式, 可在这个类里进行预处理 ，可迭代对象,
    """

    def __init__(self, folder_path):
        self.index = 0
        self.folder_path = folder_path

    def __iter__(self):

        for line in open(self.folder_path, encoding="utf-8"):
            if line == '\n':
                continue
            line = line.strip("\"")
            line = line.strip()
            line = line.strip('\n')

            content = line.split('.')

            Temp = []
            for elem in content:
                elem = elem.strip()
                if elem == "\"":
                    continue

                Temp.append(elem)

            yield Temp

        print("epoch: ", self.index)
        self.index = self.index + 1

    def __str__(self):
        return "It is a iter, create sentence"


parser = argparse.ArgumentParser()
parser.add_argument('--window', type=int, default=config.window,
                    help='feature dimension')
parser.add_argument('--vector_size', type=int, default=config.vector_size,
                    help='embedding dimension')
parser.add_argument('--min_count', type=int, default=config.min_count,
                    help='embedding network depth')
parser.add_argument('--sg', type=int, default=config.sg,
                    help='output layer dimension')
parser.add_argument('--workers', type=int, default=config.workers,
                    help='iteration times')
parser.add_argument('--sample', type=float, default=config.sample,
                    help='')
parser.add_argument('--w2v_epoch', type=int, default=config.w2v_epoch,
                    help='epoch number')
# parser.add_argument('--load_path', type=str, default=None,
#         help='path for model loading, "#LATEST#" for the latest checkpoint')
# parser.add_argument('--save_path', type=str,
#         default='./saved_model/graphnn-model', help='path for model saving')

if __name__ == '__main__':

    args = parser.parse_args()
    print("=================================")
    print(args)
    print("=================================")

    sentences = Sentences(config.corpus_path)

    window = args.window
    vector_size = args.vector_size
    min_count = args.min_count
    sg = args.sg
    workers = args.workers
    sample = args.sample
    epoch = args.w2v_epoch

    vec_path = config.w2v_save_path + "window-" + str(window) + "vector_size-" + str(vector_size) + "min_count-" + str(
        min_count) + "sg-" + \
               str(sg) + "workers-" + str(workers) + "sample-" + str(sample) + "epoch-" + str(epoch) + ".w2v"

    model = gensim.models.Word2Vec(sentences, window=window, vector_size=vector_size, min_count=min_count, sg=sg,
                                   workers=workers, sample=sample)
    model.train(sentences, total_examples=model.corpus_count, epochs=epoch, )
    model.wv.save_word2vec_format(vec_path, binary=True)

    print(model)
    # print(model.wv.key_to_index)
    X = model.wv.vectors
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    fig = plt.figure(figsize=(20, 8), dpi=200)
    plt.scatter(result[:, 0], result[:, 1], s=40, marker=".")
    words = list(model.wv.key_to_index)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=5)
    plt.show()
    fig.savefig("window" + str(window) + "vector_size" + str(vector_size) + "min_count" +
                str(min_count) + "sg" + str(sg) + "workers" + str(workers) + "sample" + str(sample) + "epoch" + str(
        epoch) + ".png")
