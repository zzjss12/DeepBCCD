import pickle
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import argparse
import json
import DL.config as config
import csv
import pandas as pd
import copy
from DL.trans_format import test_data_process, ins_to_index
from sklearn.metrics import auc, roc_curve, recall_score, f1_score, accuracy_score, precision_score
from random import *
# Define the function to evaluate
# def eval_O(ebds, TYPE1, TYPE2):
#     funcarr1 = []
#     funcarr2 = []
#
#     for i in range(len(ebds)):
#         if ebds[i].get(TYPE1) is not None and not isinstance(ebds[i][TYPE1], int):
#             if ebds[i].get(TYPE2) is not None and not isinstance(ebds[i][TYPE2], int):
#                 ebd1, ebd2 = ebds[i][TYPE1], ebds[i][TYPE2]
#                 funcarr1.append(ebd1 / np.linalg.norm(ebd1))
#                 funcarr2.append(ebd2 / np.linalg.norm(ebd2))
#         else:
#             continue
#
#     ft_valid_dataset = FunctionDataset_Fast(funcarr1, funcarr2)
#     dataloader = tf.data.Dataset.from_tensor_slices((funcarr1, funcarr2)).batch(POOLSIZE)
#
#     SIMS = []
#     Recall_AT_1 = []
#
#     for anchor, pos in tqdm(dataloader):
#         for i in range(len(anchor)):
#             vA = anchor[i:i + 1]
#             sim = np.array(tf.matmul(vA, pos, transpose_b=True).numpy().squeeze())
#             y = np.argsort(-sim)
#             posi = 0
#             for j in range(len(pos)):
#                 if y[j] == i:
#                     posi = j + 1
#                     break
#             if posi == 1:
#                 Recall_AT_1.append(1)
#             else:
#                 Recall_AT_1.append(0)
#             SIMS.append(1.0 / posi)
#
#     print(TYPE1, TYPE2, 'MRR{}: '.format(POOLSIZE), np.array(SIMS).mean())
#     print(TYPE1, TYPE2, 'Recall@1: ', np.array(Recall_AT_1).mean())
#     return np.array(Recall_AT_1).mean()


def eval(TYPE1, TYPE2,model):
    # DeepBCCD_net = DeepBCCD_nework(config)
    # build model
    # DeepBCCD_net.build_model()
    # self.model.load_weights(self.config.model_save_weights)
    #DeepBCCD_net.model.load_weights(config.model_save_weights)
    # DeepBCCD_net.model.summary()
    with open(config.path_test_json) as file_obj:
        numbers = json.load(file_obj)
    datasets=[]
    # funcarr2 = []
    SIMS = []
    Recall_AT_1 = []
    # print(numbers[0])
    count = 0
    Names=[]
    for name in numbers:
        Names.append(name)
    shuffle(Names)
    for i in Names:
        #print(i)
        # print(numbers[i])
        if TYPE1 in numbers[i]:
            if TYPE2 in numbers[i]:
                if len(str(numbers[i][TYPE1][0]).split(".")) <= 6 or len(str(numbers[i][TYPE2][0]).split("."))<=6:
                    continue
                #print(numbers[i][TYPE1])
                #print(numbers[i][TYPE1][2])
                #print("111")
                A=[count, i,numbers[i][TYPE1][3], numbers[i][TYPE1][1],numbers[i][TYPE2][3],numbers[i][TYPE2][1]]
                #B=[count, numbers[i][TYPE2][0], numbers[i][TYPE2][1]]
                #print(A)
                #print(B)
                #print()
                datasets.append(A)
                # funcarr2.append(B)
                count = count + 1
                #if count >33:
                #    break
        else:
            continue
    # count = 0
    #print(datasets)
    for i in tqdm(range(0, int(len(datasets) / config.poolsize))):
        if i % 100 == 0:
            print(i)
        dataset = datasets[i * config.poolsize:(i * config.poolsize + config.poolsize)]
        # temp2 = funcarr2[i * config.poolsize:(i * config.poolsize + config.poolsize)]
        # print(temp1)
        with open(config.MRR_Recall_k_temp_path, 'w', newline='') as f1:
            ##构建语料库的csv文件
            # f_corp=open("../corpus_functions_asm.csv", mode='a+')
            # f_corpus=csv.writer(f_corp)

            headers = ['index', 'function_name', 'f1_child', 'f1_blocks', 'f2_child', 'f2_blocks',
                       'eq']  # 定义表头(每一列的内容)
            f_csv = csv.writer(f1)  # 设置写的对象
            f_csv.writerow(headers)  # 写入 表头
            for j in range(0, len(dataset)):
                temp1 = dataset[j][2]
                for id1 in range(0, len(temp1)):
                    temp1[id1] = str(temp1[id1]).replace('.', ' ')
                #print(dataset[j][2])
                temp2 = dataset[j][4]
                for id2 in range(0, len(temp2)):
                    temp2[id2] = str(temp2[id2]).replace('.', ' ')
                row = [  #
                    [dataset[j][0], dataset[j][1], dataset[j][3],temp1,
                      dataset[j][5],temp2, 1]
                ]
                # count=count+1
                # print(row)
                # A.append(str(L[i][0][j][0]).replace('.', ' '))
                # B.append(L[i][0][j][1])

                f_csv.writerows(row)  # 写入一行为克隆对的数据
        mrr = 0
        recall_at_k = 0
        temp_df = pd.read_csv(config.MRR_Recall_k_temp_path, index_col=0)
        #print(df)
        # instrutions to index
        # print("start MRR_Recall@k datasets inst to index")
        
        temp_df = ins_to_index(temp_df, config)
        b1, g1, b2, g2, Y_test = test_data_process(temp_df, config)


        K = config.k  # 前K个最相关预测

        for idx in range(0, config.poolsize):
            b1_32 = copy.deepcopy(b1)
            g1_32 = copy.deepcopy(g1)
            b2_32 = b2
            g2_32 = g2
            Y_32 = Y_test
            for j in range(0, config.poolsize):
                b1_32[j] = b1_32[idx]
                g1_32[j] = g1_32[idx]

                Y_32[j] = j
            pred = model.predict([b1_32, g1_32, b2_32, g2_32], batch_size=1)
            #print(idx)
            # print(pred)
            pred = pred.flatten()
            #print(pred)

            loc = 0
            Max = 0
            max_loc = 0
            for index in range(0, len(pred)):
                if pred[index] > pred[idx]:
                    loc = loc + 1
                if pred[index] > Max:
                    Max = pred[index]
                    max_loc = max_loc
            rank1 = loc + 1
            mrr += 1 / rank1
            if pred[idx] == Max:
                recall_at_k += 1
            #print("rank: ", rank1)
            # idx_sorted = np.argsort(-pred)
            # # print("idx_sorted", idx_sorted)
            # idex_sorted_recall = idx_sorted[:K]
            # # print("idex_sorted_recall: ", idex_sorted_recall)
            # rank = np.where(idx_sorted == i)[0][0] + 1
            #
            # print("rank1: ", rank1)
            # print("rank: ", rank)
            # mrr += 1 / rank
            # if Y_32[i] in idex_sorted_recall:
            #     recall_at_k += 1
            #     # print("recall_at_k:", recall_at_k)

        recall_at_k /= config.poolsize
        mrr /= config.poolsize
        # print(mrr)
        # print(recall_at_k)
        SIMS.append(mrr)
        Recall_AT_1.append(recall_at_k)
        # print(SIMS)
    print(TYPE1, TYPE2, 'MRR{}: '.format(config.poolsize), np.array(SIMS).mean())
    print(TYPE1, TYPE2, 'Recall@1: ', np.array(Recall_AT_1).mean())
    # dataloader = tf.data.Dataset((funcarr1, funcarr2)).batch(POOLSIZE)
    # for anchor, pos in tqdm(dataloader):
    #     # for i in range(len(anchor)):
    #     print(anchor)
    #     print(pos)
    #     break
    # if numbers[i]

    # for i in range(len(numbers)):
    #     if ebds[i].get(TYPE1) is not None and not isinstance(ebds[i][TYPE1], int):
    #         if ebds[i].get(TYPE2) is not None and not isinstance(ebds[i][TYPE2], int):
    #             ebd1, ebd2 = ebds[i][TYPE1], ebds[i][TYPE2]
    #             funcarr1.append(ebd1 / np.linalg.norm(ebd1))
    #             funcarr2.append(ebd2 / np.linalg.norm(ebd2))
    #     else:
    #         continue


# Define the dataset class
# class FunctionDataset_Fast(tf.keras.utils.Sequence):
#     def __init__(self, arr1, arr2):
#         self.arr1 = arr1
#         self.arr2 = arr2
#         assert len(arr1) == len(arr2)
#
#     def __getitem__(self, idx):
#         return self.arr1[idx], self.arr2[idx]
#
#     def __len__(self):
#         return len(self.arr1)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="jTrans-FastEval")
#     parser.add_argument("--experiment_path", type=str, default='jTrans.pkl', help="experiment to be evaluated")
#     parser.add_argument("--poolsize", type=int, default=2, help="size of the function pool")
#     args = parser.parse_args()
#
#     POOLSIZE = args.poolsize

# eval('O0', 'O3')

# with open(args.experiment_path, 'rb') as ff:
#     ebds = pickle.load(ff)
# print(ebds)
# print(f'evaluating...poolsize={POOLSIZE}')
#
# eval_O(ebds, 'O0', 'O3')
# eval_O(ebds, 'O0', 'Os')
# eval_O(ebds, 'O1', 'Os')
# eval_O(ebds, 'O1', 'O3')
# eval_O(ebds, 'O2', 'Os')
# eval_O(ebds, 'O2', 'O3')
