import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import json
import csv
import pandas as pd
import itertools
import random
import DL.config as config
from sklearn.utils import shuffle
import argparse


# sys.path.append("./DL/")
# Pairs of samples composed of different optimization levels of the same function
# are considered positive samples
# For example:
# function:_a-o0 _a-o1 _a_o2..
# ->positive samples:(_a-o0,_a-o1),(_a-o0,_a-o2).., (_a-o1,_a-o2),(_a-o1,_a-o3)..
def make_dataset_json_True_graph(dst_path, sr_path, total):
    count = 0
    with open(dst_path, 'w', newline='')as f1:
        ##构建语料库的csv文件
        # f_corp=open("../corpus_functions_asm.csv", mode='a+')
        # f_corpus=csv.writer(f_corp)

        headers = ['index', 'f1_child', 'f1_blocks', 'f2_child', 'f2_blocks', 'eq']  # 定义表头(每一列的内容)
        f_csv = csv.writer(f1)  # 设置写的对象
        f_csv.writerow(headers)  # 写入 表头

        with open(sr_path) as file_obj:
            numbers = json.load(file_obj)

        for i in numbers:
            # print("i: ",i)##函数
            # print("numbers[i]",numbers[i])##不同优化等级
            A = []
            B = []
            C = []
            D = []
            for j in numbers[i]:

                if len(str(numbers[i][j][0]).split('.')) <= 5:
                    continue

                A.append(str(numbers[i][j][0]).replace('.', ' '))

                B.append(numbers[i][j][1])

                tempp = numbers[i][j][2]

                for ii in range(0, len(tempp)):
                    for jj in range(0, len(tempp[ii])):
                        tempp[ii][jj] = int(tempp[ii][jj])

                tempD = numbers[i][j][3]
                for id1 in range(0, len(tempD)):
                    tempD[id1] = str(tempD[id1]).replace('.', ' ')
                D.append(tempD)

                C.append(tempp)

            ## Build a sample between different optimization levels of a function
            # for elem1 in range(0, len(A)):
            #     # f_corpus.writerows([[A[elem1]]])# add to corpus
            #
            #     for elem2 in range(elem1 + 1, len(A)):
            #         rows = [  #
            #             [count, B[elem1], D[elem1], B[elem2], D[elem2], 1]
            #
            #         ]
            #
            #         f_csv.writerows(rows)
            #         count = count + 1
            #         # c=c+1
            #         if count >= total:
            #             return count

            # Build a sample between two optimization levels of only one functionn
            if(len(A)>1):
                # print(len(A))
                range_len = [i for i in range(0,len(A))]
                # print(range_len)
                fun1, fun2 = random.sample(range_len, 2)

                rows = [  #
                    [count, B[fun1], D[fun1], B[fun2], D[fun2], 1]

                ]

                f_csv.writerows(rows)  # 
                count = count + 1
                # c=c+1
                if count >= total:
                    return count

    return count


# Different functions are considered as negative samples
# For example:
# function:_a-o0 _a-o1.., _b_o0 _b_o1 -b_o2..
# ->negative samples:(_a-o0,_b-o0),(_a-o0,_b-o1).., (_a-o1,_b-o2)..
# Note: In order to maintain the proportion between positive and negative samples
# and ensure the universality of negative samples,negative samples are sampled instead of traversing
# the entire JSON file like positive sample construction
def make_dataset_json_False_graph(dst_path, sr_path, c):
    count = 0  ##保证正负样本均衡
    temp11 = c


    with open(dst_path, 'a+', newline='')as f1:

        f_csv = csv.writer(f1)  # 设置写的对象
        # f_csv.writerow(headers)  # 写入 表头
        with open(sr_path) as file_obj:
            numbers = json.load(file_obj)
        # print("numbers: ",numbers)
        L = list()
        for i in numbers:
            L.append(numbers[i])  # 将字典放在LIST存储，方便索引
        # print(L)
        # print(len(L))
        for i in range(0, len(L)):  # 第一个函数
            for o in L[i]:  ##第一个函数的某个优先级 o->>01 or 02 or 0s---
                A = str(L[i][o][0]).replace('.', ' ')
                D1 = L[i][o][1]

                # E1 =L[i][o][2]
                tempp = L[i][o][2]
                for ii in range(0, len(tempp)):
                    for jj in range(0, len(tempp[ii])):
                        tempp[ii][jj] = int(tempp[ii][jj])
                blocks1 = L[i][o][3]

                for dtemp in range(0, len(blocks1)):
                    blocks1[dtemp] = str(blocks1[dtemp]).replace('.', ' ')

                C1 = 0  ##设置一个函数某个优先级出现的次数

                for ii in range(i + 1, len(L)):  # 第二个函数

                    if C1 >= 3:
                        break
                    for O in L[ii]:  ##第二个函数的某个优先级
                        C1 = C1 + 1
                        if C1 >= 3:
                            break

                        B = str(L[ii][O][0]).replace('.', ' ')
                        D2 = L[ii][O][1]


                        tempp1 = L[ii][O][2]
                        for iii in range(0, len(tempp1)):
                            for jj in range(0, len(tempp1[iii])):
                                tempp1[iii][jj] = int(tempp1[iii][jj])
                        #
                        blocks2 = L[ii][O][3]

                        for d2temp in range(0, len(blocks2)):
                            blocks2[d2temp] = str(blocks2[d2temp]).replace('.', ' ')

                        rows = [  #
                            [temp11, D1, blocks1, D2, blocks2, 0]

                        ]

                        f_csv.writerows(rows)  # 写入 多行记录
                        count = count + 1
                        temp11 = temp11 + 1
                        if count >= c:
                            return count

    return count


# Make datasets for MRR and Recall@K
def make_test_dataset_MRR_Recall_k(poosize, temp_path, sr_path):
    count = 0
    with open(temp_path, 'w', newline='')as f1:
        ##构建语料库的csv文件
        # f_corp=open("../corpus_functions_asm.csv", mode='a+')
        # f_corpus=csv.writer(f_corp)

        headers = ['index','function_name', 'f1_child', 'f1_blocks', 'f2_child', 'f2_blocks', 'eq']  # 定义表头(每一列的内容)
        f_csv = csv.writer(f1)  # 设置写的对象
        f_csv.writerow(headers)  # 写入 表头

        with open(sr_path) as file_obj:
            numbers = json.load(file_obj)

        L = list()
        for i in numbers:
            if (len(numbers[i])) >= 2:
                L.append([numbers[i],i])

        #selected_strings = random.sample(L, 32)  # random select 32 pair func

        L=shuffle(L)
        pool = 0
        for i in range(0, len(L)):  # first func
            # for i in range(0,32):
            A = []
            B = []
            C = []
            D = []
            for j in L[i][0]:
                if len(str(L[i][0][j][0]).split('.')) <= 5:  # inst <= 5 filter
                    continue

                A.append(str(L[i][0][j][0]).replace('.', ' '))

                B.append(L[i][0][j][1])

                tempp = L[i][0][j][2]

                for ii in range(0, len(tempp)):
                    for jj in range(0, len(tempp[ii])):
                        tempp[ii][jj] = int(tempp[ii][jj])

                tempD = L[i][0][j][3]
                for id1 in range(0, len(tempD)):
                    tempD[id1] = str(tempD[id1]).replace('.', ' ')
                D.append(tempD)

                C.append(tempp)

            #  random selcet different optim
            if (len(A)) < 2:
                continue
            elem1, elem2 = random.sample(range(0, len(A)), 2)

            rows = [  #
                [count, L[i][1], B[elem1], D[elem1], B[elem2], D[elem2], 1]
            ]
            f_csv.writerows(rows)  # 写入一行为克隆对的数据
            pool += 1
            count += 1
            if pool == poosize:
                break



def make_train_sets_graph(num):
    total = 0
    print("make train true")
    c = make_dataset_json_True_graph(config.train_csv_path, config.path_train_json, num / 2)
    a = c
    total = total + c
    print("make train false")
    c = make_dataset_json_False_graph(config.train_csv_path, config.path_train_json, c)
    b = c
    total = total + c
    A = pd.read_csv(config.train_csv_path)
    A = shuffle(A)
    A.to_csv(config.train_csv_path, index=False)
    print("True samples: ", a)
    print("False samples: ", b)
    print("total samples: ", total)
    return a, b, total


def make_test_sets_graph(num):
    total = 0
    print("make test true")
    c = make_dataset_json_True_graph(config.test_csv_path, config.path_test_json, num / 2)
    a = c

    total = total + c
    print("make test false")
    c = make_dataset_json_False_graph(config.test_csv_path, config.path_test_json, c, )
    b = c
    total = total + c
    A = pd.read_csv(config.test_csv_path)
    A = shuffle(A)
    A.to_csv(config.test_csv_path, index=False)

    print("True samples: ", a)
    print("False samples: ", b)
    print("total samples: ", total)
    return a, b, total


parser = argparse.ArgumentParser()
parser.add_argument('--train_samples', type=int, default=512,
                    help='number of train samples')
parser.add_argument('--test_samples', type=int, default=60,
                    help='number of test samples')

if __name__ == '__main__':
    args = parser.parse_args()
    print("=================================")
    print(args)
    print("=================================")

    A, B, Total = make_train_sets_graph(args.train_samples)
    A1, B1, Total1 = make_test_sets_graph(args.test_samples)
    print("------train--samples----")
    print("True samples: ", A)
    print("False samples: ", B)
    print("total samples: ", Total)

    print("------test--samples----")
    print("True samples: ", A1)
    print("False samples: ", B1)
    print("total samples: ", Total1)

    #make_test_dataset_MRR_Recall_k(32, config.MRR_Recall_k_temp_path, config.path_test_json)
