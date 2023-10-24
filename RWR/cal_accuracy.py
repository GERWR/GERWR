# -*- coding: utf-8 -*-
from random import choice

def calAccuracy(train_test_file, results_file):
    precision = 0 # top在测试集里的/所有top
    recall = 0 # top在测试集里的/测试集所有数据
    test_dis_dic = {} # 测试里的dis-(gene1, gene2)
    train_dis_dic = {}  # 训练里的dis-(gene1, gene2)
    test_edge_set = set() # 统计有多少个不重复的测试数据
    train_edge_set = set()  # 统计有多少个不重复的训练数据
    # 得到测试数据
    with open(train_test_file, 'r') as fr:
        for line in fr:
            dis, gene, type = line.strip().split('\t')
            if type == 'test':
                test_edge_set.add(dis + '\t' + gene)
                if dis not in test_dis_dic:
                    test_dis_dic[dis] = set([gene])
                else:
                    test_dis_dic[dis].add(gene)
            elif type == 'train':
                train_edge_set.add(dis + gene)
    top_dis_dic = {} # dis及其top gene集合
    top_dis_score_dic = {} # dis+gene : score
    top = 200
    # 为了计算AUC
    score_not_in_test = []
    # 在预测结果当中，topN的 dis-（gene1，gene2...)
    with open(results_file, 'r') as f:
        now = ''
        count = 0
        for line in f:
            dis, gene, score = line.strip().split('\t')
            top_dis_score_dic[dis + gene] = score
            score_not_in_test.append(score)
            if dis != now:
                count = 0
                now = dis
                top_dis_dic[dis] = set([gene])
            else:
                count += 1
                if count >= top:
                    continue
                else:
                    top_dis_dic[dis].add(gene)
    #print(top_dis_dic)
    total_top = top * len(top_dis_dic)
    total_test = len(test_edge_set)
    top_gene_in_test = 0
    total_dis_gene = 0
    # 为了计算AUC
    score_in_test = []
    # 预测结果在测试集中的
    for dis, top_gene_set in top_dis_dic.items():
        if dis in test_dis_dic:
            total_dis_gene += len(test_dis_dic[dis])
        for top_gene in top_gene_set:
            #print(top_gene)
            if top_gene in test_dis_dic[dis]:
                top_gene_in_test += 1
                # for AUC
                score = top_dis_score_dic[dis + top_gene]
                del top_dis_score_dic[dis + top_gene]
                score_in_test.append(score)
                score_not_in_test.remove(score)
    precision =top_gene_in_test / float(total_top)
    recall = top_gene_in_test / float(total_dis_gene)
    #print("top_gene_in_test:  ",top_gene_in_test, "total_top:   ",total_top,"total_dis_gene:  ", total_dis_gene)
    print(pre,":  top=",top,"  precision:  "+str(round(precision, 2)) + '   ' +"recall:  " +str(round(recall, 2)))
   # print(round(2 * precision * recall / (precision + recall), 2))

    # score_not_in_test也不在训练集中的
    for edge, score in top_dis_score_dic.items():
        if edge in train_edge_set:
            print("不在训练集："+'iii'+"\t")
            score_not_in_test.remove(score)



    # 计算AUC
    if top !=0:
        len_score_in = len(score_in_test)
        len_score_not_in = len(score_not_in_test)
        #print("len_score_in:  "+str(len_score_in),"\t\t tlen_score_not_in:  "+str( len_score_not_in))
        n1 = 0
        n2 = 0
        N = 1000
        for i in range(N):
            if choice(score_in_test) > choice(score_not_in_test):
                n1 += 1
            elif choice(score_in_test) == choice(score_not_in_test):
                n2 += 1
        print("AUC: "+str(((n1 + 0.5 * n2) / N)))




for  i in range (0,5):
   pre = '../data/'+'00'+str(i)
   calAccuracy(pre + '/input.txt', pre + '/outPre.txt')
   print('********')
'''
auc = 0
for i in range(10):
    auc += calAccuracy(pre + '/input.txt', pre + '/outPre.txt')
print(auc, auc / 10)
auc = 0
for i in range(10):
    #auc += calAccuracy(pre + 'input.txt', pre + 'outPre.txt')
    auc += calAccuracy(pre + 'input.txt', pre + 's2v_embbeding_prediction_results.txt')
print(auc, auc / 10)
'''


#alAccuracy(pre + 'input.txt', pre + 'n2v_embbeding_prediction_results.txt')
