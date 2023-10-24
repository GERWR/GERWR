import cos_sim as cs
f = open("./data/nodes_example.txt", "r")
name_dic = {}
for line in f.readlines():
    name, id, type = line.strip().split('\t')
    name_dic[id] = name


def calSimVecs(inFile, outFile1, outFile2):
    fr = open(inFile, 'r')
    fw1 = open(outFile1, 'w')
    fw2 = open(outFile2, 'w')
    fw1.truncate()
    fw2.truncate()
    # 这两个数目随时更改
    geneNum = 1052
    srnaNum = 1423
    dic = {}
    for line in fr:
        arr = line.strip().split(' ')
        arr1 = [float(i) for i in arr]
        dic[int(arr1[0])] = arr1[1: ]
    dicNum = len(dic)

    for i in range(0, geneNum):
        if i not in dic: continue
        for j in range(i + 1, geneNum):
            if j not in dic: continue
            v1 = dic[i]
            v2 = dic[j]
            sim = cs.cos(v1, v2)
            fw1.write(name_dic[str(i)] + '\t' + name_dic[str(j)] + '\t' + str(round(float(sim), 4)) + '\n')

    for i in range(geneNum + 1, srnaNum):
        if i not in dic: continue
        for j in range(i + 1, srnaNum):
            if j not in dic: continue
            v1 = dic[i]
            v2 = dic[j]
            sim = cs.cos(v1, v2)
            fw2.write(name_dic[str(i)] + '\t' + name_dic[str(j)] + '\t' + str(round(float(sim), 4)) + '\n')
    fr.close()
    fw1.flush()
    fw1.close()
    fw2.flush()
    fw2.close()


pre = '../data/004/'
calSimVecs(pre + 'emb.txt', pre + 'gene.txt', pre + 'srna.txt')

