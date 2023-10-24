# -*- coding: utf-8 -*-
# simulates the entity2vec walks and serialize them to a txt file

import argparse
import networkx as nx
from node2vec import Graph
from datetime import datetime

def parse_args():

	# Parses the node2vec arguments.   解析node2vec参数
	parser = argparse.ArgumentParser(description="Run node2vec.")
	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist', help='Input graph path')
	parser.add_argument('--output', nargs='?', default='walks', help='walks file name')
	parser.add_argument('--walk-length', type=int, default=10, help='Length of walk per source. Default is 10.')
	parser.add_argument('--num-walks', type=int, default=40, help='Number of walks per source. Default is 40.')
	parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')
	parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')
	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)
	parser.add_argument('--directed', dest='directed', action='store_true',
						help='Graph is (un)directed. Default is directed.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=True)
	return parser.parse_args()

def read_graph():

	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),))
	else:
		G = nx.read_edgelist(args.input, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def run_walk(args):
	# Pipeline for representational learning for all nodes in a graph.   图形中所有结点表示学习
	start_time = datetime.now()
	print('----input parameters----')
	print('p=', args.p)
	print('q=', args.q)
	print('num_walks=', args.num_walks)
	print('walk_length=', args.walk_length)
	print('directed=', args.directed)
	print('weighted=', args.weighted)
	print('----run program----')
	nx_G = read_graph()   ##从文本中读取图
	print('read graph')
	G = Graph(nx_G, args.directed, args.p, args.q)#args.directed bool型用来标识
    #（有向图，无向图)， args.p，args.q分别是参数p和q, 这一步是生成一个图对象
	print('defined G')
	G.preprocess_transition_probs()#调用node2vec函数  #生成每个节点的转移概率向量形成G‘
	print('preprocessed')
	G.simulate_walks(args.num_walks, args.walk_length, args.output, args.p, args.q)#随机游走
	print('defined walk')
	end_time = datetime.now()
	print('run time: ', (end_time - start_time))


def walk_main(args):
	pre = '../data/004/'
	args.input = pre + 'edges_example.txt'
	args.output = pre + 'walks.txt'# 每次游走的结点序列
	args.num_walks = 40 		# num_walks每个节点作为开始节点的次数   （形成num_walks长度为walk_length的walk)
		                        # walk_length每次游走生成的节点序列的长度
	args.walk_length = 40
	args.directed = False
	args.weighted = False
	args.p = 0.25 # 较大不回头
	args.q = 2 # >1 向外探索
	run_walk(args)


if __name__ == '__main__':
	args = parse_args()
	walk_main(args)
