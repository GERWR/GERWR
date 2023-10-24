import numpy as np
import networkx as nx
import random
import gzip



class Graph():
	
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):    #由于采样时需要考虑前面2步访问过的顶点，所以当访问序列中只有1个顶点时，直接使用当前顶点和邻居顶点之间的边权作为采样依据。 当序列多余2个顶点时，使用文章提到的有偏采样
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length, output, p, q):  #随机游走
		# num_walks每个节点作为开始节点的次数
		# walk_length每次游走生成的节点序列的长度
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G

		nodes = list(G.nodes())
		print('Walk iteration:')

		# with gzip.open('walks/%s' %output,'w') as walks_file:
		with open(output, 'w') as walks_file:
			for walk_iter in range(num_walks):
			#for walk_iter in range(num_walks):
				print(str(walk_iter + 1), '/', str(num_walks))
				random.shuffle(nodes)
				#i = 0
				for node in nodes:
					#i+=1
					#if i == 3:break
					#print('start node', node)
					walk = self.node2vec_walk(walk_length=walk_length, start_node=node)
					#print('walk:', walk)
					for i in range(0,len(walk)):
						entity = walk[i]
						if i == len(walk)-1:
							walks_file.write(str(entity) + '\n')
						else:
							walks_file.write(str(entity) + ' ')

		'''
		#with gzip.open('walks/%s' %output,'w') as walks_file:
		with gzip.open(output, 'w') as walks_file:
			for walk_iter in range(num_walks):
				print (str(walk_iter+1), '/', str(num_walks))
				random.shuffle(nodes)
				for node in nodes:
					walk = self.node2vec_walk(walk_length=walk_length, start_node=node)
					last = walk[-1]
					for entity in walk:
						if entity == last:
							walks_file.write(''.join(entity.encode('utf-8')+'\n'))
						else:
							print(entity)
							walks_file.write(''.join(entity.encode('utf-8')))
							#walks_file.write(entity.encode('utf-8'))
							#walks_file.write(str(entity))
							walks_file.write(' ')

		'''
		return 

	def get_alias_edge(self, src, dst):   ##src是随机游走序列中的上一个节点，dst是当前节点
		#get_alias_edge方法返回的是在上一次访问顶点 t ，
		# 当前访问顶点为 v 时到下一个顶点 [公式] 的未归一化转移概率 Πvx=αpq(t,x)*wvx
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):  #构造采样表  ，分别生成alias_nodes和alias_edges
		#alias_nodes 存储每个顶点时决定下一次访问其邻接点时需要的alias表（不考虑当前顶点之前访问的顶点）
		#alias_edges存储着在前一个访问顶点为 t ，当前顶点为 v 时决定下一次访问哪个邻接点时需要的alias表。
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
         #self.alias_nodes仅仅用在序列中的start node选择下一个节点的时候，因为此时都是1，
		# self.alias_edges 用于其他节点来选择下一个节点的时候，self.get_alias_edge(edge[0], edge[1])中edge[0]是前一个节点，edge[1]是当前节点。

		#print alias_nodes

		#print alias_edges

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()
		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]