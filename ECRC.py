import time
import sys
from heapq import *
from collections import defaultdict
import itertools
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import laplacian
from random import choice, randint
from collections import defaultdict, deque
from functools import lru_cache

class Graph:

	def __init__(self, dataset):
		print(dataset + " is loading...")
		self.adj_left, self.adj_right= self.graph_list(dataset) 
		print("number of left_nodes" + str(len(self.adj_left)))
		print("number of right_nodes" + str(len(self.adj_right)))
		left_degree_sum = 0
		self.deg_left = {}
		self.deg_right = {}
		for u in self.adj_left:
			self.deg_left[u]=len(self.adj_left[u])
			left_degree_sum += self.deg_left[u]
		print("average_left_degree"+str(left_degree_sum/len(self.adj_left)))
		right_degree_sum=0
		for u in self.adj_right:
			self.deg_right[u] = len(self.adj_right[u])
			right_degree_sum+=self.deg_right[u]
		print("average_right_degree" + str(right_degree_sum/ len(self.adj_right)))
		m = (left_degree_sum+right_degree_sum) / 2
		print("number of edges" + str(m))
		self.core_number=self.core_decomposition()

	def graph_list(self,dataset):
		adj_left = {}
		adj_right = {}
		with open(dataset) as file:
			for line in file:
				line=line.split()
				u, v = int(line[0]),int(line[1])
				if u not in adj_left:
					adj_left[u] = set()
				if v not in adj_right:
					adj_right[v] = set()
				adj_left[u].add(v)
				adj_right[v].add(u)
		new_adj_left={}
		new_adj_right={}
		max_left=max(adj_left)+1
		for u in adj_left:
			new_adj_left[u]=set()
			for v in adj_left[u]:
				new_adj_left[u].add(v+max_left)
		for u in adj_right:
			new_adj_right[u+max_left]=set()
			for v in adj_right[u]:
				new_adj_right[u+max_left].add(v)
		return new_adj_left,new_adj_right
		
	def core_decomposition(self):  # use heapq
		core_number = {}
		max_core = 0
		flag = set()  #To determine if it has been popped out of the heap
		degree_list = []
		deg={}
		for u in self.adj_left:
			deg[u]=self.deg_left[u]
			heappush(degree_list, (deg[u], u))
		for u in self.adj_right:
			deg[u]=self.deg_right[u]
			heappush(degree_list, (deg[u], u))
		while degree_list:
			x, v = heappop(degree_list)
			if (x, v) in flag:
				continue
			if x > max_core:
				max_core = x
			core_number[v] = max_core
			if v in self.adj_left:
				for u in self.adj_left[v]:
					if u not in core_number:
						flag.add((deg[u], u))
						deg[u] = deg[u] - 1
						heappush(degree_list, (deg[u], u))
			if v in self.adj_right:
				for u in self.adj_right[v]:
					if u not in core_number:
						flag.add((deg[u], u))
						deg[u] = deg[u] - 1
						heappush(degree_list, (deg[u], u))
		return core_number



	def c2hn_H(self,p,q):
		self.p=p
		self.q=q
		cost_left = sum(deg * deg for deg in self.deg_left.values())
		cost_right = sum(deg * deg for deg in self.deg_right.values())
		self.H = {}
		if cost_left > cost_right:  # left
			self.p1, self.q1 = self.p, self.q
			sorted_neig={}
			for u in self.adj_right:
				sorted_neig[u]=[]
				tmp_dict={}
				for v in self.adj_right[u]:
					tmp_dict[v]=self.core_number[v]
				sorted_neig[u]=sorted(tmp_dict,key=lambda key: tmp_dict[key],reverse=True)
			for u in self.adj_left:
				C = defaultdict(int)
				self.H[u] = set()
				for v in self.adj_left[u]:
					for w in sorted_neig[v]:
						if self.core_number[u] <= self.core_number[w]:
							if u != w and w not in self.H[u]:
								C[w] += 1
							if (self.core_number[u] < self.core_number[w]) or (self.core_number[u] == self.core_number[w] and u < w):
								if C[w] == self.q:
									self.H[u].add(w)
						else:
							break
		else: #right
			sorted_neig={}
			for u in self.adj_left:
				sorted_neig[u]=[]
				tmp_dict={}
				for v in self.adj_left[u]:
					tmp_dict[v]=self.core_number[v]
				sorted_neig[u]=sorted(tmp_dict,key=lambda key: tmp_dict[key],reverse=True)
			for u in self.adj_right:
				C = defaultdict(int)
				self.H[u] = set()
				for v in self.adj_right[u]:
					for w in sorted_neig[v]:
						if self.core_number[u] <= self.core_number[w]:
							if u != w and w not in self.H[u]:
								C[w] += 1
							if (self.core_number[u] < self.core_number[w]) or (self.core_number[u] == self.core_number[w] and u < w):
								if C[w] == self.p:
									self.H[u].add(w)
						else:
							break
			self.p1, self.q1 = self.q, self.p 


	@lru_cache(None)
	def combination(self, n, k):
		if n-k<0:
			return 0
		else:
			return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


	
	def bccounting(self,p,q):
		self.p=p
		self.q=q
		starttime = time.time()
		self.bc_number = 0
		self.edge_weighted_graph = defaultdict(lambda: defaultdict(int))
		for u in self.H:
			if u in self.adj_left:
				self.subcouting(self.H[u],self.adj_left[u],set(),set(),{u},set())
			else:
				self.subcouting(self.adj_right[u],self.H[u],set(),set(),set(),{u})
		#print("numbe of bc"+str(self.bc_number))
		endtime=time.time()
		#print("bccounttime"+str(endtime-starttime))
		return endtime-starttime


	def subcouting(self, C_U,C_V,P_U,P_V,H_U,H_V):
		if len(H_U)>self.p or len(H_V)>self.q or len(C_U|P_U|H_U)<self.p or len(C_V|P_V|H_V)<self.q:
			return 
		c0,c1,p0,p1,h0,h1=len(C_U),len(C_V),len(P_U),len(P_V),len(H_U),len(H_V)
		if len(H_U)==self.p or len(H_V)==self.q:
			self.bc_number+=self.combination(p0+c0,self.p-h0)*self.combination(p1+c1,self.q-h1)
			self.local_counting(C_U,C_V,P_U,P_V,H_U,H_V)
			return
		index=0
		tmp_n_dict={}
		for u in C_U:
			tmp_n_dict[u]=len(self.adj_left[u]&C_V)
			if tmp_n_dict[u]!=0:
				index=1
				break
		if index==0:
			self.bc_number+=self.combination(p0+c0,self.p-h0)*self.combination(p1,self.q-h1)+self.combination(p0,self.p-h0)*self.combination(p1+c1,self.q-h1)-self.combination(p0,self.p-h0)*self.combination(p1,self.q-h1)
			self.local_counting(C_U,C_V,P_U,P_V,H_U,H_V)
			return
		C_U,C_V,P_U,P_V,H_U,H_V=set(C_U),set(C_V),set(P_U),set(P_V),set(H_U),set(H_V)
		tmp_C_U=set(C_U)
		min_u_node, min_v_node = None, None
		min_u, min_v = float("inf"), float("inf")
		for u in tmp_C_U:
			tmp_n=0
			if u in tmp_n_dict:
				tmp_n=tmp_n_dict[u]
			else:
				tmp_n=len(self.adj_left[u]&C_V)
			if tmp_n==len(C_V):
				C_U.remove(u)
				P_U.add(u)
			if tmp_n<min_u:
				min_u=tmp_n
				min_u_node=u
		for u in set(C_V):
			tmp_n=len(self.adj_right[u]&tmp_C_U)
			if tmp_n==len(tmp_C_U):
				C_V.remove(u)
				P_V.add(u)
			if tmp_n<min_v:
				min_v=tmp_n
				min_v_node=u
				
		if len(C_U)==0 and len(C_V)==0:
			self.subcouting(C_U,C_V,P_U,P_V,H_U,H_V)
		else:
			L_U={min_u_node}
			L_V={min_v_node}
			for u in L_U:
				C_U.remove(u)
				self.subcouting(C_U,C_V&self.adj_left[u],P_U,P_V,H_U|{u},H_V)
			for v in L_V:
				C_V.remove(v)
				self.subcouting(C_U&self.adj_right[v],C_V,P_U,P_V,H_U,H_V|{v})
			self.subcouting(C_U,C_V,P_U,P_V,H_U,H_V)

	def local_counting(self, C_U,C_V,P_U,P_V,H_U,H_V):
		c0,c1,p0,p1,h0,h1=len(C_U),len(C_V),len(P_U),len(P_V),len(H_U),len(H_V)
		C_U,C_V,P_U,P_V,H_U,H_V=set(C_U),set(C_V),set(P_U),set(P_V),set(H_U),set(H_V)
		if c0>0 and c1>0:
			self.local_counting(set(),set(),C_U|P_U,P_V,H_U,H_V)
			P_V=P_V|C_V
			for v in C_V:
				P_V.remove(v)
				self.local_counting(set(),set(),P_U,P_V,H_U,H_V|{v})
		else:
			p0+=c0
			p1+=c1
			P_U=P_U|C_U
			P_V=P_V|C_V
			for u in P_U:
				for v in P_V:
					if v in self.adj_left[u]:
						if self.p-h0-1<0 or p0-1<0 or self.q-h1-1<0 or p1-1<0:
							continue
						self.edge_weighted_graph[u][v]+=self.combination(p0-1,self.p-h0-1)*self.combination(p1-1,self.q-h1-1)
						self.edge_weighted_graph[v][u]+=self.combination(p0-1,self.p-h0-1)*self.combination(p1-1,self.q-h1-1)
				for v in H_V:
					if v in self.adj_left[u]:
						if self.p-h0-1<0 or p0-1<0 or self.q-h1<0:
							continue
						self.edge_weighted_graph[u][v]+=self.combination(p0-1,self.p-h0-1)*self.combination(p1,self.q-h1)
						self.edge_weighted_graph[v][u]+=self.combination(p0-1,self.p-h0-1)*self.combination(p1,self.q-h1)
			for u in H_U:
				for v in P_V:
					if v in self.adj_left[u]:
						if self.p-h0<0 or p1-1<0 or self.q-h1-1<0:
							continue
						self.edge_weighted_graph[u][v]+=self.combination(p0,self.p-h0)*self.combination(p1-1,self.q-h1-1)
						self.edge_weighted_graph[v][u]+=self.combination(p0,self.p-h0)*self.combination(p1-1,self.q-h1-1)
				for v in H_V:
					if v in self.adj_left[u]:
						if self.p-h0<0 or self.q-h1<0:
							continue
						self.edge_weighted_graph[u][v]+=self.combination(p0,self.p-h0)*self.combination(p1,self.q-h1)
						self.edge_weighted_graph[v][u]+=self.combination(p0,self.p-h0)*self.combination(p1,self.q-h1)



	def priorities(self): 
		left_rank={}
		right_rank={}
		index=0
		for (u,degree) in sorted(self.deg_left.items(), key=lambda kv: (kv[1], kv[0])):
			left_rank[u]=index
			index+=1
		index=0
		for (u,degree) in sorted(self.deg_right.items(), key=lambda kv: (kv[1], kv[0])):
			right_rank[u]=index
			index+=1

		self.new_old_id={} 
		self.proj_graph={}  
		l=len(self.adj_right)
		for u in self.adj_right:
			self.proj_graph[right_rank[u]]=set()
			self.new_old_id[right_rank[u]]=u
			for v in self.adj_right[u]:
				self.proj_graph[right_rank[u]].add(left_rank[v]+l)
		for u in self.adj_left:
			self.proj_graph[left_rank[u]+l]=set()
			self.new_old_id[left_rank[u]+l]=u
			for v in self.adj_left[u]:
				self.proj_graph[left_rank[u]+l].add(right_rank[v])
		tmp={}
		for u in self.proj_graph:
			tmp[u]=len(self.proj_graph[u])
		new_tmp=sorted(tmp.items(), key=lambda kv: (kv[1], kv[0]))
		self.pro={}
		index=0
		for (u,degree) in new_tmp:
			self.pro[u]=index
			index+=1
		self.sorted_neig={}
		for u in self.proj_graph:
			tmp={}
			for v in self.proj_graph[u]:
				tmp[v]=self.pro[v]
			self.sorted_neig[u]=sorted(tmp.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  #list


	def appbccounting(self,p,q): 
		starttime = time.time()
		self.app_edge_weighted_graph = defaultdict(lambda: defaultdict(int))
		self.p = p
		self.q = q
		self.priorities()
		B22={} 
		for u in self.proj_graph:
			count_wedge = {}
			for (v,degree) in self.sorted_neig[u]:
				for (w,degree) in  self.sorted_neig[v]:
					if self.pro[w]>self.pro[u]:
						if self.pro[w]>self.pro[v]:
							if w not in count_wedge:
								count_wedge[w]=1
							else:
								count_wedge[w]+=1
						else:
							break
					else:
						break
			for (v,degree) in self.sorted_neig[u]:
				for (w,degree) in  self.sorted_neig[v]:
					if self.pro[w]>self.pro[u]:
						if self.pro[w]>self.pro[v]:
							if count_wedge[w]<=1:
								continue
							delta=count_wedge[w]-1
							if (self.new_old_id[u],self.new_old_id[v]) not in B22:
								B22[(self.new_old_id[u],self.new_old_id[v])]=delta
							else:
								B22[(self.new_old_id[u],self.new_old_id[v])]+=delta
							
							if (self.new_old_id[v],self.new_old_id[w]) not in B22:
								B22[(self.new_old_id[v],self.new_old_id[w])] = delta
							else:
								B22[(self.new_old_id[v],self.new_old_id[w])]+=delta
						else:
							break
					else:
						break		
		for (u,v) in B22:
			if u in self.adj_left:
				upper=self.combination(len(self.adj_left[u]-{v}),self.q-1)*self.combination(len(self.adj_right[v]-{u}),self.p-1)
			else:
				upper=self.combination(len(self.adj_right[u]-{v}),self.p-1)*self.combination(len(self.adj_left[v]-{u}),self.q-1)
			self.app_edge_weighted_graph[u][v]=self.app_edge_weighted_graph[v][u]=int((B22[(u,v)]/((self.p-1)*(self.q-1))+upper)/2)
		endtime=time.time()
		#print("appcounttime"+str(endtime-starttime))
		return endtime-starttime



	def build_graph(self,method):	
		if method==1:
			weighted_graph=self.edge_weighted_graph
		if method==2:
			weighted_graph=self.app_edge_weighted_graph
		#find maximum connected componet from the  weighted_graph, which will be used for clustering
		visited = set()
		cc = {}
		for node in list(weighted_graph.keys()):
			if node not in visited:
				cc[node] = set()
				visited.add(node)
				Q = [node]
				while Q:
					v = Q.pop()
					for u in weighted_graph[v]:
						if  u not in visited:
							visited.add(u)
							cc[node].add(u)
							Q.append(u)
				cc[node].add(node)
		max_size, index_max = 0, None
		for seed in cc:
			if len(cc[seed]) > max_size:
				max_size = len(cc[seed])
				index_max = seed
		max_cc = cc[index_max]
		#print("number of  cc"+str(len(cc)))
		#print("nodes_max_cc"+str(len(max_cc)))
		return weighted_graph, max_cc





	def BHSC(self,method): 
		starttime=time.time()
		motif_weighted_graph, motif_max_cc=self.build_graph(method)
		old_new_id = {}
		new_old_id = {}
		index = 0
		for u in sorted(list(motif_max_cc)):
			old_new_id[u]=index
			new_old_id[index]=u
			index+=1
		weighted_sum=0
		n=len(motif_max_cc)
		row=[]
		col=[]
		new_motif_weighted_graph,new_degree_motif={},{}
		for u in motif_max_cc:
			new_degree_motif[old_new_id[u]]=0
			new_motif_weighted_graph[old_new_id[u]]={}
			for v in motif_weighted_graph[u]:
				row.append(old_new_id[u])
				col.append(old_new_id[v])
				new_motif_weighted_graph[old_new_id[u]][old_new_id[v]]=motif_weighted_graph[u][v]
				new_degree_motif[old_new_id[u]]+=motif_weighted_graph[u][v]
				weighted_sum +=motif_weighted_graph[u][v]
		A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
		norm_L = laplacian(A,normed=True)
		emb_eig_val, p = spla.eigsh(norm_L, k=2, which='SM')
		pi = np.real(p[:, 1])
		pi = np.argsort(pi)
		S = set()
		volS = 0
		cutS = 0
		best_condu, best_index, count = float("inf"), 0, 0
		best_set = set()
		for x in pi:
			u = x
			S.add(u)
			count += 1
			for node in new_motif_weighted_graph[u]:
				if node in S:
					cutS -= 2 * new_motif_weighted_graph[u][node]
			cutS = cutS + new_degree_motif[u]
			volS = volS + new_degree_motif[u]
			if min(volS,  weighted_sum - volS) != 0 and cutS / min(volS,  weighted_sum - volS) < best_condu:
				best_condu = cutS / min(volS,  weighted_sum - volS)
				best_index = count
		for x in range(best_index):
			best_set.add(pi[x])
		if len(best_set) > len(new_motif_weighted_graph) / 2:
			best_set = set(new_motif_weighted_graph) - set(best_set)
		endtime=time.time()
		runtime=endtime-starttime
		#计算biclique conductance
		S=set()
		vol_best_set=0
		vol_motif=0
		for v in new_motif_weighted_graph:
			if new_old_id[v] in self.adj_left:
				vol_motif+=new_degree_motif[v]/self.q
				if v in best_set:
					vol_best_set+=new_degree_motif[v]/self.q	
			else:
				vol_motif+=new_degree_motif[v]/self.p
				if v in best_set:
					vol_best_set+=new_degree_motif[v]/self.p
			if v in best_set:
				S.add(new_old_id[v])
		return self.motif_cut(S)/min(vol_best_set,vol_motif-vol_best_set), S, runtime
	
	
	def motif_cut(self, nodes): 
		self.candidate=nodes
		self.cut=0
		self.S = {}
		self.lblgraph_cut(0, set(self.H), set())
		self.p = p
		self.q = q
		return self.cut

	def lblgraph_cut(self, l, H, L):
		if l == self.p1:
			tmp_left=len(L&self.candidate)
			tmp_right=len(self.S[l - 1]&self.candidate)
			tmp=len(self.S[l - 1])-tmp_right
			if tmp_left==0:
				for i in range(1,min(tmp_right,self.q1)+1):
					self.cut+=self.combination(tmp_right, i)*self.combination(tmp, self.q1-i)	
			if tmp_left==l:
				if tmp_right<self.q1:
					for i in range(0,tmp_right+1):
						self.cut+=self.combination(tmp_right, i)*self.combination(tmp, self.q1-i)	
				if tmp_right>=self.q1:
					for i in range(0,self.q1-1+1):
						self.cut+=self.combination(tmp_right, i)*self.combination(tmp, self.q1-i)					
			else:
				self.bc_number+=self.combination(len(self.S[l - 1]), self.q1)		
		else:
			for u in H:
				if l==0:
					self.S[l] = self.adj_left[u] if u in self.adj_left else self.adj_right[u]
				else:
					self.S[l] = (self.S[l - 1] & self.adj_left[u]) if u in self.adj_left else (self.S[l - 1] & self.adj_right[u])
				H1 = self.H[u] & H
				if len(self.S[l]) < self.q1 or len(H1) < self.p1 - l - 1:
					continue
				self.lblgraph_cut(l + 1, H1, L | {u})



if __name__ == "__main__":
	dataset= sys.argv[1]
	G = Graph(dataset)
	pqs=[(2,2),(2,3),(3,2),(2,4),(3,3),(4,2)]
	for (p,q) in pqs:
		print("p,q: " + str(p) + "  " + str(q))
		G.c2hn_H(p,q)
		time_bccounting=G.bccounting(p, q)
		print(f"################ECRC###########")
		quality, S_ECRC, time_ECRC=G.BHSC(1)
		print("quality_ECRC  "+str(quality))
		print("time of ECRC  "+str(time_bccounting+time_ECRC))

		time_appbccounting=G.appbccounting(p, q)
		print(f"################ECRC_E###########")
		quality, S_ECRC_E, time_ECRC_E=G.BHSC(2)
		print("quality_ECRC_E  "+str(quality))
		print("time of ECRC_E  "+str(time_appbccounting+time_ECRC_E))
		
		

