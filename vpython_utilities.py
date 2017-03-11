from itertools import product
import numpy as np

def make_grid2D(N, M):
    PointList = [(u[0], 0, u[1]) for u in product(range(N), range(M))]
    sub2ind = {(i, j):i*M+j for i,j in product(range(N), range(M))}
    ind2sub = {i*M+j:(i,j) for i,j in product(range(N), range(M))}

    edges = []
    for i in range(N):
        for j in range(M):
            if i>0:
                edges.append((sub2ind[(i-1,j)], sub2ind[(i,j)]))
            if j>0:
                edges.append((sub2ind[(i,j-1)], sub2ind[(i,j)]))

    adj = np.zeros((N*M, N*M))
    for e in edges:
        adj[e[0],e[1]] = 1
        adj[e[1],e[0]] = 1
    
    return PointList, sub2ind, ind2sub, edges, adj

