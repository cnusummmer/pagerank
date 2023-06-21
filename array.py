import pandas as pd
from bisect import bisect_left
import netdraw
import csv
from networkx import *
import matplotlib.pyplot as plt

def list(l1, l2):
    for i in range(0, len(l1)):
        if l1[i] not in l2:
            l2.append(l1[i])
    return l2

def list_with_repeat(l1, l2):
    for i in range(0, len(l1)):
        l2.append(l1[i])
    return l2

def read(fname):
    data = pd.read_csv(fname)
    user = []
    new1 = []
    new2 = []
    comment_user = list_with_repeat(data['DATA1'], new1)
    article_user = list_with_repeat(data['DATA2'], new2)
    user = list(data['DATA1'], user)
    user.sort()
    return user, comment_user, article_user

def toMatrix(fpath, list, com, art):
    header = list
    print(list)
    bool_list = []

    with open(fpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list)

    matrix = [[0 for _ in range(len(list))] for _ in range(len(list))]
    for i in range(0,len(com)):
        x = com[i]
        y = art[i]
        x_bool = x in list
        y_bool = y in list
        bool_list.append(x_bool)
        if x_bool == True and y_bool == True:
            x_index = list.index(x)
            y_index = list.index(y)

            matrix[x_index][y_index] += 1
        else:
            pass

    with open('comdat_HL1.csv', 'w', newline='') as ff:
        writer = csv.writer(ff)
        for j in range(0,len(list)):
            writer. writerow(matrix[j])
            # print(j)

    print(matrix)
    return matrix

'''
def netdraw(nodelist, matrix):
    G = networkx.Graph()
    point = nodelist
    G.add_nodes_from(point)
    edgelist = []
    for i in range(len(point)):
        for j in range(len(point)):
            edgelist.append((matrix[i][0], matrix[i][j]))
    G = networkx.graph(edgelist)
    position = networkx.circular_layout(G)
    networkx.draw_networkx_nodes(G, position, nodelist=point, node_color='blue')
    networkx.draw_networkx_edges(G, position)
    networkx.draw_networkx_labels(G, position)
    plt.show()
'''
def main(fname):
    f = fname
    list, com, art = read(f)
    matrix = toMatrix('./data/data_matrix_HL1.csv', list, com, art)
    netdraw(list, matrix)

if __name__ == '__main__':
    filename = './data/HL_DATA.csv'
    main(filename)
