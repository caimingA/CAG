import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import lingam


def draw(matrix):
    node_num = len(matrix)
    length = 8
    pos_list = list()
    # pos_list = [(10, 10), (5, 7), (10, 7), (2, 4), (4, 4), (6, 4), (8, 4), (1, 1), (2, 1), (3, 1)]
    y_gap = (length - 1) / np.log2(node_num)

    # count_layer = 0
    count_num = 0
    for i in range(int(np.log2(node_num)) + 1):
        if 2**i > node_num - count_num:
            x_gap = (length - 1) / (node_num - count_num + 1)
            for j in range(node_num - count_num):
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
                count_num += 1
        else:
            x_gap = (length - 1) / (2**i + 1)
            for j in range(2**i):
                if count_num == node_num:
                    break
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
                count_num += 1
  
    G = nx.DiGraph()
    plt.figure(figsize=(length, length))
    for i in range(len(matrix)):
        G.add_node(i)
        for j in range(len(matrix)):
            if(matrix[i][j]):
                G.add_edge(j, i)

    nx.draw(G, 
        with_labels=True, #这个选项让节点有名称
        edge_color='b', # b stands for blue! 
        # pos=nx.planar_layout(G, scale = 0.1), # 这个是选项选择点的排列方式，具体可以用 help(nx.drawing.layout) 查看
        pos=pos_list,
      # 主要有spring_layout  (default), random_layout, circle_layout, shell_layout,    
      # 这里是环形排布，还有随机排列等其他方式  
        node_color='r', # r = red
        node_size=300, # 节点大小
        width=2, # 边的宽度
        arrowstyle='->',
        arrowsize=20,
        connectionstyle="arc3, rad=0.15"
           )
    ax = plt.gca()
    ax.set_axis_off()
    # plt.show(block=False)
    plt.show()
    return [list(nx.ancestors(G, i)) for i in range(node_num)]


def draw_sub(matrix, variables, num):
    M = np.zeros((num, num))
    
    node_num = len(M)
    length = 8
    pos_list = list()
    # pos_list = [(10, 10), (5, 7), (10, 7), (2, 4), (4, 4), (6, 4), (8, 4), (1, 1), (2, 1), (3, 1)]
    y_gap = (length - 1) / np.log2(node_num)

    # count_layer = 0
    count_num = 0
    for i in range(int(np.log2(node_num)) + 1):
        if 2**i > node_num - count_num:
            x_gap = (length - 1) / (node_num - count_num + 1)
            for j in range(node_num - count_num):
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
                count_num += 1
        else:
            x_gap = (length - 1) / (2**i + 1)
            for j in range(2**i):
                if count_num == node_num:
                    break
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
                count_num += 1
  
    G = nx.DiGraph()
    plt.figure(figsize=(length, length))
    
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            M[variables[i]][variables[j]] = matrix[i][j]
#     print(M)
    
    for i in range(len(M)):
        G.add_node(i)
        for j in range(len(M)):
            if(M[i][j]):
                G.add_edge(j, i)

    nx.draw(G, 
        with_labels=True, #这个选项让节点有名称
        edge_color='b', # b stands for blue! 
        # pos=nx.planar_layout(G, scale = 0.1), # 这个是选项选择点的排列方式，具体可以用 help(nx.drawing.layout) 查看
        pos=pos_list,
      # 主要有spring_layout  (default), random_layout, circle_layout, shell_layout,    
      # 这里是环形排布，还有随机排列等其他方式  
        node_color='r', # r = red
        node_size=300, # 节点大小
        width=2, # 边的宽度
        arrowstyle='->',
        arrowsize=20,
        connectionstyle="arc3, rad=0.15"
           )
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


def draw_diff(matrix, matrix_original):
    node_num = len(matrix)
    length = 15
    pos_list = list()
    # pos_list = [(10, 10), (5, 7), (10, 7), (2, 4), (4, 4), (6, 4), (8, 4), (1, 1), (2, 1), (3, 1)]
    y_gap = (length - 1) / np.log2(node_num)

    # count_layer = 0
    count_num = 0
    for i in range(int(np.log2(node_num)) + 1):
        if 2**i > node_num - count_num:
            x_gap = (length - 1) / (node_num - count_num + 1)
            for j in range(node_num - count_num):
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
                count_num += 1
        else:
            x_gap = (length - 1) / (2**i + 1)
            for j in range(2**i):
                if count_num == node_num:
                    break
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
                count_num += 1
  
  
    corr_count = 0
    adding_count = 0
    deleting_count = 0
  
    G = nx.DiGraph()
    plt.figure(figsize=(length, length))
    for i in range(len(matrix)):
        G.add_node(i)
        for j in range(len(matrix)):
            if(matrix[i][j]):
                if(matrix_original[i][j]):
                    G.add_edge(j, i, color='b')
                    corr_count += 1
                else:
                    G.add_edge(j, i, color='r')
                    adding_count += 1
        else:
            if(matrix_original[i][j]):
                G.add_edge(j, i, color='black')
                deleting_count += 1

    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]


    nx.draw(G, 
        with_labels=True, #这个选项让节点有名称
        edge_color=colors, # b stands for blue! 
        # pos=nx.planar_layout(G, scale = 0.1), # 这个是选项选择点的排列方式，具体可以用 help(nx.drawing.layout) 查看
        pos=pos_list,
      # 主要有spring_layout  (default), random_layout, circle_layout, shell_layout,    
      # 这里是环形排布，还有随机排列等其他方式  
        node_color='r', # r = red
        node_size=300, # 节点大小
        width=2, # 边的宽度
        arrowstyle='->',
        arrowsize=20,
        connectionstyle="arc3, rad=0.15"
    )
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

    return [corr_count, adding_count, deleting_count]


def draw_groups(data, groups, B):
    M = np.zeros_like(B)
    visit = np.zeros_like(B)
    
    N_num = len(M)
    
    for g in groups:
        n_num = len(g)
        
        X = list()
        for index in g:
            X.append(data[:, index])            
        X = np.array(X).T
            
        model = lingam.DirectLiNGAM()
        model.fit(X)
        m = model.adjacency_matrix_
        
        print(g)
        draw_sub(m, g, N_num)


def draw_Markov(matrix):
    node_num = len(matrix)
    length = 8
    pos_list = list()
    # pos_list = [(10, 10), (5, 7), (10, 7), (2, 4), (4, 4), (6, 4), (8, 4), (1, 1), (2, 1), (3, 1)]
    y_gap = (length - 1) / np.log2(node_num)

    # count_layer = 0
    count_num = 0
    for i in range(int(np.log2(node_num)) + 1):
        if 2**i > node_num - count_num:
            x_gap = (length - 1) / (node_num - count_num + 1)
            for j in range(node_num - count_num):
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
                count_num += 1
        else:
            x_gap = (length - 1) / (2**i + 1)
            for j in range(2**i):
                if count_num == node_num:
                    break
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
                count_num += 1

    G = nx.DiGraph()
    plt.figure(figsize=(length, length))
    for i in range(len(matrix)):
        G.add_node(i)
        # for j in range(len(matrix)):
        #     if(matrix[i][j]):
        #         G.add_edge(j, i)
    
    nx.draw(G, 
        with_labels=True, #这个选项让节点有名称
        pos=pos_list,
      # 主要有spring_layout  (default), random_layout, circle_layout, shell_layout,    
      # 这里是环形排布，还有随机排列等其他方式  
        node_color='r', # r = red
        node_size=300, # 节点大小
    )
    # nx.draw_networkx_nodes(G, pos=pos_list, with_labels=True)
    
    # nx.draw_networkx_nodes(G, pos=pos_list, node_color='r', node_size=300)
    # i --> j
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if(matrix[j][i] == 1 and matrix[i][j] == -1):
                G.add_edge(i, j, arr='->')
            if(matrix[j][i] == -1 and matrix[i][j] == -1):
                ind_max = np.max([i, j])
                ind_min = np.min([i, j])
                G.add_edge(ind_max, ind_min, arr='-')
            if(matrix[j][i] == 1 and matrix[i][j] == 1):
                ind_max = np.max([i, j])
                ind_min = np.min([i, j])
                G.add_edge(ind_max, ind_min, arr='<->')

    edges = G.edges()
    # print(type(edges))
    arrs = [G[u][v]['arr'] for u,v in edges]
    # print(edges)
    # print(arrs)
    edge_list = list(edges)
    # print(temp)
    # print(type(temp))
    # print(temp[0])
    for i in range(len(edge_list)):
        nx.draw_networkx_edges(G
                            , edgelist=[edge_list[i]]
                            , pos=pos_list
                            , edge_color='b'
                            , width=2
                            , arrowstyle=arrs[i]
                            , arrowsize=20
                            , connectionstyle="arc3, rad=0.15")

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


def draw_RCD(matrix):
    node_num = len(matrix)
    length = 8
    pos_list = list()
    # pos_list = [(10, 10), (5, 7), (10, 7), (2, 4), (4, 4), (6, 4), (8, 4), (1, 1), (2, 1), (3, 1)]
    y_gap = (length - 1) / np.log2(node_num)

    # count_layer = 0
    count_num = 0
    for i in range(int(np.log2(node_num)) + 1):
        if 2**i > node_num - count_num:
            x_gap = (length - 1) / (node_num - count_num + 1)
            for j in range(node_num - count_num):
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
                count_num += 1
        else:
            x_gap = (length - 1) / (2**i + 1)
            for j in range(2**i):
                if count_num == node_num:
                    break
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i))
                count_num += 1
  
    G = nx.DiGraph()
    plt.figure(figsize=(length, length))
    for i in range(len(matrix)):
        G.add_node(i)
        for j in range(len(matrix)):
            if(matrix[i][j]):
                G.add_edge(j, i)

    nx.draw(G, 
        with_labels=True, #这个选项让节点有名称
        edge_color='b', # b stands for blue! 
        # pos=nx.planar_layout(G, scale = 0.1), # 这个是选项选择点的排列方式，具体可以用 help(nx.drawing.layout) 查看
        pos=pos_list,
      # 主要有spring_layout  (default), random_layout, circle_layout, shell_layout,    
      # 这里是环形排布，还有随机排列等其他方式  
        node_color='r', # r = red
        node_size=300, # 节点大小
        width=2, # 边的宽度
        arrowstyle='->',
        arrowsize=20,
        connectionstyle="arc3, rad=0.15"
           )
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

    return [list(nx.ancestors(G, i)) for i in range(node_num)]