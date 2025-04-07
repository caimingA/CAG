import run
import data_generator
import random
import numpy as np
import xlwt

import time

if __name__ == '__main__':
    # alpha_l_list = [0.5]
    # alpha_i_list = [0.5]
    alpha_l_list = [0.01, 0.05, 0.1, 0.5]
    alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    # alpha_p = [0.01, 0.1, 1]
    alpha_p = [1]

    # experiment_set = [(40, 39, 1)]
    # experiment_set = [(5, 4, 1)]
    # experiment_set = [(10, 9, 1), (5, 4, 1), (10, 5, 1)]
    # experiment_set = [(10, 5, 1), (10, 9, 1), (20, 19, 1)]
    # experiment_set = [(20, 37, 2), (20, 54, 3)]
    # experiment_set = [(10, 10, 2), (10, 15, 3), (20, 20, 2), (20, 30, 3)]
    # experiment_set = [(10, 12, 2), (10, 16, 3)]
    experiment_set = [(10, 14, 2)]
    # experiment_set = [(30, 29, 1), (40, 39, 1)]
    # experiment_set = [(40, 39, 1)]
    # experiment_set = [(10, 5, 1), (10, 9, 1), (20, 19, 1), (30, 29, 2), (40, 39, 1), (50, 49, 1)]
    # experiment_set = [(30, 29, 1)]
    
    
    sample_size = [11, 25, 50, 100, 500, 1000]
    # sample_size = [50, 100, 1000]
    # sample_size = [11]
    # sample_size = [1000]
    population_size = 10000
    dag_size = 100



    # t = time.strftime("%Y-%m-%d", time.localtime())
    for e in experiment_set:
        print("========================")
        
        print(e)
        sample_list = list()
        x_list_list = list()
        
        DAG_list, data_list, B_list = data_generator.generate_DAGs(e[0], e[1], e[2], population_size, dag_size)
        
        
        data_list = np.array(data_list)
        print(data_list[0].shape)
        for s in sample_size:
            if e[0] < s:
                # print("------------------------")
                indexs = np.random.randint(population_size, size=s)
                name = str(e[0]) + "_" + str(e[1]) + "_" + str(e[2]) + "_" + str(s) # node_edge_indegree_sample.
                x_list = data_list[:, indexs, :]
                
                sample_list.append(s)
                x_list_list.append(x_list)
                # print(name)
                # print(x_list.shape)

        
        
        for i in range(dag_size):
            f = xlwt.Workbook()
            
            sheet1 = f.add_sheet('DAG',cell_overwrite_ok=True)            
            DAG = DAG_list[i]
            
            sheet2 = f.add_sheet('B',cell_overwrite_ok=True)
            B = B_list[i]
            
            # print(DAG)
            for x in range(len(DAG)):
                for y in range(len(DAG)):
                    sheet1.write(x,y, DAG[x][y])
                    sheet2.write(x,y, B[x][y])

            for j in range(len(sample_list)):
                sample = sample_list[j]
                x_list = x_list_list[j][i]

                # print(sample)
                # print(x_list.shape)

                sheet_temp = f.add_sheet(str(sample),cell_overwrite_ok=True)

                for x in range(x_list.shape[0]):
                    for y in range(x_list.shape[1]):
                        sheet_temp.write(x, y, x_list[x][y])
            
            f.save(str(e[0]) + "_" + str(e[1]) + "_" + "DAG_" + str(i) + '.xls')
        
                # print(DAG_list)
                
        #         # run.run(name, DAG_list, x_list, B_list, alpha_l_list, alpha_i_list, alpha_p)


# def write_cvs(name, DAG_list, x_list, B_list):
#     f = xlwt.Workbook()
#     sheet1 = f.add_sheet('DAG',cell_overwrite_ok=True)
#     sheet2 = f.add_sheet('B',cell_overwrite_ok=True)
#     sheet3 = f.add_sheet('x',cell_overwrite_ok=True)
    
    
#     # for i in range(len(res_list)):
#     #     for j in range(len(res_list[0])):
#     #         sheet1.write(i,j, res_list[i][j])

#     # for i in range(len(res_L)):
#     #     sheet2.write(0, i, res_L[i])

#     # for i in range(len(res_K)):
#     #     sheet2.write(1, i, res_K[i])

#     # for i in range(len(res_RCD_list)):
#     #     for j in range(len(res_RCD_list[0])):
#     #         sheet3.write(i,j, res_RCD_list[i][j])
    
#     f.save(name + '.xls')