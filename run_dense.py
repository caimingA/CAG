import data_generator
import execute
import execute_refined
import recorder


def run(name, DAG_list, data_list, B_list, alpha_l_list, alpha_i_list, alpha_p = [1]):
    res_list_1 = list()
    res_list_2 = list()
    res_list_3 = list()
    res_RCD_list_1 = list()
    res_RCD_list_2 = list()
    res_RCD_list_3 = list()

    res_CAPA_list_0_1 = list()
    res_CAPA_list_0_2 = list()
    res_CAPA_list_0_3 = list()
    
    res_CAPA_list_1_1 = list()
    res_CAPA_list_1_2 = list()
    res_CAPA_list_1_3 = list()
    
    res_CAPA_list_2_1 = list()
    res_CAPA_list_2_2 = list()
    res_CAPA_list_2_3 = list()

    # res_CAPA_list_1 = list()
    # res_CAPA_list_2 = list()
    # res_CAPA_list_3 = list()

    # for i in alpha_l_list:
    #     res = execute_refined.execute_CAPA(DAG_list, data_list, B_list, 0, i)
    #     res_CAPA_list_0_1.append(res[0])
    #     res_CAPA_list_0_2.append(res[1])
    #     res_CAPA_list_0_3.append(res[2])

    #     res = execute_refined.execute_CAPA(DAG_list, data_list, B_list, 1, i)
    #     res_CAPA_list_1_1.append(res[0])
    #     res_CAPA_list_1_2.append(res[1])
    #     res_CAPA_list_1_3.append(res[2])

    #     res = execute_refined.execute_CAPA(DAG_list, data_list, B_list, 2, i)
    #     res_CAPA_list_2_1.append(res[0])
    #     res_CAPA_list_2_2.append(res[1])
    #     res_CAPA_list_2_3.append(res[2])
    
    for i in alpha_l_list:

        res = execute_refined.execute_CAPA(DAG_list, data_list, B_list, 0, i)
        res_CAPA_list_0_1.append(res[0])
        res_CAPA_list_0_2.append(res[1])
        res_CAPA_list_0_3.append(res[2])

        res = execute_refined.execute_CAPA(DAG_list, data_list, B_list, 1, i)
        res_CAPA_list_1_1.append(res[0])
        res_CAPA_list_1_2.append(res[1])
        res_CAPA_list_1_3.append(res[2])

        res = execute_refined.execute_CAPA(DAG_list, data_list, B_list, 2, i)
        res_CAPA_list_2_1.append(res[0])
        res_CAPA_list_2_2.append(res[1])
        res_CAPA_list_2_3.append(res[2])

        for j in alpha_i_list:
            for k in alpha_p:
                print(i, ", ", j, ", ", k)
                res = execute_refined.execute_KCI(DAG_list, data_list, B_list, i, j, j, k)
                # res = execute_refined.execute_HSIC(DAG_list, data_list, B_list, i, j, j, k)
                res_list_1.append(res[0])
                res_list_2.append(res[1])
                res_list_3.append(res[2])

                res_RCD = execute_refined.execute_RCD(DAG_list, data_list, B_list, i, j)
                res_RCD_list_1.append(res_RCD[0])
                res_RCD_list_2.append(res_RCD[1])
                res_RCD_list_3.append(res_RCD[2])
    # print(res_list)
    # print(res_RCD_list)
    
    res_L = execute_refined.execute_L(DAG_list, data_list, B_list)
    # print(res_L)
    res_K = execute_refined.execute_K(DAG_list, data_list, B_list)

    # res_CAPA_0 = execute_refined.execute_CAPA(DAG_list, data_list, B_list, 0)
    # res_CAPA_1 = execute_refined.execute_CAPA(DAG_list, data_list, B_list, 1)
    # res_CAPA_2 = execute_refined.execute_CAPA(DAG_list, data_list, B_list, 2)

    ######
    # res_list_1 = [[0]]
    # res_list_2 = [[0]]
    # res_list_3 = [[0]]

    # res_RCD_list_1 = [[0]]
    # res_RCD_list_2 = [[0]]
    # res_RCD_list_3 = [[0]]
    ######
    
    recorder.write_excel(res_list_1, res_L, res_K[0], res_RCD_list_1, res_CAPA_list_0_1, res_CAPA_list_1_1, res_CAPA_list_2_1, name + "_Wald")
    recorder.write_excel(res_list_2, res_L, res_K[1], res_RCD_list_2, res_CAPA_list_0_2, res_CAPA_list_1_2, res_CAPA_list_2_2, name + "_Ance")
    recorder.write_excel(res_list_3, res_L, res_K[2], res_RCD_list_3, res_CAPA_list_0_3, res_CAPA_list_1_3, res_CAPA_list_2_3, name + "_Valu")
    # print(res_K)
    
    # res_K_e = execute.execute_K_eliminate(DAG_list, data_list, B_list)
    # # res_list_2 = [res_L, res_K]
    # # name = "10_9_1_11_0406"
    # recorder.write_excel(res_list, res_L, res_K, name)  
    # recorder.write_excel(res_e_list, res_L, res_K_e, name+"_eliminate")         

    # res_PC_list = list()
    # res_PC_L_list = list()
    # for i in alpha_l_list:
    #     for j in alpha_i_list:
    #         for k in alpha_p:
    #             print(i, ", ", j, ", ", k)
    #             res = execute.execute_PC_HSIC(DAG_list, data_list, B_list, i, j, j, 1)
    #             res_PC_list.append(res[0:7])

    #             time_window = res[7]
    #             groups = res[8]
    #             ancestor_dict_list = res[9]
    #             # print("////////////////////", ancestor_dict)

    #             res_PC_L = execute.execute_PC_L(DAG_list, data_list, B_list, i, j, j, 1, time_window, ancestor_dict_list)
    #             res_PC_L_list.append(res_PC_L)

    # res_PC_K = execute.execute_PC_K(DAG_list, data_list, B_list)
    # recorder.write_excel_PC(res_PC_list, res_PC_L_list, res_PC_K, name)
    
