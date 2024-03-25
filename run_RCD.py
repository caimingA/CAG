import data_generator
import execute
import execute_refined
import recorder


def run(name, DAG_list, data_list, B_list, alpha_l_list, alpha_i_list, alpha_p = [1]):
    res_list_1 = list()
    res_list_2 = list()
    res_list_3 = list()
    res_RCD_list = list()

    for i in alpha_l_list:
        for j in alpha_i_list:
            for k in alpha_p:
                print(i, ", ", j, ", ", k)
                res = execute_refined.execute_KCI(DAG_list, data_list, B_list, i, j, j, k)
                # res = execute_refined.execute_HSIC(DAG_list, data_list, B_list, i, j, j, k)
                res_list_1.append(res[0])
                res_list_2.append(res[1])
                res_list_3.append(res[2])

                res_RCD = execute_refined.execute_RCD(DAG_list, data_list, B_list, i, j)
                res_RCD_list.append(res_RCD)

    # print(res_list)
    # print(res_RCD_list)
    
    res_L = execute_refined.execute_L(DAG_list, data_list, B_list)
    # print(res_L)
    res_K = execute_refined.execute_K(DAG_list, data_list, B_list)

    recorder.write_excel(res_list_1, res_L, res_K[0], res_RCD_list, name + "_Wald")
    recorder.write_excel(res_list_2, res_L, res_K[1], res_RCD_list, name + "_Ance")
    recorder.write_excel(res_list_3, res_L, res_K[2], res_RCD_list, name + "_Valu")