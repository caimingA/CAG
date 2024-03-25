import data_generator
import execute
import recorder




def run(name, DAG_list, data_list, B_list, alpha_l_list, alpha_i_list, alpha_p = [1]):
    res_list = list()
    res_e_list = list()

    for i in alpha_l_list:
        for j in alpha_i_list:
            for k in alpha_p:
                print(i, ", ", j, ", ", k)
                res = execute.execute_HSIC_DAG(DAG_list, data_list, B_list, i, j, j, k)
                res_list = res[0]
                res_e_list = res[1]

    res_L = execute.execute_L_DAG(DAG_list, data_list, B_list)

    res_temp = execute.execute_K_DAG(DAG_list, data_list, B_list)
    res_K = res_temp[0]
    res_K_e = res_temp[1]
    # res_list_2 = [res_L, res_K]
    # name = "10_9_1_11_0406"
    recorder.write_excel_DAG(res_list, res_L, res_K, name)

    recorder.write_excel_DAG(res_e_list, res_L, res_K_e, name + "_eliminate")           

    