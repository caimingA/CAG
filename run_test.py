import data_generator
import execute_refined
import recorder




def run(name, DAG_list, data_list, B_list, alpha_l_list, alpha_i_list, alpha_p = [1]):
    res_list = list()

    for i in alpha_l_list:
        for j in alpha_i_list:
            for k in alpha_p:
                print(i, ", ", j, ", ", k)
                res = execute_refined.execute_test(DAG_list, data_list, B_list, i, j, j, k)
                res_list_0 = res[0]
                res_list_1 = res[1]
                res_list_2 = res[2]

    # res_L = execute.execute_L(DAG_list, data_list, B_list)
    # res_K = execute.execute_K(DAG_list, data_list, B_list)
    # res_list_2 = [res_L, res_K]
    # name = "10_9_1_11_0406"
    # recorder.write_excel_DAG(res_list, res_L, res_K, name)
    #
    recorder.write_excel_test(res_list_0.tolist(), name + "_Wald")
    recorder.write_excel_test(res_list_1.tolist(), name + "_Anc")
    recorder.write_excel_test(res_list_2.tolist(), name + "_Valu")
    # print(res_list)        

    