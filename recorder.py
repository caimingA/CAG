import xlwt


alpha_l_list = [0.01, 0.05, 0.1, 0.5]
alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
# def write_excel(res_list, res_L, res_K, res_RCD_list, name):
#     f = xlwt.Workbook()
#     sheet1 = f.add_sheet('proposed',cell_overwrite_ok=True)
#     sheet2 = f.add_sheet('comp',cell_overwrite_ok=True)
#     sheet3 = f.add_sheet('RCD',cell_overwrite_ok=True)
    
#     for i in range(len(res_list)):
#         for j in range(len(res_list[0])):
#             sheet1.write(i,j, res_list[i][j])

#     for i in range(len(res_L)):
#         sheet2.write(0, i, res_L[i])

#     for i in range(len(res_K)):
#         sheet2.write(1, i, res_K[i])

#     for i in range(len(res_RCD_list)):
#         for j in range(len(res_RCD_list[0])):
#             sheet3.write(i,j, res_RCD_list[i][j])
    
#     f.save(name + '.xls')


def write_excel(res_list, res_L, res_K, res_RCD_list, res_CAPA_list_0, res_CAPA_list_1, res_CAPA_list_2, name):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('proposed',cell_overwrite_ok=True)
    sheet2 = f.add_sheet('comp',cell_overwrite_ok=True)
    sheet3 = f.add_sheet('RCD',cell_overwrite_ok=True)
    
    sheet4 = f.add_sheet('CAPA_0',cell_overwrite_ok=True)
    sheet5 = f.add_sheet('CAPA_1',cell_overwrite_ok=True)
    sheet6 = f.add_sheet('CAPA_2',cell_overwrite_ok=True)
    
    for i in range(len(res_list)):
        for j in range(len(res_list[0])):
            sheet1.write(i,j, res_list[i][j])

    for i in range(len(res_L)):
        sheet2.write(0, i, res_L[i])

    for i in range(len(res_K)):
        sheet2.write(1, i, res_K[i])

    for i in range(len(res_RCD_list)):
        for j in range(len(res_RCD_list[0])):
            sheet3.write(i,j, res_RCD_list[i][j])
    
    for i in range(len(res_CAPA_list_0)):
        for j in range(len(res_CAPA_list_0[0])):
            sheet4.write(i,j, res_CAPA_list_0[i][j])

    for i in range(len(res_CAPA_list_1)):
        for j in range(len(res_CAPA_list_1[0])):
            sheet5.write(i,j, res_CAPA_list_1[i][j])

    for i in range(len(res_CAPA_list_2)):
        for j in range(len(res_CAPA_list_2[0])):
            sheet6.write(i,j, res_CAPA_list_2[i][j])
    
    # for i in range(len(res_CAPA_list_0)):
    #     sheet4.write(0, i, res_CAPA_list_0[i])
    #     sheet4.write(1, i, res_CAPA_list_1[i])
    #     sheet4.write(2, i, res_CAPA_list_2[i])
    
    f.save(name + '.xls')


def write_excel_test(res_list, name):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('err',cell_overwrite_ok=True)
    # print(res_list)
    for i in range(len(res_list)):
        sheet1.write(0,i, res_list[i])
        # for j in range(len(res_list[0])):
        #     sheet1.write(i,j, res_list[i][j])
    
    f.save(name + '_ERROR.xls')