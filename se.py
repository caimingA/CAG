import numpy as np

# 估计值 (多维向量)
b = np.array([2.5, 1.8, 3.0])

# 要检验的假设值 (多维向量)
beta = np.array([0, 0, 0])

# 估计值的标准误差 (对角矩阵)
se_b = np.array([1.0, 0.8, 1.2])

print(se_b ** 2)

# 协方差矩阵的逆矩阵 (示例中使用对角矩阵)
cov_inv = np.diag(1 / (se_b ** 2))

# 计算差异向量
diff_vector = b - beta

# 计算Wald统计量
wald_statistic = diff_vector.dot(cov_inv).dot(diff_vector)

print("Wald统计量:", wald_statistic)