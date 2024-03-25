import numpy as np
from scipy.stats import chi2
import statsmodels.api as sm
import pandas as pd

# 创建一些示例数据
np.random.seed(0)
# X = np.random.rand(100, 2)
X = np.random.rand(100, 2)
# print(X)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.rand(100)
# print(y)
# 添加常数项（截距项）到X矩阵
# X = np.column_stack((np.ones(len(X)), X))

# 拟合线性回归模型（计算系数）
coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
print(coefficients)

# 计算残差
residuals = y - X.dot(coefficients)

# 计算残差的方差
residual_variance = np.var(residuals)

# 计算X的协方差矩阵
X_covariance = np.linalg.inv(X.T.dot(X))
# print(X_covariance)

# 定义要测试的假设（这里假设X1和X2的系数都等于零）
hypothesis = np.array([0, 3])

# 计算Wald统计量
wald_statistic = (hypothesis.dot(coefficients) / np.sqrt(residual_variance * hypothesis.dot(X_covariance).dot(hypothesis)))

# 计算P值
# df = 1  # 自由度为1
# p_value = 1 - chi2.cdf(wald_statistic, df)

print("Wald统计量:", wald_statistic)
# print("P值:", p_value)

# print(residuals)
sse = np.sum(residuals**2)
# 计算自由度
n = len(y)  # 样本数量
p = X.shape[1] - 1  # 自变量的数量（去除截距项）
df = n - p - 1  # 自由度

# print("df: ", df)

# 计算均方误差
mse = sse / df

# 计算协方差矩阵
cov_matrix = mse * np.linalg.inv(X.T.dot(X))

# print(np.diag(cov_matrix))
# 从协方差矩阵中提取回归系数的标准误差
std_errors = np.sqrt(np.diag(cov_matrix))

print("回归系数的标准误差:", std_errors)

cov_inv = np.diag(1 / (std_errors ** 2))

diff_vector = hypothesis - coefficients
wald_statistic = diff_vector.dot(cov_inv).dot(diff_vector)
print("Wald统计量:", wald_statistic)

# 创建一些示例数据
# np.random.seed(0)
# X = np.random.rand(100, 2)

# y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.rand(100)
# print(X)
# 添加常数项（截距项）到X矩阵
# X = sm.add_constant(X)
# print(X)
print(len(X.shape))
print(len(y.shape))
X = pd.DataFrame(X, columns=['X1', 'X2'])

# print(X.head())
# 拟合线性回归模型
model = sm.OLS(y, X).fit()

# 打印回归模型的摘要
# print(model.summary())

# 执行Wald检验
hypotheses = '(X1 = 0, X2 = 3)'  # 在这里定义你要检验的假设，这里假设X1和X2的系数都等于零
wald_test = model.wald_test(hypotheses)
print("\nWald检验结果:")
print("Wald统计量:", wald_test.statistic)
print("P值:", wald_test.pvalue)

# 估计值 (多维向量)
# b = np.array([2.3907, 3.5131])

# # 要检验的假设值 (多维向量)
# beta = np.array([2, 3])

# # 估计值的标准误差 (对角矩阵)
# se_b = np.array([0.087, 0.091])

# # 协方差矩阵的逆矩阵 (示例中使用对角矩阵)
# cov_inv = np.diag(1 / (se_b ** 2))

# # 计算差异向量
# diff_vector = b - beta

# # 计算Wald统计量
# wald_statistic = diff_vector.dot(cov_inv).dot(diff_vector)

# print("Wald统计量:", wald_statistic)

print(wald_test.statistic/wald_statistic)
print(model.summary())