# 一些纯函数
# 辅助用


# 每行为一个数据
# 以列为基准, 标准正太化每一列
# newX = (X - 平均值) / 标准差
def normalData(Matrix):
    # Matrix -- 每一行为一个样本
    #        -- shape = (m, size) , m为样本数量, size为样本属性的数量 
    miu = np.mean(Matrix, axis = 0, keepdims=True) # 平均值, shape = (1, size)
    sigma_square = np.var(Matrix, axis = 0, keepdims=True) # 方差, shape = (1, size)
    sigma = np.sqrt(sigma_square)
    assert(miu.shape == sigma.shape)
    newMatrix = (Matrix - miu) / sigma
    return newMatrix, miu, sigma
# X_normal, X_miu, X_sigma = normalData(X)
# Y_normal, Y_miu, Y_sigma = normalData(Y)

# newX = ( X - min(X) ) / ( max(X) - min(X) )
# 每一行为一个样本数据
# 以列为基准, 处理每列的数据
def minMaxData(Matrix):
    # Matrix -- 每一行为一个样本
    #        -- shape = (m, size) , m为样本数量, size为样本属性的数量 
    M_min = np.min(Matrix, axis = 0) # 最小值, shape = (1, size)
    M_max = np.max(Matrix, axis = 0) # 最小值, shape = (1, size)
    M_min = M_min.reshape(1, -1)
    M_max = M_max.reshape(1, -1)
    newMatrix = (Matrix - M_min) / (M_max - M_min)
    return newMatrix, M_min, M_max
# X_minmax, X_min, X_max = minMaxData(X)
# Y_minmax, Y_min, Y_max = minMaxData(Y)