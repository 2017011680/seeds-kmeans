import numpy as np
import matplotlib.pyplot as plt
from Cal_Norm import Cal_Norm
from KNN import K_nearest

from sklearn import metrics

#欧氏距离计算
class Seeds_Kmeans():
    def distEclud(self,x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def initial_center(self,traindataset):
        center_dic = {}
        index = 0
        for keys in traindataset[:, len(traindataset[0]) - 1]:
            if keys not in center_dic.keys():
                center_dic[keys] = [list(map(float, traindataset[index, :len(traindataset[0]) - 1]))]
            else:
                center_dic[keys].append(list(map(float, traindataset[index, :len(traindataset[0]) - 1])))
            index += 1
        for keys in center_dic.keys():
            center_dic[keys] = np.average(center_dic[keys], axis=0)
        return center_dic

    # k均值聚类
    def KMeans(self,dataSet, train_dataset, k):
        m = np.shape(dataSet)[0]
        # 初始化一个矩阵来存储每个点的簇分配结果
        # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
        clusterAssment = np.mat(np.zeros((m, 2)))
        clusterChange = True
        train_dataset = train_dataset.tolist()
        centroids = self.initial_center(np.array(train_dataset))
        # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
        while clusterChange:
            clusterChange = False
            pre_centroids = centroids
            # 遍历所有样本（行数）
            for i in range(m):
                minDist = float('inf')
                minIndex = -1
                # 遍历所有数据找到距离每个点最近的质心,
                # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
                for keys in centroids.keys():
                    # 计算数据点到质心的距离
                    # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                    distance = self.distEclud(centroids[keys], dataSet[i, :])
                    temp_index = K_nearest().Kw_predict(np.array(train_dataset), dataSet[i], 5)
                    # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                    if distance < minDist:
                        minDist = distance
                        minIndex = keys
                if float(clusterAssment[i, 0]) != float(minIndex):
                    clusterChange = True
                    if temp_index == minIndex:
                        clusterAssment[i, :] = float(temp_index), minDist
                        temp = [np.array(str(item)) for item in dataSet[i]]
                        temp.append(str(minIndex))
                        if temp not in train_dataset:
                            train_dataset.append(temp)

                    else:
                        clusterAssment[i, :] = float(temp_index), self.distEclud(dataSet[i], centroids[temp_index])

            # 遍历所有质心并更新它们的取值
            for j in centroids.keys():
                # 通过数据过滤来获得给定簇的所有点
                pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == int(j))[0]]
                # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
                centroids[j] = np.average(pointsInCluster, axis=0)
            if pre_centroids == centroids:
                break
        print("Congratulation,cluster complete!")

        # 返回所有的类质心与点分配结果
        return centroids, clusterAssment

    def showCluster(self,dataSet, k, centroids, clusterAssment):
        m, n = dataSet.shape
        # if n != 2:
        # print("数据不是二维的")
        # return 1
        colors = ['brown', 'green', 'blue', 'y', 'r', 'tan', 'dodgerblue', 'deeppink', 'orangered', 'peru', 'blue', 'y',
                  'r',
                  'gold', 'dimgray', 'darkorange', 'peru', 'blue', 'y', 'r', 'cyan', 'tan', 'orchid', 'peru', 'blue',
                  'y',
                  'r', 'sienna']
        markers = ['*', 'h', 'H', '+', 'o', '1', '2', '3', ',', 'v', 'H', '+', '1', '2', '^',
                   '&lt;', '&gt;', '.', '4', 'H', '+', '1', '2', 's', 'p', 'x', 'D', 'd', '|', '_']

        mark = ['v', 'H', '+', '1', '2', '^']
        if k > len(mark):
            print("k值太大了")
            return 1
        # 绘制所有样本
        for i in range(m):
            markIndex = int(clusterAssment[i, 0])
            plt.plot(dataSet[i, 0], dataSet[i, 1], marker=markers[markIndex], color=colors[markIndex])

        # 绘制质心
        for i in centroids.keys():
            plt.plot(centroids[i][0], centroids[i][1], marker='*')

        plt.show()
norm = Cal_Norm()
seeds_kmeans = Seeds_Kmeans()
data = np.loadtxt(open('wine_test_data.csv','rb'),delimiter=',',skiprows=0)
data1 = K_nearest().get_data('wine_test_data_train.csv',',')
k = 3
centroids,clusterAssment = seeds_kmeans.KMeans(data[:,:13],data1,k)
seeds_kmeans.showCluster(data,k,centroids,clusterAssment)
print(norm.purity(data[:,13],clusterAssment[:,0]))
print(metrics.normalized_mutual_info_score(data[:,13],np.array(clusterAssment)[:,0].reshape(1,-1)[0]))