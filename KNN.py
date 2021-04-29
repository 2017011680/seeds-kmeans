import csv
import numpy as np
class K_nearest(object):
    def get_data(self,path,Separator):
        f = open(path, "r")
        aver = []
        for line in f:
            Item = line.strip().split(Separator)
            n = len(Item)
            a = list(map(float, Item[0:n - 1]))
            a.append(Item[n - 1])
            aver.append(a)
        return np.array(aver)
    def dis(self,aver,aver1):
        aver = np.array(aver)
        aver1 = np.array(aver1)
        return np.linalg.norm(aver-aver1)
    def K_sort(self,aver,aver1):
        d = []
        for i in aver:
            a = list(map(float,i[0:len(i)-1]))
            d.append(self.dis(a,aver1))
        return np.argsort(d),d
    def K_predict(self,data,X,n):
        dis,d = self.K_sort(data,X)
        predict = {}
        m=0
        keys = ''
        for i in range(n):
            if data[dis[i]][len(data[i])-1] in predict.keys():
                predict[data[dis[i]][len(data[i])-1]] = predict[data[dis[i]][len(data[i])-1]]+1
            else:
                predict[data[dis[i]][len(data[i])-1]] = 1
        for key in predict.keys():
            if int(predict[key]) > m:
                m=predict[key]
                keys = key
        return keys
    def Kw_predict(self,data,X,n):
        dis,d = self.K_sort(data,X)
        d1 = {}
        keys = ''
        m = 0
        # print(dis)
        # print(n-1)
        # print(dis[n-1])
        fm = d[dis[n-1]]-d[dis[0]]
        for i in range(n):
            w = (d[dis[n-1]]-d[dis[i]])/fm
            if data[dis[i]][len(data[dis[i]])-1] in d1.keys():
                d1[data[dis[i]][len(data[dis[i]])-1]] = d1[data[dis[i]][len(data[dis[i]])-1]]+w
            else:
                d1[data[dis[i]][len(data[dis[i]])-1]] = w
        for key in d1.keys():
            if d1[key] > m:
                m = d1[key]
                keys = key
        return keys
    def compare_KandKw(self,data,data1,n):
        '''
        :param data: 训练数据
        :param data1: 比较数据
        :param n: n邻近
        :return:
        '''
        K = {'True':[],'False':[]}
        Kw = {'True':[],'False':[]}
        T = 0  # 记录使用不带权重的K邻近分类正确的个数
        F = 0  # 记录使用不带权重的K邻近分类错误的个数
        Tw = 0  # 记录使用带权重的K邻近分类正确的个数
        Fw = 0  # 记录使用带权重的K邻近分类错误的个数
        for i in data1:
            x = list(map(float, i[0:len(i) - 1]))
            x.append(i[len(i) - 1])
            if self.K_predict(data, x, n) == i[len(i) - 1]:
                T += 1
                K['True'].append(x)
            else:
                F += 1
                K['False'].append(x)
            if self.Kw_predict(data, x, n) == i[len(i) - 1]:
                Tw += 1
                Kw['True'].append(x)
            else:
                Fw += 1
                Kw['False'].append(x)
        print('\n不带权重的'+str(n)+'邻近方法:\n\t总的有' + str(T + F) + '个,分对' + str(T) + '个,分错' + str(F) + '个,正确率为:.' + str(
            round(T / (F + T) * 100, 2)) + '%')
        print('错误的为:'+str(K['False']))
        print('带权重的'+str(n)+'邻近方法:\n\t总的有' + str(Tw + Fw) + '个,分对' + str(Tw) + '个,分错' + str(Fw) + '个,正确率为:.' + str(
            round(Tw / (Fw + Tw) * 100, 2)) + '%')
        print('错误的为:' + str(Kw['False']))
        return K,Kw
#x = [5.9,3,5.1,1.8,'virginica']

# k = K_nearest()
# data1 = k.get_data("iris_test.csv")#测试数据
# data = k.get_data("iris_train.csv")#训练数据
# print(data)
# k.compare_KandKw(data,data1,5)
# k.compare_KandKw(data,data1,7)
# k.compare_KandKw(data,data1,9)