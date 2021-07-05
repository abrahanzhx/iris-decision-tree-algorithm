"""1.数据获取，划分训练集测试集"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
# 数据获取
# 加载 iris 数据集

iris = datasets.load_iris()
print(iris)
# 特征数据
iris_feature = iris.data
# 分类数据
iris_target = iris.target
print('类别：',iris.target_names)
print('特征：',iris.feature_names)
iris_features = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# 划分训练集测试集: test_size测试集占比, random_stat为数据混乱程度
feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33, random_state=42)
print(f'实验： 数据集{len(iris_target)}条(训练集{len(target_train)}条，测试集{len(target_test)}条)')
#进行浅拷贝，防止对于原始数据的修改
iris_all_df = iris_features.copy()
iris_all_df['target'] = iris_target
iris_all = iris_all_df.values
print(iris_all)
"""2.模型训练->模型预测->模型评估"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 模型训练
dt_model = DecisionTreeClassifier(criterion='entropy',
                                  ) # 所以参数均置为默认状态
dt_model.fit(feature_train,target_train) # 使用训练集训练模型
# 模型预测
predict_results = dt_model.predict(feature_test) # 使用模型对测试集进行预测
# 模型评估
print('精确率(Precision)：',accuracy_score(predict_results, target_test))

"""4.模型可视化"""
import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(dt_model,
                                feature_names=['sepal length','sepal width','petal length','petal width'], # 特征名称
                                class_names=['setosa', 'versicolor', 'virginica'], # 目标变量的类别名
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(source=dot_data,filename='iris_dt_model.gv',format='png',)
# 保存模型可视化为png图片
graph.render(cleanup=True)
def entropy(iris_data):
    p0_count = 0
    p1_count = 0
    p2_count = 0
    count = sum(iris_data)
    for t in iris_data:
        if (t == 0):
            p0_count += 1
        elif (t == 1):
            p1_count += 1
        elif (t == 2):
            p2_count += 1
    # 计算各种类鸢尾花出现的概率
    p0 = p0_count / count
    p1 = p1_count / count
    p2 = p2_count / count
    # 计算鸢尾花数据集的信息熵
    ent = -(p0 * math.log(p0) / math.log(2) + p1 * math.log(p1) / math.log(2) + p2 * math.log(p2) / math.log(2))
    print(ent)
    return ent
# 鸢尾花数据集是连续的，而ID3算法是应用于离散型的特征，所以采用二分法策略，遍历每一个属性的可能值，寻找最佳选项。

