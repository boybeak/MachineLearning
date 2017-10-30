# ML 入门：交叉验证与网格搜索算法

原文地址:[ML 入门：交叉验证与网格搜索算法](http://www.davex.pw/2017/09/17/Cross-Validation/)

交叉验证（Cross Validation）和网格搜索（Grid Search）是机器学习两大法宝，前者用于检验模型的好坏，后者用于模型的调参。

为什么我们要把交叉验证放在前面，而不是后面呢？

因在探索网格搜索算法之前，我们需要了解**一个评估模型好坏的方法**。

就比如说，你要知道一辆车的好坏，我们需要一个客观公正的评价方法，不然车厂说自己的车好，顾客说这个车坏，我们就无法得到一辆车客观的好坏。

在机器学习领域中，我们经常使用一种叫做交叉验证的方法来帮助我们判断模型好坏，那么我们接下来就探讨什么是交叉验证以及怎么使用。

## 0x01 交叉验证 Cross Validation

交叉验证（Cross Validation）会把一份数据随机分成三个部分：训练集（training set）、验证集（validation set）、测试集（test set）。

其中，**训练集用来训练模型**，**验证集用于模型的选择**，**测试集用于最终对学习方法的评估**。

我们通俗点讲，以高考为例：

**训练集**就是我们平时做的**作业**，用于训练自己的能力；

**验证集**就是我们的**月考、模拟考**，用于检验和反馈自己的能力；

**测试集**就是我们的**高考**，绝对保密，用于最终告诉你的能力水平（分数）；

**以上就是交叉验证的解释，也是我们对模型进行检验的最简单常见方法之一。**

在交叉验证的基础之上，人们发明了 K 折交叉验证，帮助人们更好的对模型进行检验与调优。

举个例子，我们首先将一份数据按照 8:2 划分出训练集和测试集。（这一步没有验证集）

[![CV Demo](http://www.davex.pw/images/ML_cross_validation/cv_demo.png)](http://www.davex.pw/images/ML_cross_validation/cv_demo.png)

这里的测试集就像我们说的高考题，是**绝对保密不参与训练过程的**，用于最后检验（模型）能力好坏的。

因此，在K折交叉验证中，我们用到的数据是训练集中的所有数据。我们将训练集的所有数据平均划分成K份（通常选择K=10），取第 K 份作为验证集，它的作用就像我们用来估计高考分数的模拟题，余下的 K-1 份作为交叉验证的训练集。

训练过程如下：

以 max_depth=1 的决策树为例，我们先用第 2-10 份数据作为训练集训练模型，用第 1 份数据作为验证集对这次训练的模型进行评分，得到第一个分数；

然后重新构建一个 max_depth=1 的决策树，，用第 1 和 3-10 份数据作为训练集训练模型，用第 2 份数据作为验证集对这次训练的模型进行评分，得到第二个分数…

以此类推，最后构建一个 max_depth=1 的决策树，，用第 1-9 份数据作为训练集训练模型，用第 10 份数据作为验证集对这次训练的模型进行评分，得到第十个分数。

于是对于 max_depth=1 的决策树，我们训练了10次，验证了10次，得到了10个验证分数，然后计算这10个验证分数的平均分数，就是 max_depth=1 的决策树模型的最终验证分数。

**10 个验证分数的平均分数是模型的最终验证分数。（记住是验证集的平均分数）**

[![CV Full](http://www.davex.pw/images/ML_cross_validation/cv_full.png)](http://www.davex.pw/images/ML_cross_validation/cv_full.png)

学会了怎么检验模型的方法，那么我们就来谈谈如何基于这个检验方法来进行调参。

## 0x02 网格搜索算法 Grid Search

网格搜索法算法就是**通过交叉验证的方法去寻找最优的模型参数**。

详细点说就是模型的每个参数有很多个候选值，我们每个参数组合做一次交叉验证，最后得出交叉验证分数最高的，就是我们的最优参数。

以决策树为例，当我们确定了要使用决策树算法的时候，为了能够更好地拟合和预测，我们需要调整它的参数。在决策树算法中，我们通常选择的参数是决策树的最大深度。

于是我们会给出一系列的最大深度的值，比如 {‘max_depth’: [1,2,3,4,5]}，我们会尽可能包含最优最大深度。

那么我们就对 max_depth = 1,2,3,4,5 的模型分别进行上述的交叉验证过程，得到它们的最终验证分数。

然后我们就可以对这 5 个最大深度的决策树的最终验证分数进行比较，分数最高的那一个就是最优最大深度，对应的模型就是最优模型。

下面提供一个简单的利用决策树预测[乳腺癌](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))的例子：

```
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data['data'], data['target'], train_size=0.8, random_state=0)

regressor = DecisionTreeClassifier(random_state=0)
parameters = {'max_depth': range(1, 6)}
scoring_fnc = make_scorer(accuracy_score)
kfold = KFold(n_splits=10)

grid = GridSearchCV(regressor, parameters, scoring_fnc, cv=kfold)
grid = grid.fit(X_train, y_train)
reg = grid.best_estimator_

print('best score: %f'%grid.best_score_)
print('best parameters:')
for key in parameters.keys():
    print('%s: %d'%(key, reg.get_params()[key]))

print('test score: %f'%reg.score(X_test, y_test))

import pandas as pd
pd.DataFrame(grid.cv_results_).T
```

直接用决策树得到的分数大约是92%，经过网格搜索优化以后，我们可以在测试集得到95.6%的准确率：

> best score: 0.938462
> best parameters:
> max_depth: 4
> test score: 0.956140

## 0x03 Reference

[网格搜索算法与K折交叉验证 - 杨培文](https://ypw.io/GridSearchCV/)

[Parameter Tuning in Gradient Boosting](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)

[Parameter Tuning in XGBoost](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

[# 机器学习](http://www.davex.pw/tags/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/) 