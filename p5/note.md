# 一、评估指标

#### 精确率公式

**precision = true-positives/(true-positives + false-positives)**

#### 召回率公式

**recall = true-positives/(true-positives + false-negatives)**



## F1 分数

既然我们已讨论了精确率和召回率，接下来可能要考虑的另一个指标是 F1 分数。F1 分数会同时考虑精确率和召回率，以便计算新的分数。

可将 F1 分数理解为精确率和召回率的加权平均值，其中 F1 分数的最佳值为 1、最差值为 0：

`F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)`

有关 F1 分数和如何在 sklearn 中使用它的更多信息，请查看此链接[此处](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)。



## 回归指标

正如前面对问题的回归类型所做的介绍，我们处理的是根据连续数据进行预测的模型。在这里，我们更关注预测的接近程度。

例如，对于身高和体重预测，我们不是很关心模型能否将某人的体重 100% 准确地预测到小于零点几磅，但可能很关心模型如何能始终进行接近的预测（可能与个人的真实体重相差 3-4 磅）



## 平均绝对误差

您可能已回想起，在统计学中可以使用绝对误差来测量误差，以找出预测值与真实值之间的差距。平均绝对误差的计算方法是，将各个样本的绝对误差汇总，然后根据数据点数量求出平均误差。通过将模型的所有绝对值加起来，可以避免因预测值比真实值过高或过低而抵销误差，并能获得用于评估模型的整体误差指标。

有关平均绝对误差和如何在 sklearn 中使用它的更多信息，请查看此链接[此处](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error)。



## 均方误差

均方误差是另一个经常用于测量模型性能的指标。与绝对误差相比，残差（预测值与真实值的差值）被求平方。

对残差求平方的一些好处是，自动将所有误差转换为正数、注重较大的误差而不是较小的误差以及在微积分中是可微的（可让我们找到最小值和最大值）。

有关均方误差和如何在 sklearn 中使用它的更多信息，请查看此链接[此处](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)。



## 回归分数函数

除了误差指标之外，scikit-learn还包括了两个分数指标，范围通常从0到1，值0为坏，而值1为最好的表现，看起来和分类指标类似，都是数字越接近1.0分数就越好。

其中之一是[R2分数](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)，用来计算真值预测的可决系数。在 scikit-learn 里，这也是回归学习器默认的分数方法。

另一个是[可释方差分数](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score)

虽然眼下我们不会详细探讨这些指标，一个要记住的重点是，回归的默认指标是“分数越高越好”；即，越高的分数表明越好的表现。而当我们用到前面讲的误差指标时，我们要改变这个设定。



# 二、误差原因

## 误差原因

我们已讨论了一些用于测量模型性能的基本指标，现在来关注一下模型起初为何会出现误差。

在模型预测中，模型可能出现的误差来自两个主要来源，即：因模型无法表示基本数据的复杂度而造成的**偏差（bias）**，或者因模型对训练它所用的有限数据过度敏感而造成的**方差（variance）**。我们会对两者进行更详细的探讨。



## 偏差造成的误差 - 准确率和欠拟合

如前所述，如果模型具有足够的数据，但因不够复杂而无法捕捉基本关系，则会出现偏差。这样一来，模型一直会系统地错误表示数据，从而导致准确率降低。这种现象叫做**欠拟合（underfitting）**。

简单来说，如果模型不适当，就会出现偏差。举个例子：如果对象是按颜色和形状分类的，但模型只能按颜色来区分对象和将对象分类（模型过度简化），因而一直会错误地分类对象。

或者，我们可能有本质上是多项式的连续数据，但模型只能表示线性关系。在此情况下，我们向模型提供多少数据并不重要，因为模型根本无法表示其中的基本关系，我们需要更复杂的模型。



## 方差造成的误差 - 精度和过拟合

在训练模型时，通常使用来自较大训练集的有限数量样本。如果利用随机选择的数据子集反复训练模型，可以预料它的预测结果会因提供给它的具体样本而异。在这里，**方差（variance）**用来测量预测结果对于任何给定的测试样本会出现多大的变化。

出现方差是正常的，但方差过高表明模型无法将其预测结果泛化到更多的数据。对训练集高度敏感也称为**过拟合（overfitting）**，而且通常出现在模型过于复杂或我们没有足够的数据支持它时。

通常，可以利用更多数据进行训练，以降低模型预测结果的方差并提高精度。如果没有更多的数据可以用于训练，还可以通过限制模型的复杂度来降低方差。



## 学习曲线

现在你理解了偏差和方差的概念，让我们学习一下如何辨别模型表现的好坏。sklearn中的学习曲线函数可以帮到我们。它可以让我们通过数据点来了解模型表现的好坏。

可以先引入这个模块

```
from sklearn.learning_curve import learning_curve # sklearn 0.17
from sklearn.model_selection import learning_curve # sklearn 0.18

```

文档中一个合理的实现是：

```
 learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
```

这里`estimator`是我们正在用来预测的模型，例如它可以是`GaussianNB()`，`X`和`y`是特征和目标。`cv`是交叉验证生成器，例如`KFold()`，'n_jobs'是平行运算的参数，`train_sizes`是多少数量的训练数据用来生成曲线。



## 改进模型的有效性

我们可以看到，在给定一组固定数据时，模型不能过于简单或复杂。如果过于简单，模型无法了解数据并会错误地表示数据。但是，如果建立非常复杂的模型，则需要更多数据才能了解基本关系，否则十分常见的是，模型会推断出在数据中实际上并不存在的关系。

关键在于，通过找出正确的模型复杂度来找到最大限度降低偏差和方差的最有效点。当然，数据越多，模型随着时间推移会变得越好。

要详细了解偏差和方差，建议阅读 Scott Fortmann-Roe 撰写的[这篇文章](http://scott.fortmann-roe.com/docs/BiasVariance.html)。