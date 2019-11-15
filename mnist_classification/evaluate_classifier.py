'''
对性能的评估
'''
# 评估一个模型的好方法是使用交叉验证测量准确性

from classification import *
from sklearn.model_selection import cross_val_score

print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


'''
混淆矩阵
'''
# 大体思路是：输出类别A被分类成类别 B 的次数。
# 为了计算混淆矩阵，首先你需要有一系列的预测值，这样才能将预测值与真实值做比较。
# cross_val_predict() 也使用 K 折交叉验证。它不是返回一个评估分数，而是返回基于每一个测试折做出的一个预测值。

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# 现在使用 confusion_matrix() 函数，你将会得到一个混淆矩阵。
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5, y_train_pred)) # 混淆矩阵中的每一行表示一个实际的类, 而每一列表示一个预测的类

# 该矩阵的第一行认为“非5”（反例）中的 53272 张被正确归类为 “非 5”（他们被称为真反例，true negatives）, 而其余
# 1307 被错误归类为"是 5" （假正例，false positives）。第二行认为“是 5” （正例）中的 1077被错误地归类为“非 5”
# （假反例，false negatives），其余 4344 正确分类为 “是 5”类（真正例，true positives）。


'''
准确率与召回率
'''

# 当它声明某张图片是 5 的时候，它只有 87% 的可能性是正确的。
# 而且，它也只检测出“是 5”类图片当中的 69%。

from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))

# 通常结合准确率和召回率会更加方便，这个指标叫做“F1 值”，特别是当你需要一个简单的方
# 法去比较两个分类器的优劣的时候。F1 值是准确率和召回率的调和平均。普通的平均值平等
# 地看待所有的值，而调和平均会给小的值更大的权重。所以，要想分类器得到一个高的 F1
# 值，需要召回率和准确率同时高。

from sklearn.metrics import f1_score
print(f1_score(y_train_5, y_train_pred))

y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

# SGDClassifier 用了一个等于 0 的阈值，所以前面的代码返回了跟 predict() 方法一样的结果
# （都返回了 true ）。

threshold = 200000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

# cross_val_predict() 得到每一个样例的分数值，但是
# 这一次指定返回一个决策分数，而不是预测值。
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

from matplotlib import pyplot as plt

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# 我们假设你决定达到 90% 的准确率。你查阅第一幅图（放大一些），在 70000 附近找到一个
# 阈值。
y_train_pred_90 = (y_scores > 70000)
# 检查这些预测的准确率和召回率：
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))

'''
ROC 曲线
'''
# ROC 曲线是真正例率（true positive rate，另一个名字叫做召回率）对假正例率（false positive rate, FPR）的曲线。
# FPR是反例被错误分成正例的比率。它等于 1 减去真反例率（true negative rate， TNR）。TNR是反例被正确分类的比
# 率。TNR也叫做特异性。

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()