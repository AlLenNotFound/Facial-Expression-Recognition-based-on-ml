import os
import time

import numpy as np
import cv2
from PIL._imaging import display
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn
from sklearn import svm
import sklearn.model_selection as ms
from sklearn.svm import SVC
from skimage.feature import hog, local_binary_pattern
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
from sympy.polys.groebnertools import lbp


time_start = time.time()    # 记录开始时间


def mask(img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    bgdModel = np.zeros((1, 65), dtype=np.float64)
    fgdModel = np.zeros((1, 65), dtype=np.float64)
    mask[200:4000, 10:3900] = 3
    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    ogc = img * mask2[:, :, np.newaxis]
    return ogc


def EdgeDetect(image):
    core_Sobel_y = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    core_Sobel_x = np.array(([1, 2, 1], [0, 0, 0], [-1, -2, -1]))
    y = cv2.filter2D(image, -1, core_Sobel_y)
    x = cv2.filter2D(image, -1, core_Sobel_x)
    k = y + x
    return k


def preprocessing(src):  # 预处理-图像读入并归一化处理
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 将图像转换成灰度图
    img = cv2.resize(gray, (256, 256))  # 尺寸调整g
    img = img / 255.0  # 数据归一化
    return img


def extract_hog_features(X):
    # hog算法的区域特征提取，先进行Gramma归一,随后计算区域的每个像素点的梯度的大小和方向并统计出分布直方图
    image_descriptors = []
    for i in range(len(X)):
        fd, _ = hog(X[i], orientations=9, pixels_per_cell=(16, 16), cells_per_block=(16, 16),
                    block_norm='L2-Hys', visualize=True)
        image_descriptors.append(fd)  # 拼接得到所有图像的hog特征
    return image_descriptors  # 返回的是训练部分所有图像的hog特征


def extract_hog_features_single(X):
    # 单张图片的特征提取，和上一个一样
    image_descriptors_single = []
    fd, _ = hog(X, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(16, 16), block_norm='L2-Hys',
                visualize=True)
    image_descriptors_single.append(fd)
    return image_descriptors_single


def LBP_group(X):
    R = 2
    P = 8 * R
    image_descriptors_single = []
    for i in range(len(X)):
        res = local_binary_pattern(X[i], P, R, method='uniform')
        image_descriptors_single.append(np.array(res.flatten()))
    return image_descriptors_single


def LBP(image):
    R = 2
    P = 8 * R
    image_descriptors_single = []
    res = local_binary_pattern(image, P, R, method='uniform')
    image_descriptors_single.append(np.array(res.flatten()))
    return image_descriptors_single


def read_data(label2id):  # label2id为定义的标签
    X = []
    Y = []
    path = './jaffe'
    for label in os.listdir(path):  # os.listdir用于返回指定的文件夹包含的文件或文件夹的名字的列表，此处遍历每个文件夹
        for img_file in os.listdir(os.path.join(path, label)):  # 遍历每个表情文件夹下的图像
            image = cv2.imread(os.path.join(path, label, img_file))  # 读取图像
            if image is not None:
                result = preprocessing(image)
                X.append(result)  # 将读取到的所有图像的矩阵形式拼接在一起
                Y.append(label2id[label])  # 将读取到的所有图像的标签拼接在一起
    return X, Y  # 返回的X,Y分别是图像的矩阵表达和图像的标签


label2id = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
X, Y = read_data(label2id)
X_features = extract_hog_features(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.3, random_state=42)
'''
# 随机森林 超参数遍历
param_grid = dict(max_depth=[1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20]
                  , min_samples_leaf=[1, 2, 3, 4]
                  , n_estimators=[5, 10, 30, 50, 75, 100, 150, 200])
grid = GridSearchCV(RandomForestClassifier(criterion='gini'), param_grid=param_grid, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'gini:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))
# 'gini:'The best parameters are {'max_depth': 11, 'min_samples_leaf': 1, 'n_estimators': 200} with a score of 0.69770
param_grid = dict(max_depth=[1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20]
                  , min_samples_leaf=[1, 2, 3, 4]
                  , n_estimators=[5, 10, 30, 50, 75, 100, 150, 200])
grid = GridSearchCV(RandomForestClassifier(criterion='entropy'), param_grid=param_grid, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'entropy:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))
'''

# 逻辑回归超参数遍历
'''
param_grid = dict(fit_intercept=[0.1, 0.5, 1, 1.5, 3, 5]
                  , C=[0.01, 0.05, 0.1, 0.5, 1, 1.5, 3, 5])
grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid=param_grid, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'liblinear:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))
# 'liblinear:'The best parameters are {'C': 5, 'fit_intercept': 1} with a score of 0.56299
param_grid = dict(fit_intercept=[0.1, 0.5, 1, 1.5, 3, 5]
                  , C=[0.01, 0.05, 0.1, 0.5, 1, 1.5, 3, 5])
grid = GridSearchCV(LogisticRegression(solver='lbfgs'), param_grid=param_grid, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'lbfgs:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))
# 'lbfgs:'The best parameters are {'C': 5, 'fit_intercept': 1} with a score of 0.59655

param_grid = dict(fit_intercept=[0.1, 0.5, 1, 1.5, 3, 5]
                  , C=[0.01, 0.05, 0.1, 0.5, 1, 1.5, 3, 5])
grid = GridSearchCV(LogisticRegression(solver='newton-cg'), param_grid=param_grid, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'newton-cg:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))
# 'newton-cg:'The best parameters are {'C': 5, 'fit_intercept': 1} with a score of 0.59655

param_grid = dict(fit_intercept=[0.1, 0.5, 1, 1.5, 3, 5]
                  , C=[0.01, 0.05, 0.1, 0.5, 1, 1.5, 3, 5])
grid = GridSearchCV(LogisticRegression(solver='saga'), param_grid=param_grid, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'saga:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))
# 'saga:'The best parameters are {'C': 5, 'fit_intercept': 1} with a score of 0.59655
'''
'''
# KNN 超参数遍历
param_grid = dict(n_neighbors=[1, 3, 5, 6, 7, 8, 9, 10]
                  , weights=['uniform', 'distance']
                  , leaf_size=[5, 10, 15, 20, 30])
cv = KFold(n_splits=5, shuffle=True, random_state=520)
grid = GridSearchCV(KNeighborsClassifier(algorithm='brute'), param_grid=param_grid, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'brute:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))


param_grid = dict(n_neighbors=[1, 3, 5, 6, 7, 8, 9, 10]
                  , weights=['uniform', 'distance']
                  , leaf_size=[5, 10, 15, 20, 30])
cv = KFold(n_splits=5, shuffle=True, random_state=520)
grid = GridSearchCV(KNeighborsClassifier(algorithm='kd_tree'), param_grid=param_grid, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'kd_tree:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))


param_grid = dict(n_neighbors=[1, 3, 5, 6, 7, 8, 9, 10]
                  , weights=['uniform', 'distance']
                  , leaf_size=[5, 10, 15, 20, 30])
cv = KFold(n_splits=5, shuffle=True, random_state=520)
grid = GridSearchCV(KNeighborsClassifier(algorithm='ball_tree'), param_grid=param_grid, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'ball_tree:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))


# 决策树 超参数遍历
param_grid = dict(max_depth=[1, 5, 10, 15, 20, 30, 50, 100], min_samples_leaf=[1, 2, 3, 4])
grid = GridSearchCV(DecisionTreeClassifier(criterion='gini'), param_grid=param_grid, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'gini:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))


param_grid = dict(max_depth=[1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20], min_samples_leaf=[1, 2, 3, 4])
cv = KFold(n_splits=5, shuffle=True, random_state=520)
grid = GridSearchCV(DecisionTreeClassifier(criterion='entropy'), param_grid=param_grid, cv=cv, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'entropy:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))

# SVM 超参数选择
gamma_range = np.logspace(-10, 1, 10)
coef0_range = np.linspace(0, 5, 10)
C_range = np.linspace(0.01, 30, 10)
degree_range = np.linspace(0, 10, 11)
param_grid = dict(gamma=gamma_range
                  , coef0=coef0_range
                  , C=C_range
                  , degree=[2, 3]
                  )
cv = KFold(n_splits=5, shuffle=True, random_state=520)
grid = GridSearchCV(SVC(kernel="poly"), param_grid=param_grid, cv=cv, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'poly:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))

gamma_range = np.logspace(-10, 1, 10)
coef0_range = np.linspace(0, 5, 10)
C_range = np.linspace(0.01, 30, 20)
param_grid = dict(gamma=gamma_range
                  , coef0=coef0_range
                  , C=C_range
                  )
cv = KFold(n_splits=5, shuffle=True, random_state=520)
grid = GridSearchCV(SVC(kernel="sigmoid"), param_grid=param_grid, cv=cv, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'sigmoid:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))

gamma_range = np.logspace(-10, 1, 10)
C_range = np.linspace(0.01, 30, 20)
param_grid = dict(gamma=gamma_range
                  , C=C_range
                  )
cv = KFold(n_splits=5, shuffle=True, random_state=520)
grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv, n_jobs=-1)  # n_jobs = -1 调用所有线程
grid.fit(X_train, Y_train)
print("'rbf:'The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))

C_range = np.linspace(0.01, 30, 30)
best_acc = -1
best_c = -1
for c in C_range:
    clf = SVC(kernel="linear", C=c)
    clf.fit(X_train, Y_train)
    acc = clf.score(X_test, Y_test)
    if acc > best_acc:
        best_acc = acc
        best_c = c
print("'linear:'The best c is %0.5f with a score of %0.5f" % (best_c, best_acc))
'''
# X_features = LBP_group(X)
# X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.3, random_state=42)
'''
knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=5, n_neighbors=1, weights='uniform')  # k取1，最近邻准确率较高
knn.fit(X_train, Y_train)
Y_predict = knn.predict(X_test)
acc = accuracy_score(Y_test, Y_predict)
print('KNN准确率为: ', acc)

tree_D = DecisionTreeClassifier(criterion='gini', max_depth=10)
tree_D.fit(X_train, Y_train)
Y_predict = tree_D.predict(X_test)
acc = accuracy_score(Y_test, Y_predict)
print('决策树准确率为: ', acc)

logistic = LogisticRegression(solver='newton-cg', C=5)
logistic.fit(X_train, Y_train)
Y_predict = logistic.predict(X_test)
acc = accuracy_score(Y_test, Y_predict)
print('逻辑回归准确率为: ', acc)

Forest = RandomForestClassifier(criterion='entropy', max_depth=6, n_estimators=200, random_state=0)
Forest.fit(X_train, Y_train)
Y_predict = Forest.predict(X_test)
acc = accuracy_score(Y_test, Y_predict)
print('随机森林准确率为: ', acc)
'''
svm = SVC(C=15.52, kernel='linear')
svm.fit(X_train, Y_train)
Y_predict = svm.predict(X_test)
acc = accuracy_score(Y_test, Y_predict)
print('SVM准确率为: ', acc)

cm = confusion_matrix(Y_test, Y_predict)
xtick = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
ytick = xtick
f, ax = plt.subplots(figsize=(7, 5))
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=15)

seaborn.set(font_scale=1.2)
plt.rc('font', family='Times New Roman', size=15)
seaborn.heatmap(cm, fmt='g', cmap=plt.cm.gray_r, annot=True, cbar=True, xticklabels=xtick, yticklabels=ytick, ax=ax)
plt.title('Confusion Matrix', fontsize='x-large')
f.savefig('./混淆矩阵.png')
# plt.show()
svm = SVC(C=15.52, kernel='linear')
svm.fit(X_train, Y_train)


path = './source/test_pic.jpg'
image = cv2.imread(path)
result = preprocessing(image)
Mask = mask(image)
# cv2.imshow('mask', Mask)
# cv2.waitKey()
img_gray = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img_gray', img_gray)
# cv2.waitKey()
for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        if img_gray[i][j] == 0:
            # print(img_gray[i][j])
            img_gray[i][j] = 255
# cv2.imshow('img_gray_processed', img_gray)
# cv2.waitKey()
img_gray = cv2.resize(img_gray, (256, 256))  # 尺寸调整g
X_Single = extract_hog_features_single(img_gray)
print(X_Single)
#这里选择分类器的类别
predict = svm.predict(X_Single)

time_end = time.time()      # 记录结束时间
time_sum = time_end - time_start
print(time_sum)
if predict == 0:
    image = cv2.resize(image, (256, 256))
    cv2.imshow("angry", image)
    cv2.waitKey(0)
    print('angry')
elif predict == 1:
    image = cv2.resize(image, (256, 256))
    cv2.imshow("disgust", image)
    cv2.waitKey(0)
    print('disgust')
elif predict == 2:
    image = cv2.resize(image, (256, 256))
    cv2.imshow("fear", image)
    cv2.waitKey(0)
    print('fear')
elif predict == 3:
    image = cv2.resize(image, (256, 256))
    cv2.imshow("happy", image)
    cv2.waitKey(0)
    print('happy')
elif predict == 4:
    image = cv2.resize(image, (256, 256))
    cv2.imshow("neutral", image)
    cv2.waitKey(0)
    print('neutral')
elif predict == 5:
    image = cv2.resize(image, (256, 256))
    cv2.imshow("sad", image)
    cv2.waitKey(0)
    print('sad')
elif predict == 6:
    image = cv2.resize(image, (256, 256))
    cv2.imshow("surprise", image)
    cv2.waitKey(0)
    print('surprise')

'''
x = LBP(result)
# print(x)
predict = svm.predict(x)
if predict == 0:
    cv2.imshow("angry", image)
    cv2.waitKey(0)
    print('angry')
elif predict == 1:
    cv2.imshow("disgust", image)
    cv2.waitKey(0)
    print('disgust')
elif predict == 2:
    cv2.imshow("fear", image)
    cv2.waitKey(0)
    print('fear')
elif predict == 3:
    cv2.imshow("happy", image)
    cv2.waitKey(0)
    print('happy')
elif predict == 4:
    cv2.imshow("neutral", image)
    cv2.waitKey(0)
    print('neutral')
elif predict == 5:
    cv2.imshow("sad", image)
    cv2.waitKey(0)
    print('sad')
elif predict == 6:
    cv2.imshow("surprise", image)
    cv2.waitKey(0)
    print('surprise')
'''