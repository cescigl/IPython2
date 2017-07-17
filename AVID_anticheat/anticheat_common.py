# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, FactorAnalysis,KernelPCA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction import DictVectorizer



def makeDataMat(dic):
    idMat = []
    dataMat = []
    for key in dic:
        dataMat.append(dic[key])
        idMat.append(key)
    return idMat, dataMat

def makeVec(mat):
    vec = DictVectorizer()
    dataVec = vec.fit_transform(mat).toarray()
    print dataVec.shape
    #print dataVec[0]
    return dataVec


#规范化数据
def scalerMaxMin(dm):
    '''将属性缩放到一个指定的最大和最小值（通常是1-0）之间'''
    X_train = np.array(dm)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    return X_train_minmax
    #将相同的缩放应用到测试集数据中
    #X_test = np.array([[ -3., -1.,  4.]])
    #X_test_minmax = min_max_scaler.transform(X_test)

def scale(dm):
    '''    公式为：(X-mean)/std  计算时对每个属性/每列分别进行。
    使用sklearn.preprocessing.StandardScaler类，使用该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据。
    '''
    X_train = np.array(dm)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scale=scaler.transform(X_train)
    return X_train_scale
    #可以直接使用训练集对测试集数据进行转换
    #scaler.transform([[-1.,  1., 0.]])

def normalization(dm):
    '''正则化的过程是将每个样本缩放到单位范数（每个样本的范数为1）'''
    X_train = np.array(dm)
    normalizer = preprocessing.Normalizer().fit(X_train)  # fit does nothing
    X_train_norm=normalizer.transform(X_train)
    return X_train_norm
    #可以直接使用训练集对测试集数据进行转换
    normalizer.transform([[-1.,  1., 0.]])


#kmean聚类
def k_means(data,h=1,dm=[],dmid=[]):
    data=data
    #dm=np.array(dm)
    #dataid=np.array(dmid)
    model1 = KMeans(init='k-means++',n_clusters=2,n_init=10)
    model1.fit(data)

    model2 = KMeans(init='random',n_clusters=2,n_init=10)
    model2.fit(data)


    ###############################################################################
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    print x_min, x_max
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
    print y_min, y_max
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Obtain labels for each point in mesh. Use last trained model.
    print np.c_[xx.ravel(), yy.ravel()]
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    #new_mat=np.column_stack((dm,kmeans.labels_))
    #np.savetxt("output.txt",model2.labels_,fmt="%d",delimiter="\t")


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def k_means2(data,h=1,dm=[],dmid=[]):
    data=data
    #dm=np.array(dm)
    #dataid=np.array(dmid)
    model1 = KMeans(init='k-means++',n_clusters=2,n_init=10)
    model1.fit(data)

    model2 = KMeans(init='random',n_clusters=2,n_init=10)
    model2.fit(data)


    ###############################################################################
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 1     # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    print x_min, x_max
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
    print y_min, y_max
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Obtain labels for each point in mesh. Use last trained model.
    print np.c_[xx.ravel(), yy.ravel()]
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    #new_mat=np.column_stack((dm,kmeans.labels_))
    #np.savetxt("uncheat_neg.txt",kmeans.labels_,fmt="%d",delimiter="\t")


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def k_means_kernal_pca(data,h=1):
    data=data
    ###############################################################################
    # Visualize the results on PCA-reduced data

    kpca = KernelPCA(n_components=2,kernel="rbf", fit_inverse_transform=True, gamma=10)
    X_kpca = kpca.fit_transform(data)

    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
    kmeans.fit(X_kpca)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X_kpca[:, 0].min()-0.1, X_kpca[:, 0].max()+0.1
    print x_min, x_max
    y_min, y_max = X_kpca[:, 1].min()-0.1, X_kpca[:, 1].max()+0.1
    print y_min, y_max
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Obtain labels for each point in mesh. Use last trained model.
    print np.c_[xx.ravel(), yy.ravel()]
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])



    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')

    plt.plot(X_kpca[:, 0], X_kpca[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def dbscan(dm):
    reduced_data = KernelPCA(n_components=2,kernel="rbf", fit_inverse_transform=True, gamma=10).fit_transform(dm)
    db = DBSCAN(eps=1.5, min_samples=60).fit(reduced_data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(reduced_data, labels))
    # Plot result
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = reduced_data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)

        xy = reduced_data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def noveltyDetection(dm):
    reduced_data = PCA(n_components=2).fit_transform(dm)
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(reduced_data)
    x_min, x_max = reduced_data[:, 0].min()-10, reduced_data[:, 0].max()+2

    y_min, y_max = reduced_data[:, 1].min()-2, reduced_data[:, 1].max()+2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.title("Novelty Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

    b1 = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='white')
    plt.axis('tight')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def mahalanobisDistances(dm):
    reduced_data = PCA(n_components=2).fit_transform(dm)
    robust_cov = MinCovDet().fit(reduced_data)

    emp_cov = EmpiricalCovariance().fit(reduced_data)
    fig = plt.figure()
    plt.subplots_adjust(hspace=-.1, wspace=.4, top=.95, bottom=.05)
    subfig1 = plt.subplot(3, 1, 1)
    inlier_plot = subfig1.scatter(reduced_data[:, 0], reduced_data[:, 1], color='black', label='inliers')

    subfig1.set_xlim(subfig1.get_xlim()[0], 11.)
    subfig1.set_title("Mahalanobis distances of a contaminated data set:")

    # Show contours of the distance functions
    xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
                     np.linspace(plt.ylim()[0], plt.ylim()[1], 100))
    zz = np.c_[xx.ravel(), yy.ravel()]

    mahal_emp_cov = emp_cov.mahalanobis(zz)
    mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
    emp_cov_contour = subfig1.contour(xx, yy, np.sqrt(mahal_emp_cov),
                                  cmap=plt.cm.PuBu_r,
                                  linestyles='dashed')

    mahal_robust_cov = robust_cov.mahalanobis(zz)
    mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
    robust_contour = subfig1.contour(xx, yy, np.sqrt(mahal_robust_cov),
                                 cmap=plt.cm.YlOrBr_r, linestyles='dotted')

    plt.xticks(())
    plt.yticks(())

    # Plot the scores for each point
    emp_mahal = emp_cov.mahalanobis(reduced_data - np.mean(reduced_data, 0)) ** (0.33)
    subfig2 = plt.subplot(2, 2, 3)

    plt.yticks(())

    robust_mahal = robust_cov.mahalanobis(reduced_data - robust_cov.location_) ** (0.33)
    subfig3 = plt.subplot(2, 2, 4)

    plt.yticks(())

    plt.show()

def plotData(data,label,h=1):
    data=data
    ###############################################################################
    # Visualize the results on PCA-reduced data
    reduced_data = PCA(n_components=2).fit_transform(data)

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    print x_min, x_max
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
    print y_min, y_max
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Obtain labels for each point in mesh. Use last trained model.
    print np.c_[xx.ravel(), yy.ravel()]

    label = np.array(label)
    pos = np.where(label==1)
    x1=data[pos][:,3]
    y1=data[pos][:,4]

    label = np.array(label)
    pos = np.where(label==0)
    x2=data[pos][:,3]
    y2=data[pos][:,4]

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.plot(x1,y1,'ro',color='red',label='a')
    plt.plot(x2,y2,'ro',color='green',label='b')
    # Plot the centroids as a white X
    plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.grid(True)
    plt.show()

def KDE(data):
    from sklearn.datasets import load_digits
    from sklearn.neighbors import KernelDensity
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV

    data = data
    params = {'bandwidth': np.logspace(-3, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(data)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    plt.plot()


#k_means(data)
#k_means_kernal_pca(data)
#dbscan(dataVec)
#noveltyDetection(data)
#mahalanobisDistances(data)

#对细分结果再聚类（正样本）
#dm=loadDataSetPositive('result2.log')
#data=scale(dm)
#k_means2(data,dm,dmid)
#dbscan(data)

N = 10
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, 3 * N),
                    np.random.normal(5, 1, 7 * N)))[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
        label='input distribution')

for kernel in ['gaussian', 'tophat', 'epanechnikov']:
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
            label="kernel = '{0}'".format(kernel))

ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc='upper left')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
plt.show()




