import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import scipy.stats as stats
from sklearn import neighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math

def filterRows(target):
    row_filter = [1] * len(target)
    for i in range (0, len(target)):
        if target[i] > 5581 :
            row_filter[i] = 0
    return row_filter

def countZeros(feature):
    count = 0
    for i in range(0, len(feature)):
        x = int(feature[i])
        if x == 0:
            count = count + 1
    return count

def cleanData(feature):
    for i in range(0,len(feature)):
        if feature[i] == 0:
            feature[i] = np.NaN
    return pd.DataFrame(feature)

# extract all predictive features from data 

predictors = np.loadtxt(
    '/home/stevie/hw/csi_hw/4810/project1/OnlineNewsPopularity/OnlineNewsPopularity.csv', delimiter=',',skiprows=1, dtype=float, 
    usecols=(7,9,12,14,16,17,25,26,27,36,37,38,39,43,44,45,46))

shares = np.loadtxt(
    '/home/stevie/hw/csi_hw/4810/project1/OnlineNewsPopularity/OnlineNewsPopularity.csv', delimiter=',',skiprows=1, dtype=float, 
    usecols=(60))

# corrTest = np.loadtxt(
#     '/home/stevie/hw/csi_hw/4810/project1/OnlineNewsPopularity/OnlineNewsPopularity.csv', delimiter=',',skiprows=1, dtype=float, 
#     usecols=(2))

predictor_names = ['num_href','num_img','num_keywords','data_channel_is_entertainment','data_channel_is_socmed', 
'data_channel_is_tech','kw_min_avg','kw_max_avg','kw_avg_avg','weekday_is_saturday','weekday_is_sunday','is_weekend','LDA_00',
'LDA_04','global_subjectivity','global_sentiment_polarity','global_rate_positive_words']
# create data frames to house the features
df_shares = pd.DataFrame(shares)
df_predictors = pd.DataFrame(predictors, columns=predictor_names)
# df_corrTest = pd.DataFrame(corrTest)

# obtain a filter for what rows to exclude due to outliers
row_filter = pd.DataFrame(filterRows(shares))

# Create a new dataframe that only uses the rows where shares =/= outlier values
df_predictors = df_predictors[row_filter.all(axis=1)]
df_shares=df_shares[row_filter.all(axis=1)]
# df_corrTest = df_corrTest[row_filter.all(axis=1)]

# =========================================================
# Using a bucket system for the targets to hopefully improve accuracy. using 20 buckets with ranges 300 wide

cut_shares = pd.cut(df_shares[0], bins=[0,275,550,825,1100,1375,1650,1925,2200,2475,2750,3025,3300,3575,3850,
4125,4400,4575,4950,5225,5500], labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
df_shares[0]=cut_shares

# =========================================================

# data cleaning; determine the mean of each feature and replace its zero values with the mean
# there are some exceptions due to zero values making sense, so they likely aren't placeholders
# df_predictors['num_keywords']=cleanData(df_predictors['num_keywords'])
# df_predictors['kw_min_avg']=cleanData(df_predictors['kw_min_avg'])
# df_predictors['kw_max_avg']=cleanData(df_predictors['kw_max_avg'])
# df_predictors['kw_avg_avg']= cleanData(df_predictors['kw_avg_avg'])
# df_predictors['global_subjectivity']= cleanData(df_predictors['global_subjectivity'])
# df_predictors['global_sentiment_polarity']= cleanData(df_predictors['global_sentiment_polarity'])
# df_predictors['global_rate_positive_words']= cleanData(df_predictors['global_rate_positive_words'])
# df_predictors['LDA_04']= cleanData(df_predictors['LDA_04'])
# df_predictors['LDA_00']= cleanData(df_predictors['LDA_00'])
# # df_predictors['LDA_02']= cleanData(np.array(df_predictors['LDA_02']))

# df_predictors['num_keywords']=df_predictors['num_keywords'].fillna(math.floor(df_predictors['num_keywords'].mean()))
# df_predictors['kw_min_avg']=df_predictors['kw_min_avg'].fillna(df_predictors['kw_min_avg'].mean())
# df_predictors['kw_max_avg']=df_predictors['kw_max_avg'].fillna(df_predictors['kw_max_avg'].mean())
# df_predictors['kw_avg_avg']=df_predictors['kw_avg_avg'].fillna(df_predictors['kw_avg_avg'].mean())
# df_predictors['global_subjectivity']=df_predictors['global_subjectivity'].fillna(df_predictors['global_subjectivity'].mean())
# df_predictors['global_sentiment_polarity']=df_predictors['global_sentiment_polarity'].fillna(df_predictors['global_sentiment_polarity'].mean())
# df_predictors['global_rate_positive_words']=df_predictors['global_rate_positive_words'].fillna(df_predictors['global_rate_positive_words'].mean())
# df_predictors['LDA_04']=df_predictors['LDA_04'].fillna(df_predictors['LDA_04'].mean())
# df_predictors['LDA_00']=df_predictors['LDA_00'].fillna(df_predictors['LDA_00'].mean())
# # df_predictors['LDA_02']=df_predictors['LDA_02'].fillna(df_predictors['LDA_02'].mean())

# determine and print correlation value by pearson method between target and a feature
# print(df_shares.corrwith(df_corrTest, method='pearson', axis=0))


# ================================================================
# dimensionality reduction with PCA
from sklearn import decomposition

pca = decomposition.PCA(n_components=1)
pca.fit(df_predictors)
df_predictors=pca.transform(df_predictors)


# ============================ Normalization and standardization ===============================
# normalization

# minmax= MinMaxScaler()
# minmax.fit(df_predictors)
# x_norm = minmax.transform(df_predictors)

# standardization

# scaler = StandardScaler()
# scaler.fit(x_train,y_train)
# x_stand = scaler.transform(x_train)

# ==============================================================
# Split data into testing and training portions
x_train, x_test, y_train, y_test = train_test_split(df_predictors, df_shares, test_size = 0.1, random_state=0)

# ======================================================================
# perform knn classification
# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors = 15)
# knn.fit(x_train, np.ravel(y_train))
# share_predict=knn.predict(x_test)

# print("accuracy: "+ str(knn.score(x_test,y_test)))

# cvScore= np.mean(cross_val_score(KNeighborsClassifier(), x_train, np.ravel(y_train),cv=10))
# print("cross validation score: "+ str(cvScore))

# ======================================================================
# perform naive bayes

from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
gnb.fit(x_train, np.ravel(y_train))
nb_shares = gnb.predict(x_test)
print("Number of mislabeled points out of a total "+ str(x_test.shape[0])+ " points : " + str((np.ravel(y_test) != nb_shares).sum()))
cvScore= np.mean(cross_val_score(GaussianNB(), x_train, np.ravel(y_train),cv=10))
print("cross validation score: "+ str(cvScore))

# ======================================================================
# perform multilayer perceptron

# from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(hidden_layer_sizes=(5), activation = 'logistic', learning_rate_init=0.05, 
#                     solver='lbfgs',random_state=0,max_iter=200).fit(x_train, np.ravel(y_train))
# result = mlp.predict(x_test)
# marker = np.append(np.ravel(y_train), result, axis=0)

# print("training accuracy: {:.2f}".format(mlp.score(x_train, np.ravel(y_train))))
# y_pred=mlp.predict(x_test)
# print("Test accuracy: {:2f}".format(mlp.score(x_test, y_test)))

# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# x_min, x_max = df_predictors[:,0].min()-1, df_predictors[:, 0].max()+1
# y_min, y_max = df_predictors[:,1].min()-1, df_predictors[:,1].max()+1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
# z =mlp.predict(np.c_[xx.ravel(), yy.ravel()])

# cvScore= np.mean(cross_val_score(MLPClassifier(), x_train, np.ravel(y_train),cv=10))
# print("cross validation score: "+ str(cvScore))

# z=z.reshape(xx.shape)
# pyplot.pcolormesh(xx,yy,z, cmap=cmap_light)
# pyplot.scatter(x_train[:,0], x_train[:,1], c=np.ravel(y_train),s=20)
# pyplot.title("10 hidden neurons")
# pyplot.show()

# ======================================================================
# perform linear regression
# LinReg = LinearRegression()
# LinReg.fit(x_train, y_train)

# share_prediction = LinReg.predict(x_test)

# # calculate and print accuracy evaluators and cross-validation

# mse = np.sum((y_test - share_prediction)**2)/(len(df_shares)-17-1)
# print("mse: "+str(mse))

# regrScore = LinReg.score(x_test, y_test)
# print("regr score: "+str(regrScore))

# cvScore= np.mean(cross_val_score(LinearRegression(), x_train, y_train,cv=20))
# print("cross validation score: "+ str(cvScore))

# =============================================================
# MLP regression

# from sklearn.neural_network import MLPRegressor

# mlp = MLPRegressor(hidden_layer_sizes=(8), max_iter=200, solver = 'lbfgs', alpha=0.02, activation='tanh', random_state =0)
# yfit=mlp.fit(df_predictors, np.array(df_shares))
# y_1 = yfit.predict(x_test)

# cvScore= np.mean(cross_val_score(MLPRegressor(), x_train, y_train,cv=10))
# print("cross validation score: "+ str(cvScore))

# pyplot.figure()
# pyplot.scatter(df_predictors, df_shares, edgecolor="black", c="darkOrange", label="data")
# pyplot.plot(x_test, y_1, '-r', label="predicted", linewidth=2)
# pyplot.legend()
# pyplot.show()

# ====================================================================
# regression tree

# regr_1 = DecisionTreeRegressor(max_depth=4)
# regr_2 = DecisionTreeRegressor(max_depth=8)
# regr_1.fit(df_predictors, df_shares)
# regr_2.fit(df_predictors, df_shares)

# treeResult1=regr_1.predict(x_test)
# treeResult2=regr_2.predict(x_test)

# cvScore= np.mean(cross_val_score(DecisionTreeRegressor(), x_train, y_train,cv=10))
# print("cross validation score depth 4: "+ str(cvScore))

# cvScore= np.mean(cross_val_score(DecisionTreeRegressor(), x_train, y_train,cv=10))
# print("cross validation score depth 8: "+ str(cvScore))

# print(df_predictors.shape)
# print(df_shares.shape)

# pyplot.figure()
# pyplot.scatter(df_predictors, df_shares, edgecolor="black", c="darkOrange", label="data")
# pyplot.plot(x_test, treeResult1, color="cornflowerBlue", label="max depth = 4", linewidth=2)
# pyplot.plot(x_test, treeResult2, color="red", label="max depth = 8", linewidth=2)
# pyplot.legend()
# pyplot.show()

# ===========================================
# count the number of zeros in a given feature to eliminate any feature for which >20% of the fields are 0
# print(countZeros(np.array(df_topics)))
# print(len(np.array(x_df)))


# =================================== Plots and histograms ===================================
# (for feature selection process and preliminary data visualization)
# Create box plot 
# pyplot.boxplot(df_shares)
# pyplot.show()

# create histogram
# pyplot.style.use('ggplot')
# pyplot.hist(df_days, bins=7, color='g')
# pyplot.xlabel("Number of shares")
# pyplot.ylabel("number of articles")
# pyplot.show()

# create a grid of histograms for the independent features

# fig, axes = pyplot.subplots(4, 5, figsize=(12, 10))
# for i in range(1,len(predictor_names)+1):
#     pyplot.subplot(4, 5, i)
#     pyplot.hist(df_predictors[predictor_names[i-1]], bins=20, color='g')
#     pyplot.xlabel(predictor_names[i-1])
#     pyplot.ylabel("amount")
# pyplot.subplots_adjust(wspace=0.9, hspace=0.9)
# # fig.delaxes(axes[3][1])
# fig.delaxes(axes[3][2])
# fig.delaxes(axes[3][3])
# fig.delaxes(axes[3][4])
# pyplot.show()