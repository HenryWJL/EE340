import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from collections import Counter
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_data():
    # Read data from the document
    data = pd.read_csv('./python/EE340/project2/kddcup.data_10_percent_corrected.csv', header=None)
    # Transform qualitative attributes into quantitative attributes
    data[1]=data[1].map({'tcp':0, 'udp':1, 'icmp':2})
    
    data[2]=data[2].map({'aol':0, 'auth':1, 'bgp':2, 'courier':3, 'csnet_ns':4,'ctf':5, 'daytime':6, 'discard':7, 'domain':8, 
                         'domain_u':9,'echo':10, 'eco_i':11, 'ecr_i':12, 'efs':13, 'exec':14,'finger':15, 'ftp':16, 'ftp_data':17, 
                         'gopher':18, 'harvest':19,'hostnames':20, 'http':21, 'http_2784':22, 'http_443':23, 'http_8001':24,'imap4':25, 
                         'IRC':26, 'iso_tsap':27, 'klogin':28, 'kshell':29,'ldap':30, 'link':31, 'login':32, 'mtp':33, 'name':34,
                         'netbios_dgm':35, 'netbios_ns':36, 'netbios_ssn':37, 'netstat':38, 'nnsp':39,'nntp':40, 'ntp_u':41, 'other':42, 
                         'pm_dump':43, 'pop_2':44,'pop_3':45, 'printer':46, 'private':47, 'red_i':48, 'remote_job':49,'rje':50, 'shell':51, 
                         'smtp':52, 'sql_net':53, 'ssh':54,'sunrpc':55, 'supdup':56, 'systat':57, 'telnet':58, 'tftp_u':59,'tim_i':60, 
                         'time':61, 'urh_i':62, 'urp_i':63, 'uucp':64,'uucp_path':65, 'vmnet':66, 'whois':67, 'X11':68, 'Z39_50':69})
    
    data[3]=data[3].map({'OTH':0, 'REJ':0, 'RSTO':0,'RSTOS0':0, 'RSTR':0, 'S0':0,'S1':0, 'S2':0, 'S3':0,'SF':1, 'SH':0})
    
    data[41]=data[41].map({'normal.':0, 'ipsweep.':2, 'nmap.':2, 'portsweep.':2, 'satan.':2, 'back.':1, 'land.':1, 'neptune.':1, 'pod.':1, 
                           'smurf.':1, 'teardrop.':1, 'buffer_overflow.':4, 'loadmodule.':4, 'perl.':4, 'rootkit.':4, 'ftp_write.':3,
                           'guess_passwd.':3, 'imap.':3, 'multihop.':3, 'phf.':3, 'spy.':3, 'warezclient.':3,'warezmaster.':3})
    # Assign the values to the input X and the response y
    y = data[41]
    del data[41]
    X = data
    # Further processing
    X = np.array(X)
    X = StandardScaler().fit_transform(X)  # Standardize the observations
    y = np.array(y)
    y = np.nan_to_num(y)
    print(X.shape)
    print(y.shape)
    return X, y

# 检测异常值
def detect_outliers(df, n, features):
    outlier_indices = []
    # 遍历每个特征
    for col in features:        # 将特征转换为数值类型
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # 计算第一四分位数（Q1）和第三四分位数（Q3）
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        # 计算四分位距（IQR）
        IQR = Q3 - Q1
        # 计算异常值的阈值
        outlier_threshold = n * IQR
        # 确定异常值的索引
        outlier_list_col = df[(df[col] < Q1 - outlier_threshold) | (df[col] > Q3 + outlier_threshold)].index
        # 将异常值的索引添加到列表中
        outlier_indices.extend(outlier_list_col)
    # 选择包含多个异常值的行
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers

# 数据清洗
def data_cleaning(X):
    X.dropna(inplace=True)
    X.drop_duplicates(inplace=True)
    outliers_to_drop = detect_outliers(X, 4, X.columns)
    X.drop(outliers_to_drop, inplace=True)

# 特征选择
def feature_selection(X):
    # 方法1：截距项
    name = X.columns
    X = np.matrix(X)
    print(X)
    VIF_list = [variance_inflation_factor(X,i) for i in range(X.shape[1])]
    VIF = pd.DataFrame({'feature':name,"VIF":VIF_list})
    print(VIF)
    # 方差膨胀系数VIF大于10时表示特征之间有较高的共线性

    # 方法2：person相关系数
    corr_df = X.corr()
    # 热力图
    sns.heatmap(corr_df)
    plt.show()
    # 剔除相关性系数高于threshold的corr_drop
    threshold = 0.9
    upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(np.bool_))
    corr_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
    print(corr_drop)
    

if __name__ == '__main__':
    # Loading data
    X, y = load_data()
    # Feature selection
    feature_selection(X)
    # Split the dataset into training dataset and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ''' Training different models ''' 
    # 1. KNN
    KNN = KNeighborsClassifier(n_neighbors=2, algorithm='kd_tree')
    KNN.fit(X_train, y_train)
    # Evaluating the model
    knn_scores = cross_validate(KNN, X_train, y_train, scoring=['precision_macro', 'recall_macro', 'f1_macro', 'accuracy'])
    print('knn_accuracy: ', knn_scores['test_accuracy'].mean())
    print('knn_macro_precision: ', knn_scores['test_precision_macro'].mean())
    print('knn_macro_recall: ', knn_scores['test_recall_macro'].mean())
    print('knn_macro_f1_score: ', knn_scores['test_f1_macro'].mean())
    print('\n')
    # Obtaining the confusion matrix
    knn_conf_matrix = confusion_matrix(y_test, KNN.predict(X_test))
    knn_conf_matrix = pd.DataFrame(knn_conf_matrix, index=['Normal', 'DOS', 'Probing', 'R2L', 'U2R'], columns=['Normal', 'DOS', 'Probing', 'R2L', 'U2R']) 
    sns.heatmap(knn_conf_matrix, annot=True)
    plt.ylabel('True Labels', fontsize=14)
    plt.xlabel('Predictions', fontsize=14)
    plt.title('KNN Confusion Matrix')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    # 2. Decision Tree
    Decision_Tree = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5000)
    Decision_Tree.fit(X_train, y_train)
    # Evaluating the model
    decision_tree_scores = cross_validate(Decision_Tree, X_train, y_train, scoring=['precision_macro', 'recall_macro', 'f1_macro', 'accuracy'])
    print('decision_tree_accuracy: ', decision_tree_scores['test_accuracy'].mean())
    print('decision_tree_macro_precision: ', decision_tree_scores['test_precision_macro'].mean())
    print('decision_tree_macro_recall: ', decision_tree_scores['test_recall_macro'].mean())
    print('decision_tree_macro_f1_score: ', decision_tree_scores['test_f1_macro'].mean())
    print('\n')
    # Obtaining the confusion matrix
    decision_tree_conf_matrix = confusion_matrix(y_test, Decision_Tree.predict(X_test))
    decision_tree_conf_matrix = pd.DataFrame(decision_tree_conf_matrix, index=['Normal', 'DOS', 'Probing', 'R2L', 'U2R'], columns=['Normal', 'DOS', 'Probing', 'R2L', 'U2R'])  
    sns.heatmap(decision_tree_conf_matrix, annot=True)
    plt.ylabel('True Labels', fontsize=14)
    plt.xlabel('Predictions', fontsize=14)
    plt.title('Decision Tree Confusion Matrix')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()    

    # 3. SVM
    SVM = svm.LinearSVC(dual=False, max_iter=2000, random_state=42)
    SVM.fit(X_train, y_train)
    # Evaluating the model
    svm_scores = cross_validate(SVM, X_train, y_train, scoring=['precision_macro', 'recall_macro', 'f1_macro', 'accuracy'])
    print('svm_accuracy: ', svm_scores['test_accuracy'].mean())
    print('svm_macro_precision: ', svm_scores['test_precision_macro'].mean())
    print('svm_macro_recall: ', svm_scores['test_recall_macro'].mean())
    print('svm_macro_f1_score: ', svm_scores['test_f1_macro'].mean())
    print('\n')
    # Obtaining the confusion matrix
    svm_conf_matrix = confusion_matrix(y_test, SVM.predict(X_test))
    svm_conf_matrix = pd.DataFrame(svm_conf_matrix, index=['Normal', 'DOS', 'Probing', 'R2L', 'U2R'], columns=['Normal', 'DOS', 'Probing', 'R2L', 'U2R'])  
    sns.heatmap(svm_conf_matrix, annot=True)
    plt.ylabel('True Labels', fontsize=14)
    plt.xlabel('Predictions', fontsize=14)
    plt.title('SVM Confusion Matrix')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()       

    # 4. Random Forest
    Random_Forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=3000)
    Random_Forest.fit(X_train, y_train)
    # Evaluating the model
    random_forest_scores = cross_validate(Random_Forest, X_train, y_train, scoring=['precision_macro', 'recall_macro', 'f1_macro', 'accuracy'])
    print('random_forest_accuracy: ', random_forest_scores['test_accuracy'].mean())
    print('random_forest_macro_precision: ', random_forest_scores['test_precision_macro'].mean())
    print('random_forest_macro_recall: ', random_forest_scores['test_recall_macro'].mean())
    print('random_forest_macro_f1_score: ', random_forest_scores['test_f1_macro'].mean())
    # Obtaining the confusion matrix
    random_forest_conf_matrix = confusion_matrix(y_test, Random_Forest.predict(X_test))
    random_forest_conf_matrix = pd.DataFrame(random_forest_conf_matrix, index=['Normal', 'DOS', 'Probing', 'R2L', 'U2R'], columns=['Normal', 'DOS', 'Probing', 'R2L', 'U2R'])
    sns.heatmap(random_forest_conf_matrix, annot=True)
    plt.ylabel('True Labels', fontsize=14)
    plt.xlabel('Predictions', fontsize=14)
    plt.title('Random Forest Confusion Matrix')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
