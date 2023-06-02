import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_data():
    data = pd.read_csv('./Python_codes/EE340/project2/kddcup.data_10_percent_corrected.csv', header=None)
    data[1]=data[1].map({'tcp':0, 'udp':1, 'icmp':2})
    data[2]=data[2].map({'aol':0, 'auth':1, 'bgp':2, 'courier':3, 'csnet_ns':4,'ctf':5, 'daytime':6, 'discard':7, 'domain':8, 'domain_u':9,'echo':10, 'eco_i':11, 'ecr_i':12, 'efs':13, 'exec':14,'finger':15, 'ftp':16, 'ftp_data':17, 'gopher':18, 'harvest':19,'hostnames':20, 'http':21, 'http_2784':22, 'http_443':23, 'http_8001':24,'imap4':25, 'IRC':26, 'iso_tsap':27, 'klogin':28, 'kshell':29,'ldap':30, 'link':31, 'login':32, 'mtp':33, 'name':34,'netbios_dgm':35, 'netbios_ns':36, 'netbios_ssn':37, 'netstat':38, 'nnsp':39,'nntp':40, 'ntp_u':41, 'other':42, 'pm_dump':43, 'pop_2':44,'pop_3':45, 'printer':46, 'private':47, 'red_i':48, 'remote_job':49,'rje':50, 'shell':51, 'smtp':52, 'sql_net':53, 'ssh':54,'sunrpc':55, 'supdup':56, 'systat':57, 'telnet':58, 'tftp_u':59,'tim_i':60, 'time':61, 'urh_i':62, 'urp_i':63, 'uucp':64,'uucp_path':65, 'vmnet':66, 'whois':67, 'X11':68, 'Z39_50':69})
    data[3]=data[3].map({'OTH':0, 'REJ':0, 'RSTO':0,'RSTOS0':0, 'RSTR':0, 'S0':0,'S1':0, 'S2':0, 'S3':0,'SF':1, 'SH':0})
    data[41]=data[41].map({'normal.':0, 'ipsweep.':1, 'mscan.':2, 'nmap.':3, 'portsweep.':4, 'saint.':5, 'satan.':6, 'apache2.':7,'back.':8, 'land.':9, 'mailbomb.':10, 'neptune.':11, 'pod.':12,'processtable.':13, 'smurf.':14, 'teardrop.':15, 'udpstorm.':16, 'buffer_overflow.':17, 'httptunnel.':18, 'loadmodule.':19, 'perl.':20, 'ps.':21,'rootkit.':22, 'sqlattack.':23, 'xterm.':24, 'ftp_write.':25,'guess_passwd.':26, 'imap.':27, 'multihop.':28, 'named.':29, 'phf.':30,'sendmail.':31, 'snmpgetattack.':32, 'snmpguess.':33, 'spy.':34, 'warezclient.':35,'warezmaster.':36, 'worm.':37, 'xlock.':38, 'xsnoop.':39})
    y = data[41]
    del data[41]
    X = data
    X = np.array(X)
    X = StandardScaler().fit_transform(X)
    y = np.array(y)
    y = np.nan_to_num(y)
    # y = y.reshape(y.shape[0], -1)
    # y = OneHotEncoder(sparse=False).fit_transform(y)
    
    return X, y


X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN
KNN = KNeighborsClassifier(n_neighbors=2, algorithm='kd_tree')
KNN.fit(X_train, y_train)
print('knn_accuracy: ', accuracy_score(y_test, KNN.predict(X_test)))
print('knn_macro_precision: ', precision_score(y_test, KNN.predict(X_test), average='macro'))
print('knn_macro_recall: ', recall_score(y_test, KNN.predict(X_test), average='macro'))
print('knn_macro_f1_score: ', f1_score(y_test, KNN.predict(X_test), average='macro'))
print('\n')
knn_conf_matrix = confusion_matrix(y_test, KNN.predict(X_test))
knn_conf_matrix = pd.DataFrame(knn_conf_matrix, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'], 
                               columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'])  # 数据有40个类别
sns.heatmap(knn_conf_matrix, annot=True)
plt.ylabel('True Labels', fontsize=14)
plt.xlabel('Predictions', fontsize=14)
plt.title('KNN Confusion Matrix')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Decision Tree
Decision_Tree = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5000)
Decision_Tree.fit(X_train, y_train)
print('decision_tree_accuracy: ', accuracy_score(y_test, Decision_Tree.predict(X_test)))
print('decision_tree_macro_precision: ', precision_score(y_test, Decision_Tree.predict(X_test), average='macro'))
print('decision_tree_macro_recall: ', recall_score(y_test, Decision_Tree.predict(X_test), average='macro'))
print('decision_tree_macro_f1_score: ', f1_score(y_test, Decision_Tree.predict(X_test), average='macro'))
print('\n')
decision_tree_conf_matrix = confusion_matrix(y_test, Decision_Tree.predict(X_test))
decision_tree_conf_matrix = pd.DataFrame(decision_tree_conf_matrix, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'], 
                               columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'])  # 数据有40个类别
sns.heatmap(decision_tree_conf_matrix, annot=True)
plt.ylabel('True Labels', fontsize=14)
plt.xlabel('Predictions', fontsize=14)
plt.title('Decision Tree Confusion Matrix')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()    

# SVM
SVM = svm.LinearSVC(dual=False, max_iter=2000, random_state=42)
SVM.fit(X_train, y_train)
print('svm_accuracy: ', accuracy_score(y_test, SVM.predict(X_test)))
print('svm_macro_precision: ', precision_score(y_test, SVM.predict(X_test), average='macro'))
print('svm_macro_recall: ', recall_score(y_test, SVM.predict(X_test), average='macro'))
print('svm_macro_f1_score: ', f1_score(y_test, SVM.predict(X_test), average='macro'))
print('\n')
svm_conf_matrix = confusion_matrix(y_test, SVM.predict(X_test))
svm_conf_matrix = pd.DataFrame(svm_conf_matrix, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'], 
                               columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'])  # 数据有22个类别
sns.heatmap(svm_conf_matrix, annot=True)
plt.ylabel('True Labels', fontsize=14)
plt.xlabel('Predictions', fontsize=14)
plt.title('SVM Confusion Matrix')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()       

# Random Forest
Random_Forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=3000)
Random_Forest.fit(X_train, y_train)
print('random_forest_accuracy: ', accuracy_score(y_test, Random_Forest.predict(X_test)))
print('random_forest_macro_precision: ', precision_score(y_test, Random_Forest.predict(X_test), average='macro'))
print('random_forest_macro_recall: ', recall_score(y_test, Random_Forest.predict(X_test), average='macro'))
print('random_forest_macro_f1_score: ', f1_score(y_test, Random_Forest.predict(X_test), average='macro'))
random_forest_conf_matrix = confusion_matrix(y_test, Random_Forest.predict(X_test))
random_forest_conf_matrix = pd.DataFrame(random_forest_conf_matrix, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'], 
                               columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])  # 数据有20个类别
sns.heatmap(random_forest_conf_matrix, annot=True)
plt.ylabel('True Labels', fontsize=14)
plt.xlabel('Predictions', fontsize=14)
plt.title('Random Forest Confusion Matrix')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()