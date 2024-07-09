import os
import numpy as np
import pandas as pd

import joblib
import nltk

from sklearn.model_selection import train_test_split

import torch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier

from preprocessor import PreProcessor

from consts import COLUMN_TITLE
from consts import COLUMN_TEXT
from consts import COLUMN_DO_RECOMMEND
from consts import COLUMN_RATING

from consts import LABEL_POS
from consts import LABEL_NEG

from consts import MODEL_SAVE_PATH
from consts import MODEL_PREPROCESSOR

from consts import MODEL_KNN
from consts import MODEL_DECISION_TREE
from consts import MODEL_NAIVE_BAYES
from consts import MODEL_RANDOM_FOREST
from consts import MODEL_LOGIC_REGRESSION
from consts import MODEL_SVM
from consts import MODEL_VOTE

from consts import DATASET_ROOT

from utils import load_model
from utils import save_model

# 指定亚马讯数据集存储路径
raw_data_file = os.path.join(DATASET_ROOT, "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

class AmazonReviewTrainer:
    """
    半监督训练器，将原始数据集拆分为正样本集、负样本集和缺失标签集，其中正样本1.5万条，负样本仅733条，缺失样本1.2万条，样本分布极不均衡，
    故通过半监督学习，将每轮迭代训练好的投票模型，用于预测缺失样本集的1.2万条数据，然后将1.2万条数据用于训练下一迭代的模型。
    """
    
    def __init__(self, raw_data_file):
        # 从原始数据集挑选的候选字段
        self.selected_columns = [COLUMN_TITLE, COLUMN_TEXT, COLUMN_RATING, COLUMN_DO_RECOMMEND]
        
        df = pd.read_csv(raw_data_file)[self.selected_columns]
        
        df_pos = df[df[COLUMN_DO_RECOMMEND] == True]
        df_neg = df[df[COLUMN_DO_RECOMMEND] == False]
        # 过滤缺失样本
        self.df_null = df[df[COLUMN_DO_RECOMMEND].isna()]

        print(f"正样本数：{len(df_pos)}, 负样本数：{len(df_neg)}, 原始缺失样本数：{len(self.df_null)}")
        
        # 合并正负样本为总样本集，从中挑选测试集
        data = []
        data.extend(df_pos.values.tolist())
        data.extend(df_neg.values.tolist())
        
        data_index = []
        data_index.extend(df_pos.index.tolist())
        data_index.extend(df_neg.index.tolist())

        data = np.array(data)
        data_index = np.array(data_index).astype(np.int32)

        X, y = data[:, :-1], data[:, -1]
        # 切分数据集
        _, X_test, _, y_test, _, X_index_test = train_test_split(X, y, data_index, test_size=.4, random_state=0, shuffle=True)
        
        # 保留专门的测试数据，不参与训练
        self.X_test = X_test
        self.y_test = y_test
        
        self.y_test[self.y_test == "True"] = LABEL_POS
        self.y_test[self.y_test == "False"] = LABEL_NEG
        self.y_test = self.y_test.astype(np.int32)
        
        # 将测试集数据，从原始数据集中剔除
        df = df.drop(X_index_test)
        
        print(f"测试集：{len(self.X_test)}")
        
        # 获取剔除测试本后的正负样本集合
        self.df_pos = df[df[COLUMN_DO_RECOMMEND] == True]
        self.df_neg = df[df[COLUMN_DO_RECOMMEND] == False]
        
        preprocessor_path = os.path.join(MODEL_SAVE_PATH, MODEL_PREPROCESSOR)
        if not os.path.exists(preprocessor_path):
            print("构建预处理器")
            self.preprocessor = PreProcessor()
            
            X_vocab = df[[COLUMN_TITLE, COLUMN_TEXT]].values.tolist()
            self.preprocessor.construct_vocab(X_vocab)
        else:
            print("加载预处理器")
            self.preprocessor = load_model(model_path=preprocessor_path)
            
        print(f"词表数：{len(self.preprocessor.voc_model.vocab)}")
        
    def _cluster_null_sample(self, data_pos, data_neg, data_null):
        """
        对无标签的样本进行聚类
        """
        pass
    
    def _prepare_fit(self):
        self._load_model()
        
        df_null_pos = self.df_null[self.df_null[COLUMN_DO_RECOMMEND] == True]
        df_null_neg = self.df_null[self.df_null[COLUMN_DO_RECOMMEND] == False]
        
        print(f"从原始缺失样本中，提取{len(df_null_pos)}个正样本，{len(df_null_neg)}个负样本,"               f"{len(self.df_null) - len(df_null_pos) - len(df_null_neg)}个仍缺失")
        
        data_null_pos = df_null_pos.values.tolist()
        data_null_neg = df_null_neg.values.tolist()
        
        data_neg = self.df_neg.values.tolist()
        data_pos = self.df_pos.values.tolist()
                   
        data_neg.extend(data_null_neg)
        data_pos.extend(data_null_pos)
                 
        data_neg = np.array(data_neg)
        data_pos = np.array(data_pos)
        
        # 随机采样正样本，使正负样本数量均衡
        data_neg_count = len(data_neg)
        data_pos_count = len(data_pos)
        data_sample_count = data_neg_count if data_neg_count <= data_pos_count else data_pos_count
        
        data_neg_sample_idx = np.random.choice(range(data_neg_count), size=(data_sample_count,))
        data_pos_sample_idx = np.random.choice(range(data_pos_count), size=(data_sample_count,))
                 
        data_neg = data_neg[data_neg_sample_idx]
        data_pos = data_pos[data_pos_sample_idx]
        
        print(f"调整采样后，正样本：{len(data_neg)}，负样本：{len(data_pos)}")
        
        data = []
        data.extend(data_neg)
        data.extend(data_pos)
        
        data = np.array(data)
        
        X_train, y_train = data[:, :-1], data[:, -1]
        
        y_train[y_train == "True"] = LABEL_POS
        y_train[y_train == "False"] = LABEL_NEG

        y_train = y_train.astype(np.int32)
        
        X_null = np.array(self.df_null.values.tolist())
        
#         self._cluster_null_sample()
    
        X_train = self.preprocessor.filter_stopwords_(X_train)
        self.X_test = self.preprocessor.filter_stopwords_(self.X_test)
    
        X_train_vec = self.preprocessor.vectorize(X_train, y_train)
        self.X_test_vec = self.preprocessor.vectorize(self.X_test)
        
        pca = self.preprocessor.init_PCA(X_train_vec)
        X_train_vec = pca.transform(X_train_vec)
        self.X_test_vec = pca.transform(self.X_test_vec)

        if "X_null_vec" not in locals():
            X_null = self.preprocessor.filter_stopwords_(X_null)
            self.X_null_vec = self.preprocessor.vectorize(X_null[:, :-1])
            self.X_null_vec = pca.transform(self.X_null_vec)
            
        save_model(self.preprocessor, model_name=MODEL_PREPROCESSOR, model_save_path=MODEL_SAVE_PATH)
        
        return X_train_vec, y_train
    
    def fit(self, epochs=1):
        for i in range(epochs):
            print(f"Epoch {i}: --------------------------------")
            X_train, y_train = self._prepare_fit()
            
            print(self.y_test)
            # TODO: 增加KMeans
            dtc = self._fit_dtc(X_train, self.X_test_vec, y_train, self.y_test)
#             knn = self._fit_knn(X_train, self.X_test_vec, y_train, self.y_test)
            gnb = self._fit_gnb(X_train, self.X_test_vec, y_train, self.y_test)
            rfc = self._fit_rfc(X_train, self.X_test_vec, y_train, self.y_test)
            lr = self._fit_lr(X_train, self.X_test_vec, y_train, self.y_test)
            svc = self._fit_svc(X_train, self.X_test_vec, y_train, self.y_test)

            estimators = [("dtc", dtc), ("gnb", gnb), ("rfc", rfc), ("logic", lr), ("svc", svc)] # ("knn", knn), 
            self.vote = self._fit_vote(estimators, X_train, self.X_test_vec, y_train, self.y_test)
            
            y_null = self.predict(self.X_null_vec)
            
            del self.df_null[COLUMN_DO_RECOMMEND]
            self.df_null[COLUMN_DO_RECOMMEND] = y_null.astype(bool)
            
#             dr_new = pd.DataFrame({COLUMN_DO_RECOMMEND:y_null.astype(bool)})
#             self.df_null.update(dr_new)

        print("训练完成")
        
    def _fit_dtc(self, X_train, X_test, y_train, y_test):
        self.dtc.fit(X_train, y_train)

        dtc_train_score = self.dtc.score(X_train, y_train)
        dtc_test_score = self.dtc.score(X_test, y_test)

        save_model(self.dtc, MODEL_DECISION_TREE, MODEL_SAVE_PATH)
        
        print(f"DecisionTreeClassifier：训练集分数：{dtc_train_score:.4}, 测试集分数：{dtc_test_score:.4}")
        
        return self.dtc
        
    def _fit_knn(self, X_train, X_test, y_train, y_test):
        self.knn.fit(X_train, y_train)

        knn_train_score = self.knn.score(X_train, y_train)
        knn_test_score = self.knn.score(X_test, y_test)

        save_model(self.knn, MODEL_KNN, MODEL_SAVE_PATH)

        print(f"KNN：训练集分数：{knn_train_score:.4}, 测试集分数：{knn_test_score:.4}")
        
        return self.knn
        
    def _fit_gnb(self, X_train, X_test, y_train, y_test):
        self.gnb.fit(X_train, y_train)

        gnb_train_score = self.gnb.score(X_train, y_train)
        gnb_test_score = self.gnb.score(X_test, y_test)

        print(f"GaussianNB：训练集分数：{gnb_train_score:.4}, 测试集分数：{gnb_test_score:.4}")

        save_model(self.gnb, MODEL_NAIVE_BAYES, MODEL_SAVE_PATH)
        
        return self.gnb
        
    def _fit_rfc(self, X_train, X_test, y_train, y_test):
        self.rfc.fit(X_train, y_train)

        rfc_train_score = self.rfc.score(X_train, y_train)
        rfc_test_score = self.rfc.score(X_test, y_test)

        save_model(self.rfc, MODEL_RANDOM_FOREST, MODEL_SAVE_PATH)

        print(f"RandomForestClassifier：训练集分数：{rfc_train_score:.4}, 测试集分数：{rfc_test_score:.4}")
        
        return self.rfc
    
    def _fit_lr(self, X_train, X_test, y_train, y_test):
        self.lr.fit(X_train, y_train)

        lr_train_score = self.lr.score(X_train, y_train)
        lr_test_score = self.lr.score(X_test, y_test)

        save_model(self.lr, MODEL_LOGIC_REGRESSION, MODEL_SAVE_PATH)
        
        print(f"LogisticRegression：训练集分数：{lr_train_score:.4}, 测试集分数：{lr_test_score:.4}")
        
        return self.lr
        
    def _fit_svc(self, X_train, X_test, y_train, y_test):
        self.svc.probability = True
        self.svc.fit(X_train, y_train)

        svc_train_score = self.svc.score(X_train, y_train)
        svc_test_score = self.svc.score(X_test, y_test)

        save_model(self.svc, MODEL_SVM, MODEL_SAVE_PATH)

        print(f"SVC：训练集分数：{svc_train_score:.4}, 测试集分数：{svc_test_score:.4}")
        
        return self.svc
    
    def _fit_vote(self, estimators, X_train, X_test, y_train, y_test):
#         vote_hard = VotingClassifier(estimators=estimators, voting="hard")
        vote = VotingClassifier(estimators=estimators, 
                                voting="soft", 
                                weights=[1, 0.8, 2, 1.5, 1]) # 1,
        vote.fit(X_train, y_train)
        
        vote_train_score = vote.score(X_train, y_train)
        vote_test_score = vote.score(X_test, y_test)

        save_model(vote, MODEL_VOTE, MODEL_SAVE_PATH)

        print(f"VotingClassifier：训练集分数：{vote_train_score:.4}, 测试集分数：{vote_test_score:.4}")
        
        return vote
    
    def _load_model(self):
        dtc_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_DECISION_TREE)
        knn_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_KNN)
        gnb_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAIVE_BAYES)
        rfc_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_RANDOM_FOREST)
        lr_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_LOGIC_REGRESSION)
        svc_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_SVM)

        self.dtc = load_model(dtc_model_path) if os.path.exists(dtc_model_path) else DecisionTreeClassifier()
        self.knn = load_model(knn_model_path) if os.path.exists(knn_model_path) else KNeighborsClassifier()
        self.gnb = load_model(gnb_model_path) if os.path.exists(gnb_model_path) else GaussianNB()
        self.rfc = load_model(rfc_model_path) if os.path.exists(rfc_model_path) else RandomForestClassifier(bootstrap=False, max_features=17, n_estimators=100)
        self.lr = load_model(lr_model_path) if os.path.exists(lr_model_path) else LogisticRegression(max_iter=1000)
        self.svc = load_model(svc_model_path) if os.path.exists(svc_model_path) else SVC()
    
    def predict(self, X):
        return self.vote.predict(X=X)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean()

art = AmazonReviewTrainer(raw_data_file=raw_data_file)
art.fit(epochs=5)