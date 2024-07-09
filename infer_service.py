import os
import numpy as np

from utils import load_model
from utils import get_label_desc

from consts import MODEL_SAVE_PATH
from consts import MODEL_PREPROCESSOR

from consts import MODEL_KNN
from consts import MODEL_DECISION_TREE
from consts import MODEL_NAIVE_BAYES
from consts import MODEL_RANDOM_FOREST
from consts import MODEL_LOGIC_REGRESSION
from consts import MODEL_SVM
from consts import MODEL_VOTE

from consts import MODEL_KNN_SHOW_NAME
from consts import MODEL_DECISION_TREE_SHOW_NAME
from consts import MODEL_NAIVE_BAYES_SHOW_NAME
from consts import MODEL_RANDOM_FOREST_SHOW_NAME
from consts import MODEL_LOGIC_REGRESSION_SHOW_NAME
from consts import MODEL_SVM_SHOW_NAME
from consts import MODEL_VOTE_SHOW_NAME

from preprocessor import PreProcessor
from vocab import Voc

preprocessor_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_PREPROCESSOR)
preprocessor = load_model(preprocessor_model_path)

dtc_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_DECISION_TREE)
knn_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_KNN)
gnb_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAIVE_BAYES)
rfc_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_RANDOM_FOREST)
lr_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_LOGIC_REGRESSION)
svc_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_SVM)

vote_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_VOTE)

dtc = None
gnb = None
rfc = None
lr = None
svc = None
vote = None

def get_classifier(cls_type):
    global dtc, gnb, rfc, lr, svc, vote
    if cls_type == MODEL_DECISION_TREE_SHOW_NAME:
        if dtc is None:
            dtc = load_model(dtc_model_path)
        return dtc
    elif cls_type == MODEL_NAIVE_BAYES_SHOW_NAME:
        if gnb is None:
            gnb = load_model(gnb_model_path)
        return gnb
    elif cls_type == MODEL_RANDOM_FOREST_SHOW_NAME:
        if rfc is None:
            rfc = load_model(rfc_model_path)
        return rfc
    elif cls_type == MODEL_SVM_SHOW_NAME:
        if svc is None:
            svc = load_model(svc_model_path)
        return svc
    elif cls_type == MODEL_LOGIC_REGRESSION_SHOW_NAME:
        if lr is None:
            lr = load_model(lr_model_path)
        return lr
    else:
        if vote is None:
            vote = load_model(model_path=vote_model_path)
        return vote

def predict_review_emotion(cls_type, samples):
    """
    :param samples ndarray，待预测用户评论:
    :return 用户情感列表:
    """
    classifier = get_classifier(cls_type)

    content = preprocessor.transform_data(samples)

    predict_result = classifier.predict(content)
    predict_result = [get_label_desc(pred) for pred in predict_result]

    return predict_result
