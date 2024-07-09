import numpy as np
import streamlit as st

from preprocessor import PreProcessor
from vocab import Voc

from consts import MODEL_KNN_SHOW_NAME
from consts import MODEL_DECISION_TREE_SHOW_NAME
from consts import MODEL_NAIVE_BAYES_SHOW_NAME
from consts import MODEL_RANDOM_FOREST_SHOW_NAME
from consts import MODEL_LOGIC_REGRESSION_SHOW_NAME
from consts import MODEL_SVM_SHOW_NAME
from consts import MODEL_VOTE_SHOW_NAME

from infer_service import predict_review_emotion

st.sidebar.title("用户调参控制面板")
select_algorithm = st.sidebar.radio("请选择分类算法",
                                     [MODEL_DECISION_TREE_SHOW_NAME, MODEL_RANDOM_FOREST_SHOW_NAME,
                                      MODEL_NAIVE_BAYES_SHOW_NAME, MODEL_SVM_SHOW_NAME,
                                      MODEL_LOGIC_REGRESSION_SHOW_NAME, MODEL_VOTE_SHOW_NAME],
                                     index=5)

remark_title = "Its ok for the price"
remark_text = "I bought this on Black Friday, simply because it was dirt cheap. It works ok but kinda slow."
remark_rate = 2

predict_result_text = ""

title = st.text_input('评论标题', value=remark_title)
text = st.text_area('评论内容', value=remark_text)
rate = st.number_input('评论星级', 0, 5, value=remark_rate)

if st.button("预测"):
    samples = np.array([[title, text, rate]])
    predict_results = predict_review_emotion(select_algorithm, samples)

    for predict_result in predict_results:
        predict_result_text += predict_result + ", "
    predict_result_text = predict_result_text[:predict_result_text.rindex(", ")]

if len(predict_result_text) > 0:
    st.write("预测结果:&nbsp;&nbsp;&nbsp;&nbsp;【", predict_result_text + "】")