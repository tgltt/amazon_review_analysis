import os
import numpy as np

from utils import load_model
from utils import get_label_desc

from consts import MODEL_SAVE_PATH
from consts import MODEL_PREPROCESSOR

from consts import MODEL_VOTE
from preprocessor import PreProcessor
from vocab import Voc

preprocessor_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_PREPROCESSOR)
preprocessor = load_model(preprocessor_model_path)

model_path = os.path.join(MODEL_SAVE_PATH, MODEL_VOTE)
vote = load_model(model_path=model_path)

sample1 = ["Good pad", "This is good good pad!", 3]
sample2 = ["Bad pad", "Too bad pad!", 3]
sample3 = ["First Tablet. Lots of possibilities.",
           "Great size, easy to carry for traveling. Need to spend more time Looking into apps for contact manegement, interactive calenders and most important, music storage and use.",
           5
          ]
sample4 = ["Acceptable for the price",
           "Love everything about the unit except that it gets hot when used for 10 to 15 minutes. Hope it doesn't burn or explode.",
           2
          ]
sample5 = ["It was average",
           "Purchased for my kids but it was hard to navigate and ended up with purchasing an iPad",
           2
          ]
sample6 = ["Works as advertised",
           "This tablet was purchased for my son. It does exactly what i want it to do which is play games and surf the web, namely YouTube. Great value.",
           5
          ]
sample7 = ["Its ok for the price",
           "I bought this on Black Friday, simply because it was dirt cheap. It works ok but kinda slow.",
           2
          ]

samples = np.array([sample1, sample2, sample3, sample4, sample5, sample6, sample7])
samples_label = np.array([0, 1, 0, 1, 1, 0, 1]).astype(np.int32)

content = preprocessor.transform_data(samples)
predict_result = vote.predict(content)

print(f"预测准确率：{(samples_label == predict_result).mean():.2}")

for content, pred in zip(samples, predict_result):
    print(content, "，", get_label_desc(pred))