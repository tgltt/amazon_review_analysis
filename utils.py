import os
import torch
import joblib

from consts import LABEL_POS
from consts import LABEL_NEG

from consts import FRAMEWORK_SKLEARN
from consts import FRAMEWORK_PYTORCH

def get_label_desc(label):
    if label == LABEL_POS:
        return "正面评价"
    elif label == LABEL_NEG:
        return "负面评价"
    else:
        return "Unknown"

def save_model(model, model_name, model_save_path, dev_fw_type=FRAMEWORK_SKLEARN, model_input_shape=None):
    if model is None or model_name is None:
        raise Exception("save model error, because model or model name is none")

    if not os.path.exists(model_save_path):
        print(f"创建 {model_save_path}")
        os.makedirs(model_save_path)

    model_path = os.path.join(model_save_path, model_name)
    print(f"Saving ", model_path)

    if dev_fw_type == FRAMEWORK_PYTORCH:
        if model_input_shape is None:
            torch.save(model, model_path)
        else:
            torch_script_model = torch.jit.trace(model, torch.randn(model_input_shape))
            torch.jit.save(torch_script_model, model_path)
    else:
        joblib.dump(model, model_path)

def load_model(model_path, dev_fw_type=FRAMEWORK_SKLEARN):
    if dev_fw_type == FRAMEWORK_PYTORCH:
        model = torch.load(model_path)
    else:
        model = joblib.load(filename=model_path)
    return model