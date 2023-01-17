import pickle

import torch
# This is a sample Python script.
import transformers
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

model_transformer = pickle.load(open('models/model2_pkl','rb'))
f="gfgf"
print(model_transformer.predict(f).str[0].str['label'])
