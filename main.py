from encoder.label_encoder import LabelEncoder
from encoder.one_hot_encoder import OneHotEncoder


y = ['Re_300', 'Re_300', 'Re_500', 'Re_2000', 'Re_10000', 'Re_50000', 'Re_500']

le = LabelEncoder(y)
ohe = OneHotEncoder(y)

res_le = le.transform
print("label encode: {}\nClass: {}".format(res_le, le.classes_))

res_ohe = ohe.fit_transform
print("One Hot encode: {}\nClass: {}".format(res_ohe, ohe.classes_))
