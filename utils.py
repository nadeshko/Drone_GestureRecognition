from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class Utils():
    def __init__(self):
        # One-hot Encoder:
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse=False)

    def to_categorical(self, labels):
        int_encoded = self.label_encoder.fit_transform(labels)
        int_encoded = int_encoded.reshape(len(int_encoded), 1)
        one_hot_encoded = self.one_hot_encoder.fit_transform(int_encoded)
        return one_hot_encoded