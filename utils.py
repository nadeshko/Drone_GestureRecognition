from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class Utils():
    def __init__(self):
        """
        Initiates utilities required for project
        """

        # One-hot Encoder:
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse=False)

    def to_categorical(self, labels):
        """
        Function to convert data to one-hot format
        :param labels: video labels
        :return: labels in one-hot format, to distinguish labels digitally
        """

        # encode integers
        int_encoded = self.label_encoder.fit_transform(labels)
        int_encoded = int_encoded.reshape(len(int_encoded), 1)
        one_hot_encoded = self.one_hot_encoder.fit_transform(int_encoded)
        return one_hot_encoded