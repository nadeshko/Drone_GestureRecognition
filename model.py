from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from keras.layers import *
import tensorflow as tf

class models():
    def __init__(self, seed):
        # initialize sequential model
        self.model = Sequential()
        tf.random.set_seed(seed)

    def create_LRCN(self, SEQUENCE=20, h=int(), w=int(), CLASS_LIST=list()):
        """
        Function to add on the model architecture
        :param SEQUENCE: length of video
        :param h: video height
        :param w: video width
        :return: returns LRCN model
        """
        self.model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'),
                                       input_shape= (SEQUENCE, h, w, 3)))
        self.model.add(TimeDistributed(MaxPooling2D(4)))
        self.model.add(TimeDistributed(Dropout(0.25)))

        self.model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
        self.model.add(TimeDistributed(MaxPooling2D(4)))
        self.model.add(TimeDistributed(Dropout(0.2)))

        self.model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        self.model.add(TimeDistributed(MaxPooling2D(2)))
        self.model.add(TimeDistributed(Dropout(0.25)))

        self.model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        self.model.add(TimeDistributed(MaxPooling2D(2)))
        self.model.add(TimeDistributed(Dropout(0.1))) #DEBUG?

        self.model.add(TimeDistributed(Flatten()))

        self.model.add(LSTM(32))

        self.model.add(Dense(len(CLASS_LIST), activation='softmax'))

        return self.model

    def compile_model(self, model, epochs, feature_train, feature_test, label_train, label_test):
        model.compile(
            loss='categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy'])

        early_stop_callback = EarlyStopping(monitor='val_loss', patience=15,
                                            mode='min', restore_best_weights=True)

        model_history = model.fit(x=feature_train, y=label_train,
                                      batch_size=200, epochs=epochs,
                                      shuffle=True, validation_split=0.2,
                                      callbacks=[early_stop_callback])

        # Show model summary
        model.summary()  # DEBUG

        # Evaluate Accuracy
        test_loss, test_acc = model.evaluate(feature_test, label_test, verbose=2)
        print(f'\Test Accuracy: {test_acc}')

        return model_history, test_loss, test_acc