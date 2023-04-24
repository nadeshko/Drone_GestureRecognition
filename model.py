#from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import *
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.layers import *

class models():
    def __init__(self, seed):
        # initialize sequential model
        self.model = Sequential()
        tf.random.set_seed(seed) # set seed constant (for same results)

    def create_LRCN(self, SEQUENCE, h, w, CLASS_LIST):
        """
        Function to create model architecture
        :param SEQUENCE: length of video
        :param h: video height
        :param w: video width
        :return: returns LRCN model
        """

        # time-distributed wrap the layers together and consider them in the temporal dimension
        self.model.add(TimeDistributed(Conv2D(32, (7, 7), padding='same', activation='relu'),
                                       input_shape= (SEQUENCE, h, w, 3)))
        self.model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
        self.model.add(TimeDistributed(MaxPooling2D(4)))

        self.model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        self.model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        self.model.add(TimeDistributed(MaxPooling2D(2)))

        self.model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
        self.model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
        self.model.add(TimeDistributed(MaxPooling2D(2)))

        self.model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
        self.model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
        self.model.add(TimeDistributed(MaxPooling2D(2)))


        self.model.add(TimeDistributed(Flatten()))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(128))

        self.model.add(Dense(len(CLASS_LIST), activation='softmax'))

        return self.model

    def compile_model(self, model, epochs, feature_train, feature_test, label_train, label_test):
        """
        Function to compile and train model
        :param model: model used for training
        :param epochs: no. of iterations
        :param feature_train: feature used for training
        :param feature_test: feature used for testing
        :param label_train: feature labels used for training
        :param label_test: feature labels used for testing
        :return: model history, testing accuracy and loss
        """

        # compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy'])

        # create instance of early stopping callback
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=15,
                                            mode='min', restore_best_weights=True)

        # stack list of tensors into one higher rank tensor
        feat_train = tf.stack(feature_train)
        lab_train = tf.stack(label_train)

        # training model
        model_history = model.fit(x=feat_train, y=lab_train,
                                      batch_size=8, epochs=epochs,
                                      shuffle=True, validation_split=0.2,
                                      callbacks=[early_stop_callback])

        # plot LRCN model
        plot_model(model, to_file='model results/plot/LRCN_model.png',
                   show_shapes=True, show_layer_names=True)

        # Show model summary
        model.summary()  # DEBUG

        # Evaluate Accuracy
        test_loss, test_acc = model.evaluate(feature_test, label_test, verbose=2)
        print(f'Test Accuracy: {test_acc}')

        return model_history, test_loss, test_acc