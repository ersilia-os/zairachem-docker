import os
import numpy as np
import collections
import shutil

import tensorflow as tf
from tensorflow import keras

from autokeras.tuners.hyperband import keras_tuner as kt
import autokeras as ak
from tensorflow.keras.models import load_model



COLUMNS_FILENAME = "columns.json"

EPOCHS = 100
VALIDATION_SPLIT = 0.2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class KerasTunerClassifier(object):
    def __init__(self, X, y):
        self.loss = "binary_crossentropy"
        self.metrics = [keras.metrics.AUC()]
        self.objective = "val_auc"
        self.X = X
        self.y = y
        self.input_shape = X.shape[1]
        self.output_shape = 1 if len(y.shape) == 1 else y.shape[1:]

    def _model_builder(self, hp):
        model = keras.Sequential()
        hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model.add(
            keras.layers.Dense(
                units=hp_units, activation="relu", input_shape=(self.input_shape,)
            )
        )
        model.add(keras.layers.Dense(self.output_shape, activation="sigmoid"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=self.loss,
            metrics=self.metrics,
        )
        return model

    def _search(self, save_path):
        tuner_directory = os.path.join(save_path, 'trials')
        print(tuner_directory)
        if os.path.exists(tuner_directory):
            shutil.rmtree(tuner_directory)  # Delete old trials directory

        print("Using save path:", save_path)
        self.tuner = kt.Hyperband(
            self._model_builder,
            objective=kt.Objective(self.objective, direction="max"),
            max_epochs=10,
            factor=3,
            directory=os.path.join(save_path),
            project_name="trials"
        )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        self.tuner.search(
            self.X,
            self.y,
            epochs=100,
            validation_split=VALIDATION_SPLIT,
            callbacks=[stop_early],
            verbose=True,
        )
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

    def _get_best_epoch(self):
        model = self.tuner.hypermodel.build(self.best_hps)
        history = model.fit(self.X, self.y, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
        val_per_epoch = history.history[self.objective]
        self.best_epoch = val_per_epoch.index(max(val_per_epoch)) + 1
        print("Best epoch: %d" % (self.best_epoch,))

    def _final_train(self):
        self.hypermodel = self.tuner.hypermodel.build(self.best_hps)
        self.hypermodel.fit(
            self.X, self.y, epochs=self.best_epoch, validation_split=VALIDATION_SPLIT
        )

    def fit(self, save_path):
        self._search(save_path)
        self._get_best_epoch()
        self._final_train()

    def save(self, save_path):
        print("Saving")
        self.hypermodel.save(
            os.path.join(save_path)
        )
    
    def clean(self, save_path):
        tuner_directory = os.path.join(save_path, 'trials')
        print(tuner_directory)
        if os.path.exists(tuner_directory):
            shutil.rmtree(tuner_directory) 
    
    def clear(self):
        keras.backend.clear_session()

    def export_model(self):
        return self.hypermodel

    def load(self, path):
        model = load_model(path, custom_objects=ak.CUSTOM_OBJECTS)
        return KerasTunerClassifierArtifact(model)


class KerasTunerClassifierArtifact(object):
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        probas = self.model.predict(X)
        return probas.flatten().tolist()
    
    def predict(self, X):
        probas = self.predict_proba(X)
        threshold = 0.5
        return [1 if proba > threshold else 0 for proba in probas]
    
    def run(self, X, y=None):
        results = collections.OrderedDict()
        results["main"] = {
            "idxs": None,
            "y": y,
            "y_hat": self.predict_proba(X),
            "b_hat": self.predict(X),
        }
        return results

class KerasTunerRegressor(object):
    def __init__(self):
        self.loss = "mean_squared_error"
        self.metrics = [keras.metrics.RootMeanSquaredError()]
        self.objective = "val_loss"

    def _model_builder(self, hp):
        model = keras.Sequential()
        hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model.add(
            keras.layers.Dense(
                units=hp_units, activation="relu", input_shape=(self.input_shape,)
            )
        )
        model.add(keras.layers.Dense(self.output_shape,))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=self.loss,
            metrics=self.metrics,
        )
        return model

    def _search(self, X, y):
        self.tuner = kt.Hyperband(
            self._model_builder,
            objective=self.objective,
            max_epochs=10,
            factor=3,
            directory=os.path.join(self.save_path, TUNER_PROJECT_NAME, self.task),
            project_name="trials",
        )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        self.tuner.search(
            X,
            y,
            epochs=100,
            validation_split=VALIDATION_SPLIT,
            callbacks=[stop_early],
            verbose=True,
        )
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

    def _get_best_epoch(self, X, y):
        model = self.tuner.hypermodel.build(self.best_hps)
        history = model.fit(X, y, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
        val_per_epoch = history.history[self.objective]
        self.best_epoch = val_per_epoch.index(min(val_per_epoch)) + 1
        print("Best epoch: %d" % (self.best_epoch,))

    def _final_train(self, X, y):
        self.hypermodel = self.tuner.hypermodel.build(self.best_hps)
        self.hypermodel.fit(
            X, y, epochs=self.best_epoch, validation_split=VALIDATION_SPLIT
        )

    def fit(self, X, y):
        self._search(X, y)
        self._get_best_epoch(X, y)
        self._final_train(X, y)

    def save(self, save_path):
        print("Saving")
        self.hypermodel.save(
            os.path.join(save_path)
        )

    def export_model(self):
        return self.hypermodel
    
    def load(self, path):
        model = load_model(path, custom_objects=ak.CUSTOM_OBJECTS)
        return KerasTunerRegressorArtifact(model)

class KerasTunerRegressorArtifact(object):
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def run(self, X, y=None):
        results = collections.OrderedDict()
        results["main"] = {"idxs": None, "y": y, "y_hat": self.predict(X)}
        return results
