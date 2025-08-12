import lazyqsar, os, sys, time
import pandas as pd
from sklearn.metrics import roc_curve, auc


model_type = "random_forest"

train = pd.read_csv("../../files/data/bioavailability_ma_train.csv")
smiles_train = train["Drug"].tolist()
y_train = train["Y"].tolist()
test = pd.read_csv("../../files/data/bioavailability_ma_test.csv")
smiles_test = test["Drug"].tolist()
y_test = test["Y"].tolist()

print(len(smiles_train))

# Precalculated descriptors for train and test sets
X_train = "../../model_dir/descriptors/eos5axz/raw.h5"
X_test = "../../model_dir/descriptors/eos5axz/raw.h5"


def fit():
  st = time.perf_counter()
  model = lazyqsar.LazyBinaryClassifier(model_type=model_type, mode="quick")
  model.fit(h5_file=X_train, y=y_train)
  model_path = os.path.abspath("test_model")
  model.save_model(model_path)
  et = time.perf_counter()
  print(f"Training takes: {et - st:.4} seconds")


def predict():
  st = time.perf_counter()
  model_path = os.path.abspath("test_model")
  model = lazyqsar.LazyBinaryClassifier.load_model(model_path)
  y_hat = model.predict_proba(h5_file=X_test)[:, 1]
  fpr, tpr, _ = roc_curve(y_test, y_hat)
  print("AUROC", auc(fpr, tpr))
  et = time.perf_counter()
  print(f"Predicting takes: {et - st:.4} seconds")


if __name__ == "__main__":
  fit()
  predict()
