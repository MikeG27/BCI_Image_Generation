from sklearn.externals import joblib

def save_preprocessing_object(object, filename):
    joblib.dump(object, filename)
    print(f"Object was saved into path {filename}")

def load_preprocessing_object(filename):
    object = joblib.load(filename)
    return object