import pickle

p = 'c:/Users/NIDA RAHAMATH/OneDrive/Desktop/cancer/breast_cancer_model.pkl'
with open(p, 'rb') as f:
    obj = pickle.load(f)

print(type(obj))
print(repr(obj))
print('feature_names_in_', getattr(obj, 'feature_names_in_', None))
print('n_features_in_', getattr(obj, 'n_features_in_', None))
print('classes_', getattr(obj, 'classes_', None))
print('predict_proba_sample', obj.predict_proba_([[12.0, 20.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))
