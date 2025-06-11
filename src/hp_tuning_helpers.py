import numpy as np 

def create_folds(data, folds):
    cv_folds = {}
    # sort years highest to lowest 
    years = sorted(data['year'].unique())[::-1]
    for f in range(folds):
        y = years[f]
        df_train = data[data['year']<y]
        df_val = data[data['year']==y]
        cv_folds[f] = [df_train, df_val]
    return cv_folds

def cross_val_metrics(model_class, folds, cv=3):
    metrics = []
    for fold in range(cv):
        train, test = folds[fold]
        model_fold = model_class.train_model(train)
        metric = model_class.get_metrics(model_fold, test)
        metrics.append(metric)
    return np.array([metrics]).mean()