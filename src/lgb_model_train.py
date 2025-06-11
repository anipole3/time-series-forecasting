import lightgbm 
from sklearn.metrics import mean_absolute_error

class LGBModelTrainer:
    def __init__(self, params, target, features, num_boost_round):
        self.params = params
        self.target = target
        self.features = features
        self.num_boost_round = num_boost_round
        pass

    def _to_lgbm_dataset(self, df):
        lgb_df = lightgbm.Dataset(
            df[self.features], 
            label = df[self.target]
        )
        return lgb_df
    
    def train_model(self, df_train):
        lgb_train = self._to_lgbm_dataset(df_train)
        model_final = lightgbm.train(
                params =self.params,
                train_set = lgb_train,
                num_boost_round = self.num_boost_round,
        )
        return model_final
    
    def get_metrics(self, model, df_test):
        data = df_test.copy()
        data['pred'] = model.predict(df_test[self.features])
        return mean_absolute_error(data[self.target], data['pred'])