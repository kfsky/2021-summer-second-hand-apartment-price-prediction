from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor

from .base import BaseModel


class MyExtraTreeClassifier(BaseModel):
    def build_model(self):
        model = ExtraTreesClassifier(**self.params)
        return model

    def predict(self, x):
        preds = self.model.predict_proba(x)[:, 1]
        return preds


class MyExtraTreeRegressor(BaseModel):
    def build_model(self):
        model = ExtraTreesRegressor(**self.params)
        return model

    def predict(self, x):
        preds = self.model.predict(x)
        return preds
