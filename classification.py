import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


class BaseClassifier:
    def __init__(self, model, name='Base'):
        self.model = model
        self.name = name
        self.metrics = {}

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        return self.metrics

    def report(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print(f'--- {self.name} ---')
        print(classification_report(y_test, y_pred))
        print(f'Confusion Matrix:\n{self.metrics["confusion_matrix"]}')


class DecisionTreeModel(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(DecisionTreeClassifier(**kwargs), name='Decision Tree')


class RandomForestModel(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(RandomForestClassifier(**kwargs), name='Random Forest')


class GradientBoostingModel(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(GradientBoostingClassifier(**kwargs), name='Gradient Boosting')


class ModelComparer:
    def __init__(self, models: list):
        self.models = models
        self.results = {}

    def fit_all(self, X_train, y_train):
        for model in self.models:
            model.fit(X_train, y_train)

    def evaluate_all(self, X_test, y_test):
        for model in self.models:
            metrics = model.evaluate(X_test, y_test)
            self.results[model.name] = metrics

    def summary(self):
        print('=== Porownanie modeli ===')
        summary_df = pd.DataFrame(self.results).T[['accuracy', 'f1_score']]
        print(summary_df)