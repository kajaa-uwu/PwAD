import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from classification import DecisionTreeModel, RandomForestModel, GradientBoostingModel, ModelComparer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_pickle('clean_data.pkl')
"""
print(f'Rozklad fraudow:\n{data['fraud_reported'].value_counts()}')


fig, ax = plt.subplots()
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
sns.histplot(data=data, x='age', hue='fraud_reported', kde=True, multiple='stack', ax=ax)
plt.xlabel('Wiek')
plt.ylabel('Liczba zgłoszonych oszustw')
plt.show()

fig, ax = plt.subplots()
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
sns.countplot(data=data, x='incident_type', hue='fraud_reported', ax=ax)
plt.xlabel('Rodzaj zdarzenia')
plt.ylabel('Liczba zgłoszonych oszustw')
plt.show()

fig, ax = plt.subplots()
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
sns.heatmap(data.select_dtypes(include=['int64', 'float64']).corr(), annot=False, ax=ax)
plt.title('Macierz korelacji cech numerycznych')
plt.xticks(rotation=45)
plt.show()


print(pd.crosstab(data['is_weekend'], data['fraud_reported'], normalize='index'))
"""


def drop_features(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop


new_data = drop_features(data.select_dtypes(include=['int64', 'float64']))[0]
final = pd.concat([new_data, data.drop(columns=data.select_dtypes(include=['int64', 'float64']), axis=1)], axis=1)
final['policy_bind_year'] = final['policy_bind_date'].dt.year
final['policy_bind_month'] = final['policy_bind_date'].dt.month
final['policy_bind_day'] = final['policy_bind_date'].dt.day
final['incident_year'] = final['incident_date'].dt.year
final['incident_month'] = final['incident_date'].dt.month
final['incident_day'] = final['incident_date'].dt.day
final.drop(columns=['policy_bind_date', 'incident_date'], inplace=True)
x = final.select_dtypes(include=['object', 'category']).columns
for col in x:
    le = LabelEncoder()
    final[col] = le.fit_transform(final[col])

X = final.drop(columns=['fraud_reported'])
y = final['fraud_reported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeModel(max_depth=5)
rf = RandomForestModel(n_estimators=100)
gb = GradientBoostingModel(n_estimators=100)

comparer = ModelComparer(models=[dt, rf, gb])
comparer.fit_all(X_train, y_train)
comparer.evaluate_all(X_test, y_test)
comparer.summary()

dt.report(X_test, y_test)
rf.report(X_test, y_test)
gb.report(X_test, y_test)
