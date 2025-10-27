import pandas as pd

data = pd.read_csv('insurance_claims.csv', sep=',')  # Wczytywanie zbioru danych z pliku CSV

"""
Sprawdzanie podstawowych informacji o zbiorze danych, typach danych, selekcja kolumn, które mogą być przydatne do analizy.
"""

# print(data.head())
# print(data.dtypes)
# print(data.describe())
# data.info()
# print(data.columns)
# print(f'Data shape: {data.shape}')
# print(f'Liczba wartości NaN: \n{data.isna().sum()}')

"""
Usuwanie prawdopodobnie niepotrzebnych kolumn.
"""

to_drop = ['_c39', 'policy_number', 'incident_location']
data = data.drop(to_drop, axis=1)
# print(data.head())

"""
Zamiana nieodpowiednich wartości na odpowiednie w wybranych kolumnach, poprawienie typów danych.
"""

data['authorities_contacted'] = data['authorities_contacted'].fillna('Unknown')
data[['collision_type', 'police_report_available']] = data[['collision_type', 'police_report_available']].replace('?', 'Unknown')

data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'])
data['incident_date'] = pd.to_datetime(data['incident_date'])

binary_columns = [
    'fraud_reported', 'police_report_available',
    'property_damage', 'authorities_contacted'
]
for col in binary_columns:
    data[col] = data[col].replace('?', 'Unknown')
    data[col] = data[col].astype('category')

categorical_columns = [
    'policy_state', 'insured_sex', 'insured_education_level',
    'insured_occupation', 'insured_hobbies', 'insured_relationship',
    'incident_type', 'collision_type', 'incident_severity',
    'incident_state', 'incident_city', 'auto_make', 'auto_model'
]
for col in categorical_columns:
    data[col] = data[col].astype('category')

# print(data.dtypes)
# print(data.head())
# print(data.isna().sum())

"""
Dodanie nowych kolumn
"""

data['policy_age_at_incident'] = (data['incident_date'] - data['policy_bind_date']).dt.days
data['incident_day_of_week'] = data['incident_date'].dt.dayofweek
data['incident_month'] = data['incident_date'].dt.month
data['is_weekend'] = data['incident_day_of_week'].isin([5, 6])
data['is_weekend'] = data['is_weekend'].map({True: 'Yes', False: 'No'})
data['is_weekend'] = data['is_weekend'].astype('category')
data['incident_hour_bucket'] = pd.cut(data['incident_hour_of_the_day'], bins=[-1, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
data['incident_hour_bucket'] = data['incident_hour_bucket'].astype('category')

# print(data.head())
# print(data.dtypes)

data.to_pickle('clean_data.pkl')
