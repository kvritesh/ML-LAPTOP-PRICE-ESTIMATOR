import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# Load dataset
df = pd.read_csv('laptop_price.csv', encoding='latin1')

print(df.head())


# Remove missing values
df.dropna(inplace=True)


# Clean RAM column
df['Ram'] = df['Ram'].str.replace('GB', '')
df['Ram'] = df['Ram'].astype(int)


# Clean Weight column
df['Weight'] = df['Weight'].str.replace('kg', '')
df['Weight'] = df['Weight'].astype(float)


# Features and target
X = df[['Company',
        'TypeName',
        'Ram',
        'Weight',
        'OpSys',
        'Cpu',
        'Gpu',
        'Memory']]

y = df['Price_euros']


# Preprocessing
transformer = ColumnTransformer([
    ('tnf1',
     OneHotEncoder(handle_unknown='ignore'),
     ['Company', 'TypeName', 'OpSys', 'Cpu', 'Gpu', 'Memory'])
], remainder='passthrough')


# Model pipeline
pipe = Pipeline([
    ('transformer', transformer),
    ('model', RandomForestRegressor())
])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
pipe.fit(X_train, y_train)


# Predictions
y_pred = pipe.predict(X_test)


# Accuracy
print("R2 Score:", r2_score(y_test, y_pred))


# Save model
pickle.dump(pipe, open('pipe.pkl', 'wb'))

print("Model trained and saved successfully!")