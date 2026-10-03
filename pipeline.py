# ==========================================
# 1. Imports
# ==========================================
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from columns import NumericData, OrdinalData, NominalData
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler


# ==========================================
# 2. Load Dataset
# ==========================================
# Read the dataset from CSV
df = pd.read_csv('bank-marketing.csv')
# print(df.info())


# ==========================================
# 4. Individual Preprocessing Sub-Pipelines
# ==========================================
# Numerical: fill missing values with mean, then standardize (zero mean, unit variance)
Numeric_pipeline = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='mean')),
    ('Scaling', StandardScaler()),
])

# Ordinal: fill missing values with most frequent, encode with specified ranking, then scale
Ordinal_pipeline = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('Encoding', OrdinalEncoder(categories=[['unknown', 'primary', 'secondary', 'tertiary']])),
    ('Scaling', StandardScaler())
])

# Nominal: fill missing values with most frequent, then convert to one-hot binary columns
Nominal_pipeline = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('Encoding', OneHotEncoder(handle_unknown='ignore'))
])


# ==========================================
# 5. Combine Preprocessors (ColumnTransformer)
# ==========================================
# Applies the appropriate sub-pipeline to each subset of columns
features = ColumnTransformer(transformers=[
    ("Numeric", Numeric_pipeline, NumericData),
    ("Ordinal", Ordinal_pipeline, OrdinalData),
    ("Nominal", Nominal_pipeline, NominalData)
])


# ==========================================
# 6. Full End-to-End Pipeline
# ==========================================
# Combines feature transformation + final classifier into a single reproducible estimator
model = Pipeline(steps=[
  ('features', features),
  ('classifier', LogisticRegression(max_iter=1000))
])


# ==========================================
# 7. Target Encoding & Train/Test Split
# ==========================================
# Encode target variable labels ('yes'/'no') into binary integers (1/0)
le = LabelEncoder()
Y = le.fit_transform(df['targeted'])

# Separate input features by dropping target and response columns
X = df.drop(columns=['targeted', 'response'])

# Split data: 80% training set, 20% testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# ==========================================
# 8. Train, Predict & Evaluate
# ==========================================
# Fit the pipeline: transforms features and fits the Logistic Regression model
model.fit(X_train, y_train)

# Predict outcomes on unseen test data
Y_predict = model.predict(X_test)

# Calculate and print the accuracy score
print(accuracy_score(y_test, Y_predict))
