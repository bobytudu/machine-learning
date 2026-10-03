# ==========================================
# 3. Feature Selection & Grouping
# ==========================================
# Continuous / numerical features that need mean imputation and standard scaling
NumericData = ['age', 'salary', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Categorical features with inherent order (unknown < primary < secondary < tertiary)
OrdinalData = ['education']

# Categorical features with no inherent order, to be one-hot encoded
NominalData = ['eligible', 'job', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
