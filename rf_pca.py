import math
import pandas as pd
from IPython.display import display
from functools import reduce
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Y Vars
USE_Y_REPURCHASE = True
USE_Y_AFYP = False
USE_Y_POL_CNT = False
def get_purchase_behaviors():
    if USE_Y_REPURCHASE:
        return ["Y1_repurchase"]
    if USE_Y_AFYP:
        return ["TOT_AFYP"]
    if USE_Y_POL_CNT:
        return ["tot_pol_cnt_his"]
    return []
PURCHASE_BEHAVIORS = get_purchase_behaviors()

COL_BASIC = ["AGE", "GENDER", "CLIENT_INCOME", "total_aum", "ternure_m", "recency_m", "topcard", "UNDERTAKE", "LABEL_NUM"]
COL_POL = ["SIN1_POL_CNT", "SIN2_POL_CNT", "REG1_POL_CNT", "REG2_POL_CNT", "ILP1_POL_CNT", "ILP2_POL_CNT", "AHa_POL_CNT", "AHb_POL_CNT", "AHc_POL_CNT", "AHd_POL_CNT", "product_density", "product_density_his"]
COL_CHAR_PC = ["characteristic_PC1", "characteristic_PC2", "characteristic_PC3", "characteristic_PC4"]
COL_LABEL_PC = ["label_PC1", "label_PC2", "label_PC3", "label_PC4", "label_PC5", "label_PC6", "label_PC7", "label_PC8", "label_PC9"]
COL_PRODUCT_PC = ["product_PC1", "product_PC2", "product_PC3", "product_PC4", "product_PC5", "product_PC6", "product_PC7", "product_PC8"]
COL_POLICY_PC = ["policy_ex_RATE_PC1", "policy_ex_RATE_PC2", "policy_ex_RATE_PC3", "policy_ex_RATE_PC4", "policy_ex_RATE_PC5", "policy_ex_RATE_PC6", "policy_ex_RATE_PC7", "policy_ex_RATE_PC8", "policy_ex_RATE_PC9", "policy_ex_RATE_PC10", "policy_ex_RATE_PC11", "policy_ex_RATE_PC12", "policy_ex_RATE_PC13", "policy_ex_RATE_PC14", "policy_ex_RATE_PC15", "policy_ex_RATE_PC16", "policy_ex_RATE_PC17", "policy_ex_RATE_PC18", "policy_ex_RATE_PC19", "policy_ex_RATE_PC20", "policy_ex_RATE_PC21", "policy_ex_RATE_PC22", "policy_ex_RATE_PC23", "policy_ex_RATE_PC24", "policy_ex_RATE_PC25", "policy_ex_RATE_PC26", "policy_ex_RATE_PC27"]
COL_CONTACT_PC = ["contact_PC1", "contact_PC2", "contact_PC3", "contact_PC4", "contact_PC5", "contact_PC6", "contact_PC7", "contact_PC8", "contact_PC9", "contact_PC10", "contact_PC11", "contact_PC12", "contact_PC13", "contact_PC14", "contact_PC15", "contact_PC16"]
COL_OTHER = ["Y1_repurchase"]
DATA_COLS = [COL_BASIC, COL_POL, COL_CHAR_PC, COL_LABEL_PC, COL_PRODUCT_PC, COL_POLICY_PC, COL_CONTACT_PC, COL_OTHER]
# DATA_COLS = [COL_BASIC, COL_POL, COL_OTHER]
cols = reduce(lambda s, c: s+c, DATA_COLS)
df = pd.read_csv("../data/post_PCA.csv", usecols=cols)

# Check for missing values
# display(df.isnull().sum().to_string())

# # Handle missing values, example: fill missing values with the median or mode
# df.fillna(df.median(), inplace=True)

# Convert categorical variables to numerical
# label_encoders = {}
# for column in ['gender', 'occupation', 'residence', 'product_type_density', 'VIP_status', 'membership_status']:
#     le = LabelEncoder()
#     df[column] = le.fit_transform(df[column])
#     label_encoders[column] = le

# Separate features and target
X = df.drop(PURCHASE_BEHAVIORS, axis=1)
def get_y():
    if USE_Y_REPURCHASE:
        return df["Y1_repurchase"]
    if USE_Y_AFYP:
        return df["TOT_AFYP"]
    if USE_Y_POL_CNT:
        return df["tot_pol_cnt_his"]
    print("\n\nNo Y Flag!!!!!\n\n")
    return None
y = get_y()

# Separate majority and minority classes
majority = df[df.Y1_repurchase == False]
minority = df[df.Y1_repurchase == True]
print("Minority Length", len(minority))

# Upsample minority class
n_len = math.floor(len(majority) / 20)
minority_upsampled = minority.sample(n=n_len, replace=True, random_state=42)  # replace=True for resampling
print("New Minority Length", len(minority_upsampled))

# Combine majority class with upsampled minority class
upsampled_data = pd.concat([majority, minority_upsampled])

# Shuffle the dataset to prevent any order bias
upsampled_data = upsampled_data.sample(frac=1, random_state=42)

# Separate features and labels after upsampling
X_upsampled = upsampled_data.drop(PURCHASE_BEHAVIORS, axis=1)
y_upsampled = upsampled_data['Y1_repurchase']

# smote = SMOTE(random_state=42, sampling_strategy='auto')
# X_train_smote, y_train_smote = smote.fit_resample(X_upsampled, y_upsampled)

# Split the data into training+validation sets and test set
X_train_val, X_test, y_train_val, y_test = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)
# X_train_val, X_test, y_train_val, y_test = train_test_split(X_train_smote, y_train_smote, test_size=0.2, random_state=42)
# Split the training+validation set into separate training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
# rf_model.fit(X_train, y_train)
rf_model.fit(X_train_smote, y_train_smote)

validation_predictions = rf_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, validation_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, validation_predictions))
print("Valuation Classification Report:\n", classification_report(y_val, validation_predictions))

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the predictions
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
print(feature_importance_df)

