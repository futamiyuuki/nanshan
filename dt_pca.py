import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

def clean_and_convert(value):
    try:
        if isinstance(value, str) == False:
            return value
        return float(value.replace('%', '').replace(',', ''))
    except ValueError:
        # Return 0 indicating conversion was not possible
        return 0
# for prc in PRODUCT_COLS: df[prc] = df[prc].apply(clean_and_convert)

# display(df)

# count zeroes
def count_zero(series):
    return len(series) - np.count_nonzero(series)
# print("Income 0's", count_zero(df["CLIENT_INCOME"]))
# print("Asset 0's", count_zero(df["total_aum"]))

# X & y
X = df.drop(PURCHASE_BEHAVIORS, axis=1)
# y = df["tot_pol_cnt_his"]
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
# display(X)
# display(y)

# Separate majority and minority classes
majority = df[df.Y1_repurchase == False]
minority = df[df.Y1_repurchase == True]

# Upsample minority class
n_len = math.floor(len(majority) / 5)
minority_upsampled = minority.sample(n=n_len, replace=True, random_state=42)  # replace=True for resampling

# Combine majority class with upsampled minority class
upsampled_data = pd.concat([majority, minority_upsampled])

# Shuffle the dataset to prevent any order bias
upsampled_data = upsampled_data.sample(frac=1, random_state=42)

# Separate features and labels after upsampling
X_upsampled = upsampled_data.drop(PURCHASE_BEHAVIORS, axis=1)
y_upsampled = upsampled_data['Y1_repurchase']

# Split the data into training+validation sets and test set
X_train_val, X_test, y_train_val, y_test = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)
# X_train_val, X_test, y_train_val, y_test = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)
# Split the training+validation set into separate training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Initialize the DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=15)

# Fit the model on the training data
tree.fit(X_train, y_train)

validation_predictions = tree.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, validation_predictions))
print("Valuation Classification Report:\n", classification_report(y_val, validation_predictions))

# Make predictions on the test set
predictions = tree.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Plot the tree
# plt.figure(figsize=(20,10))
# # plot_tree(tree, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
# plot_tree(tree, filled=True)
# plt.show()
