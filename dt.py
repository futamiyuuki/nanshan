import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris


# Load the Iris dataset
# iris = load_iris()
# X = iris.data
# y = iris.target

# Y Vars
USE_Y_REPURCHASE = True
USE_Y_AFYP = False
USE_Y_POL_CNT = False
def get_purchase_behaviors():
    if USE_Y_REPURCHASE:
        return ["tot_txn_cnt_his", "tot_txn_cnt"]
    if USE_Y_AFYP:
        return ["TOT_AFYP"]
    if USE_Y_POL_CNT:
        return ["tot_pol_cnt_his"]
    return []
PURCHASE_BEHAVIORS = get_purchase_behaviors()

# X Vars
USE_PRODUCT_COUNT_VS_RATE = True
USE_SIMPLE_PRODUCT_CATEGORIES = False
USE_PRODUCT_AFYP = False
IDENTITY_COLS = ["AGE", "GENDER"]
WEALTH_COLS = ["CLIENT_INCOME", "total_aum"]
CUSTOMER_STATUS_COLS = ["ternure_m", "recency_m", "topcard", "UNDERTAKE", "LABEL_NUM"]
CUSTOMER_PAYMENT_COLS = ["SELFPAY_CUST_FLG", "LOAN_CUST_FLG", "APL_CUST_FLG"]
CUSTOMER_PRODUCT_CHAR = ["prodtype_density", "prodtype_density_his"] if USE_SIMPLE_PRODUCT_CATEGORIES else ["product_density", "product_density_his"]

PRODUCT_CNT_SIMPLE_COLS = ["SIN_POL_CNT", "REG_POL_CNT", "ILP_POL_CNT", "AH_POL_CNT"]
PRODUCT_CNT_COMPLEX_COLS = ["SIN1_POL_CNT", "SIN2_POL_CNT", "REG1_POL_CNT", "REG2_POL_CNT", "ILP1_POL_CNT", "ILP2_POL_CNT",
                "AHa_POL_CNT", "AHb_POL_CNT", "AHc_POL_CNT", "AHd_POL_CNT"]
PRODUCT_CNT_COLS = PRODUCT_CNT_SIMPLE_COLS if USE_SIMPLE_PRODUCT_CATEGORIES else PRODUCT_CNT_COMPLEX_COLS
PRODUCT_RATE_SIMPLE_COLS = ["SIN_POL_RATE", "REG_POL_RATE", "ILP_POL_RATE", "AH_POL_RATE"]
PRODUCT_RATE_COMPLEX_COLS = ["SIN1_POL_RATE", "SIN2_POL_RATE", "REG1_POL_RATE", "REG2_POL_RATE", "ILP1_POL_RATE", "ILP2_POL_RATE",
                     "AHa_POL_RATE", "AHb_POL_RATE", "AHc_POL_RATE", "AHd_POL_RATE"]
PRODUCT_RATE_COLS = PRODUCT_RATE_SIMPLE_COLS if USE_SIMPLE_PRODUCT_CATEGORIES else PRODUCT_RATE_COMPLEX_COLS
PRODUCT_VOL_COLS = PRODUCT_CNT_COLS if USE_PRODUCT_COUNT_VS_RATE else PRODUCT_RATE_COLS

PRODUCT_AFYP_CNT_SIMPLE_COLS = ["SIN_AFYP", "REG_AFYP", "ILP_AFYP", "AH_AFYP"]
PRODUCT_AFYP_CNT_COMPLEX_COLS = ["SIN1_AFYP", "SIN2_AFYP", "REG1_AFYP", "REG2_AFYP", "ILP1_AFYP", "ILP2_AFYP",
                                 "AHa_AFYP", "AHb_AFYP", "AHc_AFYP", "AHd_AFYP"]
PRODUCT_AFYP_CNT_COLS = PRODUCT_AFYP_CNT_SIMPLE_COLS if USE_SIMPLE_PRODUCT_CATEGORIES else PRODUCT_AFYP_CNT_COMPLEX_COLS
PRODUCT_AFYP_RATE_SIMPLE_COLS = ["SIN_AFYP_RATE", "REG_AFYP_RATE", "ILP_AFYP_RATE", "AH_AFYP_RATE"]
PRODUCT_AFYP_RATE_COMPLEX_COLS = ["SIN1_AFYP_RATE", "SIN2_AFYP_RATE", "REG1_AFYP_RATE", "REG2_AFYP_RATE", "ILP1_AFYP_RATE",
                                  "ILP2_AFYP_RATE", "AHa_AFYP_RATE", "AHb_AFYP_RATE", "AHc_AFYP_RATE", "AHd_AFYP_RATE"]
PRODUCT_AFYP_RATE_COLS = PRODUCT_AFYP_RATE_SIMPLE_COLS if USE_SIMPLE_PRODUCT_CATEGORIES else PRODUCT_AFYP_RATE_COMPLEX_COLS
PRODUCT_AFYP_COLS = PRODUCT_AFYP_CNT_COLS if USE_PRODUCT_COUNT_VS_RATE else PRODUCT_AFYP_RATE_COLS

PRODUCT_COLS = PRODUCT_AFYP_COLS if USE_PRODUCT_AFYP else PRODUCT_VOL_COLS

# Load dataset
DATA_COLS = [PURCHASE_BEHAVIORS, IDENTITY_COLS, WEALTH_COLS, CUSTOMER_STATUS_COLS, CUSTOMER_PAYMENT_COLS, CUSTOMER_PRODUCT_CHAR, PRODUCT_COLS]
cols = reduce(lambda s, c: s+c, DATA_COLS)
print("Using Columns", cols)
df0 = pd.read_csv("../data/sas/sas0.csv", usecols=cols)
df8000 = pd.read_csv("../data/sas/sas8000.csv", usecols=cols)
df16000 = pd.read_csv("../data/sas/sas16000.csv", usecols=cols)
df24000 = pd.read_csv("../data/sas/sas24000.csv", usecols=cols)
df32000 = pd.read_csv("../data/sas/sas32000.csv", usecols=cols)
df40000 = pd.read_csv("../data/sas/sas40000.csv", usecols=cols)
df48000 = pd.read_csv("../data/sas/sas48000.csv", usecols=cols)
df56000 = pd.read_csv("../data/sas/sas56000.csv", usecols=cols)
df64000 = pd.read_csv("../data/sas/sas64000.csv", usecols=cols)
df72000 = pd.read_csv("../data/sas/sas72000.csv", usecols=cols)
df80000 = pd.read_csv("../data/sas/sas80000.csv", usecols=cols)
df88000 = pd.read_csv("../data/sas/sas88000.csv", usecols=cols)
df96000 = pd.read_csv("../data/sas/sas96000.csv", usecols=cols)
dfs = [df0, df8000, df16000, df24000, df32000, df40000, df48000, df56000, df64000, df72000, df80000, df88000, df96000]
df = pd.concat(dfs, ignore_index=True)

def clean_and_convert(value):
    try:
        if isinstance(value, str) == False:
            return value
        return float(value.replace('%', '').replace(',', ''))
    except ValueError:
        # Return 0 indicating conversion was not possible
        return 0
for prc in PRODUCT_COLS: df[prc] = df[prc].apply(clean_and_convert)

# def filter_df():
#     if USE_Y_REPURCHASE:
#         cond = df["tot_txn_cnt_his"] - df["tot_txn_cnt"] > 0
#         return df[cond]
#     return df
# df = filter_df()
display(df)

# count zeroes
def count_zero(series):
    return len(series) - np.count_nonzero(series)

print("Income 0's", count_zero(df["CLIENT_INCOME"]))
print("Asset 0's", count_zero(df["total_aum"]))

# X & y
X = df.drop(PURCHASE_BEHAVIORS, axis=1)
# y = df["tot_pol_cnt_his"]
def get_y_repurchase(row):
    if row["recency_m"] > row["ternure_m"] and row["tot_txn_cnt_his"] >= 2:
        return "True"
    else:
        return "False"

def get_y():
    if USE_Y_REPURCHASE:
        # df["repurchase"] = df["recency_m"] > df["ternure_m"] and df["tot_txn_cnt_his"] >= 2
        df["repurchase"] = df.apply(get_y_repurchase, axis=1)
        return df["repurchase"]
    if USE_Y_AFYP:
        return df["TOT_AFYP"]
    if USE_Y_POL_CNT:
        return df["tot_pol_cnt_his"]
    print("\n\nNo Y Flag!!!!!\n\n")
    return None
y = get_y()
display(X)
display(y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
tree.fit(X_train, y_train)

# Make predictions on the test set
predictions = tree.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, predictions))

# Plot the tree
# plt.figure(figsize=(20,10))
# plot_tree(tree, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
# plot_tree(tree, filled=True)
# plt.show()
