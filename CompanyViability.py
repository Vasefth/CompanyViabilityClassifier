# imports
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from tabulate import tabulate

drive.mount('/content/drive')

# Task 1
# Get data from excel
data = pd.read_excel('/content/drive/MyDrive/Machine Learning/Data/Dataset2Use_Assignment1.xlsx')
data.head()

# Task 2a
# Count number of healthy and bankrupt businesses per year
counter = Counter()
for index, row in data.iterrows():
  business_status = row['ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)']
  year = row['ΕΤΟΣ']
  counter.update([(year, business_status)])

count1 = [] # Number of non-bankrupt businesses
year1 = []  # per year
count2 = [] # Number of bankrupt businesses
year2 = []  # per year


for item in counter.items():
  if item[0][1] == 1:
    year1.append(item[0][0])
    count1.append(item[1])
  else:
    year2.append(item[0][0])
    count2.append(item[1])

# Plot Stacked Bar Chart
plt.style.use('fivethirtyeight')
plt.bar(year1, count1)
plt.bar(year1, count2, bottom = count1)
plt.xticks(year1)
plt.xlabel("Έτη")
plt.ylabel("Πλήθος Επιχειρήσεων")
plt.legend(["Υγιείς Επιχειρήσεις", "Χρεωκοπημένες Επιχειρήσεις"],
           loc='upper center', bbox_to_anchor=(0.5, -0.15),
           fancybox=True, ncol=2)
plt.title("Πλήθος Υγιών και Χρεωκοπημένων Επιχειρήσεων ανά έτος\n")
plt.show()

# Task 2b

# Indicator column indexes
indicator_indexes = list(range(0, 10))  # Columns 1 to 11

# Extract unique years from the 'ΕΤΟΣ' column and sort them
unique_years = sorted(data.iloc[:, 12].unique())

# Process data for each indicator
for index in indicator_indexes:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))  # Two subfigures side by side

    # Initialize variables to find the global minimum and maximum y-values
    global_min = float('inf')
    global_max = float('-inf')

    # First pass to determine the global min and max for the y-axis range
    for status in [1, 2]:  # 1 for healthy, 2 for bankrupt
        filtered_data = data[data.iloc[:, 11] == status]
        grouped = filtered_data.groupby(filtered_data.iloc[:, 12])[filtered_data.columns[index]].agg(['min', 'max', 'mean'])
        current_min = grouped['min'].min()
        current_max = grouped['max'].max()
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)

    # Define a margin to add space above the max and below the min
    margin = (global_max - global_min) * 0.05  # 5% of the range as margin

    # Second pass to plot the data with a fixed y-axis range based on the global min and max
    for i, status in enumerate([1, 2]):  # 1 for healthy, 2 for bankrupt
        filtered_data = data[data.iloc[:, 11] == status]
        grouped = filtered_data.groupby(filtered_data.iloc[:, 12])[filtered_data.columns[index]].agg(['min', 'max', 'mean'])

        # Plotting for the current status
        axes[i].plot(grouped.index, grouped['min'], label='Min', marker='o')
        axes[i].plot(grouped.index, grouped['max'], label='Max', marker='o')
        axes[i].plot(grouped.index, grouped['mean'], label='Average', marker='o')
        axes[i].set_title(f'{"Healthy" if status == 1 else "Bankrupt"} Companies')
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel(data.columns[index])  # Use the original column name
        axes[i].set_xticks(unique_years)

        # Set the same y-axis range for both subplots with added margin
        axes[i].set_ylim(global_min - margin, global_max + margin)

        axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                       fancybox=True, ncol=3)

    plt.suptitle(f'Indicator: {data.columns[index]}')  # Use the original column name for the super title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout
    plt.show()

# Task 3
# Find missing values
# Check if there are NaN or null values
data_has_NaN_values = data.isna().any().any()
data_has_null_values = data.isnull().any().any()

if not data_has_NaN_values:
  print("There are no NaN values in the dataset")
else:
  for i in list(range(data.shape[0])):
    for j in list(1, range(data.shape[1])):
      if data.isna().iloc[i,j]:
        print("NaN value in row", i+1, "and column", j+1)
if not data_has_null_values:
  print("There are no null values in the dataset")
else:
  for i in list(range(data.shape[0])):
    for j in list(1, range(data.shape[1])):
      if data.isnull().iloc[i,j]:
        print("Null value in row", i+1, "and column", j+1)

# Task 4

data = data.drop('ΕΤΟΣ', axis=1)
# Perform Min-Max scaling on the selected columns
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
data.head()

import random

# Given data
total_companies = 10716
healthy_companies = 10468
bankrupt_companies = 248
k_fold = 4

# Calculate the number of bankrupt companies in the train set
train_bankrupt_lim = int(bankrupt_companies * 0.75)
# Calculate the number of healthy companies needed to maintain a 3:1 ratio in the train set
train_healthy_lim = train_bankrupt_lim * 3
# Ensure the number of healthy companies in the train set doesn't exceed the available healthy companies
train_healthy_lim = min(train_healthy_lim, healthy_companies)
# Calculate the number of bankrupt companies needed to balance the train set
train_bankrupt_lim = train_healthy_lim // 3

print("=== Train (3:1 ratio) ===")
print(f"Healthy: {train_healthy_lim}")
print(f"Bankrupt: {train_bankrupt_lim}")

print("\n=== Test ===")
print(f"Healthy: {healthy_companies - train_healthy_lim}")
print(f"Bankrupt: {bankrupt_companies - train_bankrupt_lim}")

def plot_confusion_matrices(model1,model2,cm_train_model1, cm_test_model1, metrics_model1,
                            cm_train_model2, cm_test_model2, metrics_model2, fold):
    """
    Plots confusion matrices for two models along with their performance metrics.

    :param model1: Name of model 1
    :param model2: Name of model 2
    :param cm_train_model1: Confusion matrix for training set of model 1
    :param cm_test_model1: Confusion matrix for testing set of model 1
    :param metrics_model1: A dictionary containing accuracy, precision, recall, F1, and ROC AUC for model 1
    :param cm_train_model2: Confusion matrix for training set of model 2
    :param cm_test_model2: Confusion matrix for testing set of model 2
    :param metrics_model2: A dictionary containing accuracy, precision, recall, F1, and ROC AUC for model 2
    :param fold: The fold number (if applicable)
    """
    # Plotting confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Fold {fold + 1} Metrics', fontsize=16)

    # Define a smaller font size for subtitles and axis labels
    subtitle_fontsize = 9
    annot_fontsize = 12  # Size of the numbers inside the heatmap
    axis_label_fontsize = 10  # Font size for the axis labels

    # Define common axis labels
    x_axis_label = "Predicted Label"
    y_axis_label = "True Label"

    # Plot for Model 1 - Training Set
    sns.heatmap(cm_train_model1, annot=True, ax=axes[0, 0], fmt="d", annot_kws={"size": annot_fontsize})
    axes[0, 0].set_title(f"{model1} - Training Set\n"
                      f"Accuracy: {metrics_model1['train']['accuracy']:.2f}, "
                      f"Precision: {metrics_model1['train']['precision']:.2f}, "
                      f"Recall: {metrics_model1['train']['recall']:.2f}, "
                      f"F1: {metrics_model1['train']['f1']:.2f}, "
                      f"ROC AUC: {metrics_model1['train']['roc_auc']:.2f}", fontsize=subtitle_fontsize)

    # Plot for Model 1 - Testing Set
    sns.heatmap(cm_test_model1, annot=True, ax=axes[0, 1], fmt="d", annot_kws={"size": annot_fontsize})
    axes[0, 1].set_title(f"{model1} - Testing Set\n"
                      f"Accuracy: {metrics_model1['test']['accuracy']:.2f}, "
                      f"Precision: {metrics_model1['test']['precision']:.2f}, "
                      f"Recall: {metrics_model1['test']['recall']:.2f}, "
                      f"F1: {metrics_model1['test']['f1']:.2f}, "
                      f"ROC AUC: {metrics_model1['test']['roc_auc']:.2f}", fontsize=subtitle_fontsize)

    # Plot for Model 2 - Training Set
    sns.heatmap(cm_train_model2, annot=True, ax=axes[1, 0], fmt="d", annot_kws={"size": annot_fontsize})
    axes[1, 0].set_title(f"{model2} - Training Set\n"
                      f"Accuracy: {metrics_model2['train']['accuracy']:.2f}, "
                      f"Precision: {metrics_model2['train']['precision']:.2f}, "
                      f"Recall: {metrics_model2['train']['recall']:.2f}, "
                      f"F1: {metrics_model2['train']['f1']:.2f}, "
                      f"ROC AUC: {metrics_model2['train']['roc_auc']:.2f}", fontsize=subtitle_fontsize)

    # Plot for Model 2 - Testing Set
    sns.heatmap(cm_test_model2, annot=True, ax=axes[1, 1], fmt="d", annot_kws={"size": annot_fontsize})
    axes[1, 1].set_title(f"{model2} - Testing Set\n"
                      f"Accuracy: {metrics_model2['test']['accuracy']:.2f}, "
                      f"Precision: {metrics_model2['test']['precision']:.2f}, "
                      f"Recall: {metrics_model2['test']['recall']:.2f}, "
                      f"F1: {metrics_model2['test']['f1']:.2f}, "
                      f"ROC AUC: {metrics_model2['test']['roc_auc']:.2f}", fontsize=subtitle_fontsize)

    # Setting common labels and adjustments
    for ax in axes.flat:
        ax.set_xlabel(x_axis_label, fontsize=axis_label_fontsize)
        ax.set_ylabel(y_axis_label, fontsize=axis_label_fontsize)
        ax.set_xticklabels(['Healthy', 'Bankrupt'])
        ax.set_yticklabels(['Healthy', 'Bankrupt'])

    # Adjust layout spacing to prevent overlap
    plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.3, wspace=0.2)
    plt.show()

# Set a seed for reproducibility
np.random.seed(42)

# Create an empty DataFrame to store the results
unbalanced_results_df = pd.DataFrame(columns=[
    "Classifier Name",
    "Dataset",
    "Balance",
    "Number of Training Samples",
    "Number of Bankrupt Companies",
    "TP",
    "TN",
    "FP",
    "FN",
    "ROC-AUC"
])

# Iterate through each fold
skf = StratifiedKFold(n_splits=4)

for fold, (train_index, test_index) in enumerate(skf.split(data, data["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"])):
    # Retrieve the train and test sets
    train_set = data.iloc[train_index]
    test_set = data.iloc[test_index]

    # Calculating the number of healthy and bankrupt companies in the train and test sets
    train_healthy = train_set[train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 0.0]
    train_bankrupt = train_set[train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 1.0]

    # ==========================
    X_train = train_set.drop(columns=["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"])
    y_train = train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"]

    X_test = test_set.drop(columns=["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"])
    y_test = test_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"]

    # Create a Linear Discriminant Analysis Classifier
    lda = LinearDiscriminantAnalysis(solver="lsqr")
    # Create a Logistic Regression model
    logreg = LogisticRegression(solver="liblinear", max_iter=200)
    # Create a Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth = 5)
    # Create a Random Forest Classifier
    rf = RandomForestClassifier(min_samples_split = 4, max_depth = 8)
    # Create a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors = 3)
    # Create a Naive Bayes Classifier
    nb = GaussianNB()
    # Create an SVM Classifier
    svm = SVC(random_state=42, C=10, kernel='rbf', probability=True)
    # Create an MLP Classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(12,),
        activation='relu',
        solver='lbfgs',
        alpha=0.0001,
        max_iter=10000,
        early_stopping=True,
        random_state=42
    )
    # Fit the LDA to the data
    lda.fit(X_train, y_train)
    # Fit the Logistic Regression to the data
    logreg.fit(X_train, y_train)
    # Fit the Decision Tree to the data
    dt.fit(X_train, y_train)
    # Fit the Random Forest to the data
    rf.fit(X_train, y_train)
    # Fit the KNN to the data
    knn.fit(X_train, y_train)
    # Fit the Naive Bayes to the data
    nb.fit(X_train, y_train)
    # Fit the SVM to the data
    svm.fit(X_train, y_train)
    # Fit the MLP to the data
    mlp.fit(X_train, y_train)

    # Predictions and probabilities for the test set
    y_pred_test_lda = lda.predict(X_test)
    y_prob_test_lda = lda.predict_proba(X_test)[:, 1]
    # Predictions and probabilities for the test set
    y_pred_test_logreg = logreg.predict(X_test)
    y_prob_test_logreg = logreg.predict_proba(X_test)[:, 1]
    # Decision Tree | Predictions and probabilities for the test set
    y_pred_test_dt = dt.predict(X_test)
    y_prob_test_dt = dt.predict_proba(X_test)[:,1]
    # Random Forest | Predictions and probabilities for the test set
    y_pred_test_rf = rf.predict(X_test)
    y_prob_test_rf = rf.predict_proba(X_test)[:,1]
    # KNN | Predictions and probabilities for the test set
    y_pred_test_knn = knn.predict(X_test)
    y_prob_test_knn = knn.predict_proba(X_test)[:, 1]
    # Naive Bayes | Predictions and probabilities for the test set
    y_pred_test_nb = nb.predict(X_test)
    y_prob_test_nb = nb.predict_proba(X_test)[:, 1]
    # SVM | Predictions and probabilities for the test set
    y_pred_test_svm = svm.predict(X_test)
    y_prob_test_svm = svm.predict_proba(X_test)[:, 1]
    # MLP | Predictions and probabilities for the test set
    y_pred_test_mlp = mlp.predict(X_test)
    y_prob_test_mlp = mlp.predict_proba(X_test)[:, 1]

    # LDA | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_lda = lda.predict(X_train)
    y_prob_train_lda = lda.predict_proba(X_train)[:, 1]
    # Logistic Regression | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_logreg = logreg.predict(X_train)
    y_prob_train_logreg = logreg.predict_proba(X_train)[:, 1]
    # Decision Tree | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_dt = dt.predict(X_train)
    y_prob_train_dt = dt.predict_proba(X_train)[:, 1]
    # Random Forest | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_rf = rf.predict(X_train)
    y_prob_train_rf = rf.predict_proba(X_train)[:, 1]
    # KNN | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_knn = knn.predict(X_train)
    y_prob_train_knn = knn.predict_proba(X_train)[:, 1]
    # Naive Bayes | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_nb = nb.predict(X_train)
    y_prob_train_nb = nb.predict_proba(X_train)[:, 1]
    # SVM | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_svm = svm.predict(X_train)
    y_prob_train_svm = svm.predict_proba(X_train)[:, 1]
    # MLP | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_mlp = mlp.predict(X_train)
    y_prob_train_mlp = mlp.predict_proba(X_train)[:, 1]

    # Calculate confusion matrices
    cm_train_lda = confusion_matrix(y_train, y_pred_train_lda)
    cm_test_lda = confusion_matrix(y_test, y_pred_test_lda)
    cm_train_logreg = confusion_matrix(y_train, y_pred_train_logreg)
    cm_test_logreg = confusion_matrix(y_test, y_pred_test_logreg)
    cm_train_dt = confusion_matrix(y_train, y_pred_train_dt)
    cm_test_dt = confusion_matrix(y_test, y_pred_test_dt)
    cm_train_rf = confusion_matrix(y_train, y_pred_train_rf)
    cm_test_rf = confusion_matrix(y_test, y_pred_test_rf)
    cm_train_knn = confusion_matrix(y_train, y_pred_train_knn)
    cm_test_knn = confusion_matrix(y_test, y_pred_test_knn)
    cm_train_nb = confusion_matrix(y_train, y_pred_train_nb)
    cm_test_nb = confusion_matrix(y_test, y_pred_test_nb)
    cm_train_svm = confusion_matrix(y_train, y_pred_train_svm)
    cm_test_svm = confusion_matrix(y_test, y_pred_test_svm)
    cm_train_mlp = confusion_matrix(y_train, y_pred_train_mlp)
    cm_test_mlp = confusion_matrix(y_test, y_pred_test_mlp)

    # LDA | Calculate performance metrics for the test set
    accuracy_test_lda = accuracy_score(y_test,y_pred_test_lda)
    precision_test_lda = precision_score(y_test,y_pred_test_lda, average='macro')
    recall_test_lda = recall_score(y_test,y_pred_test_lda, average='macro')
    f1_test_lda = f1_score(y_test,y_pred_test_lda, average='macro')
    roc_auc_test_lda = roc_auc_score(y_test, y_prob_test_lda, average='macro')

    # LDA | Calculate performance metrics for the train set
    accuracy_train_lda = accuracy_score(y_train, y_pred_train_lda)
    precision_train_lda = precision_score(y_train, y_pred_train_lda, average='macro')
    recall_train_lda = recall_score(y_train, y_pred_train_lda, average='macro')
    f1_train_lda = f1_score(y_train, y_pred_train_lda, average='macro')
    roc_auc_train_lda = roc_auc_score(y_train, y_prob_train_lda, average='macro')

    # Logistic Regression | Calculate performance metrics for the test set
    accuracy_test_logreg = accuracy_score(y_test,y_pred_test_logreg)
    precision_test_logreg = precision_score(y_test,y_pred_test_logreg, average='macro')
    recall_test_logreg = recall_score(y_test,y_pred_test_logreg, average='macro')
    f1_test_logreg = f1_score(y_test,y_pred_test_logreg, average='macro')
    roc_auc_test_logreg = roc_auc_score(y_test, y_prob_test_logreg, average='macro')

    # Logistic Regression | Calculate performance metrics for the train set
    accuracy_train_logreg = accuracy_score(y_train, y_pred_train_logreg)
    precision_train_logreg = precision_score(y_train, y_pred_train_logreg, average='macro')
    recall_train_logreg = recall_score(y_train, y_pred_train_logreg, average='macro')
    f1_train_logreg = f1_score(y_train, y_pred_train_logreg, average='macro')
    roc_auc_train_logreg = roc_auc_score(y_train, y_prob_train_logreg, average='macro')

    # Decision Tree | Calculate performance metrics for the test set
    accuracy_test_dt = accuracy_score(y_test, y_pred_test_dt)
    precision_test_dt = precision_score(y_test, y_pred_test_dt, average='macro')
    recall_test_dt = recall_score(y_test, y_pred_test_dt, average='macro')
    f1_test_dt = f1_score(y_test, y_pred_test_dt, average='macro')
    roc_auc_test_dt = roc_auc_score(y_test, y_prob_test_dt, average='macro')

    # Decision Tree | Calculate performance metrics for the train set
    accuracy_train_dt = accuracy_score(y_train, y_pred_train_dt)
    precision_train_dt = precision_score(y_train, y_pred_train_dt, average='macro')
    recall_train_dt = recall_score(y_train, y_pred_train_dt, average='macro')
    f1_train_dt = f1_score(y_train, y_pred_train_dt, average='macro')
    roc_auc_train_dt = roc_auc_score(y_train, y_prob_train_dt, average='macro')

    # Random Forest | Calculate performance metrics for the test set
    accuracy_test_rf = accuracy_score(y_test, y_pred_test_rf)
    precision_test_rf = precision_score(y_test, y_pred_test_rf, average='macro')
    recall_test_rf = recall_score(y_test, y_pred_test_rf, average='macro')
    f1_test_rf = f1_score(y_test, y_pred_test_rf, average='macro')
    roc_auc_test_rf = roc_auc_score(y_test, y_prob_test_rf, average='macro')

    # Random Forest | Calculate performance metrics for the train set
    accuracy_train_rf = accuracy_score(y_train, y_pred_train_rf)
    precision_train_rf = precision_score(y_train, y_pred_train_rf, average='macro')
    recall_train_rf = recall_score(y_train, y_pred_train_rf, average='macro')
    f1_train_rf = f1_score(y_train, y_pred_train_rf, average='macro')
    roc_auc_train_rf = roc_auc_score(y_train, y_prob_train_rf, average='macro')

    # KNN | Calculate performance metrics for the test set
    accuracy_test_knn = accuracy_score(y_test, y_pred_test_knn)
    precision_test_knn = precision_score(y_test, y_pred_test_knn, average='macro')
    recall_test_knn = recall_score(y_test, y_pred_test_knn, average='macro')
    f1_test_knn = f1_score(y_test, y_pred_test_knn, average='macro')
    roc_auc_test_knn = roc_auc_score(y_test, y_prob_test_knn, average='macro')

    # KNN | Calculate performance metrics for the train set
    accuracy_train_knn = accuracy_score(y_train, y_pred_train_knn)
    precision_train_knn = precision_score(y_train, y_pred_train_knn, average='macro')
    recall_train_knn = recall_score(y_train, y_pred_train_knn, average='macro')
    f1_train_knn = f1_score(y_train, y_pred_train_knn, average='macro')
    roc_auc_train_knn = roc_auc_score(y_train, y_prob_train_knn, average='macro')

    # Naive Bayes | Calculate performance metrics for the test set
    accuracy_test_nb = accuracy_score(y_test, y_pred_test_nb)
    precision_test_nb = precision_score(y_test, y_pred_test_nb, average='macro')
    recall_test_nb = recall_score(y_test, y_pred_test_nb, average='macro')
    f1_test_nb = f1_score(y_test, y_pred_test_nb, average='macro')
    roc_auc_test_nb = roc_auc_score(y_test, y_prob_test_nb, average='macro')

    # Naive Bayes | Calculate performance metrics for the train set
    accuracy_train_nb = accuracy_score(y_train, y_pred_train_nb)
    precision_train_nb = precision_score(y_train, y_pred_train_nb, average='macro')
    recall_train_nb = recall_score(y_train, y_pred_train_nb, average='macro')
    f1_train_nb = f1_score(y_train, y_pred_train_nb, average='macro')
    roc_auc_train_nb = roc_auc_score(y_train, y_prob_train_nb, average='macro')

    # SVM | Calculate performance metrics for the test set
    accuracy_test_svm = accuracy_score(y_test, y_pred_test_svm)
    precision_test_svm = precision_score(y_test, y_pred_test_svm, average='macro')
    recall_test_svm = recall_score(y_test, y_pred_test_svm, average='macro')
    f1_test_svm = f1_score(y_test, y_pred_test_svm, average='macro')
    roc_auc_test_svm = roc_auc_score(y_test, y_prob_test_svm, average='macro')
    # SVM | Calculate performance metrics for the train set
    accuracy_train_svm = accuracy_score(y_train, y_pred_train_svm)
    precision_train_svm = precision_score(y_train, y_pred_train_svm, average='macro')
    recall_train_svm = recall_score(y_train, y_pred_train_svm, average='macro')
    f1_train_svm = f1_score(y_train, y_pred_train_svm, average='macro')
    roc_auc_train_svm = roc_auc_score(y_train, y_prob_train_svm, average='macro')

    # MLP | Calculate performance metrics for the test set
    accuracy_test_mlp = accuracy_score(y_test, y_pred_test_mlp)
    precision_test_mlp = precision_score(y_test, y_pred_test_mlp, average='macro')
    recall_test_mlp = recall_score(y_test, y_pred_test_mlp, average='macro')
    f1_test_mlp = f1_score(y_test, y_pred_test_mlp, average='macro')
    roc_auc_test_mlp = roc_auc_score(y_test, y_prob_test_mlp, average='macro')
    # MLP | Calculate performance metrics for the train set
    accuracy_train_mlp = accuracy_score(y_train, y_pred_train_mlp)
    precision_train_mlp = precision_score(y_train, y_pred_train_mlp, average='macro')
    recall_train_mlp = recall_score(y_train, y_pred_train_mlp, average='macro')
    f1_train_mlp = f1_score(y_train, y_pred_train_mlp, average='macro')
    roc_auc_train_mlp = roc_auc_score(y_train, y_prob_train_mlp, average='macro')

    # ==========================

    # Recalculate the counts after moving the companies
    train_healthy_count = train_set[train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 0.0].shape[0]
    train_bankrupt_count = train_set[train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 1.0].shape[0]
    test_healthy_count = test_set[test_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 0.0].shape[0]
    test_bankrupt_count = test_set[test_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 1.0].shape[0]

    # Print the results
    print(f"==== Fold {fold + 1} (After Balancing) ====")
    print(f"Train: {train_bankrupt_count} bankrupt, {train_healthy_count} healthy, Total: {len(train_set)}")
    print(f"Test: {test_bankrupt_count} bankrupt, {test_healthy_count} healthy, Total: {len(test_set)}")

    metrics_lda = {
    'train': {'accuracy': accuracy_train_lda, 'precision': precision_train_lda,
              'recall': recall_train_lda, 'f1': f1_train_lda, 'roc_auc': roc_auc_train_lda},
    'test': {'accuracy': accuracy_test_lda, 'precision': precision_test_lda,
             'recall': recall_test_lda, 'f1': f1_test_lda, 'roc_auc': roc_auc_test_lda}
    }

    metrics_logreg = {
    'train': {'accuracy': accuracy_train_logreg, 'precision': precision_train_logreg,
              'recall': recall_train_logreg, 'f1': f1_train_logreg, 'roc_auc': roc_auc_train_logreg},
    'test': {'accuracy': accuracy_test_logreg, 'precision': precision_test_logreg,
             'recall': recall_test_logreg, 'f1': f1_test_logreg, 'roc_auc': roc_auc_test_logreg}
    }

    metrics_dt = {
    'train': {'accuracy': accuracy_train_dt, 'precision': precision_train_dt,
              'recall': recall_train_dt, 'f1': f1_train_dt, 'roc_auc': roc_auc_train_dt},
    'test': {'accuracy': accuracy_test_dt, 'precision': precision_test_dt,
             'recall': recall_test_dt, 'f1': f1_test_dt, 'roc_auc': roc_auc_test_dt}
    }

    metrics_rf = {
    'train': {'accuracy': accuracy_train_rf, 'precision': precision_train_rf,
              'recall': recall_train_rf, 'f1': f1_train_rf, 'roc_auc': roc_auc_train_rf},
    'test': {'accuracy': accuracy_test_rf, 'precision': precision_test_rf,
             'recall': recall_test_rf, 'f1': f1_test_rf, 'roc_auc': roc_auc_test_rf}
    }

    metrics_knn = {
    'train': {'accuracy': accuracy_train_knn, 'precision': precision_train_knn,
              'recall': recall_train_knn, 'f1': f1_train_knn, 'roc_auc': roc_auc_train_knn},
    'test': {'accuracy': accuracy_test_knn, 'precision': precision_test_knn,
             'recall': recall_test_knn, 'f1': f1_test_knn, 'roc_auc': roc_auc_test_knn}
    }

    metrics_nb = {
    'train': {'accuracy': accuracy_train_nb, 'precision': precision_train_nb,
              'recall': recall_train_nb, 'f1': f1_train_nb, 'roc_auc': roc_auc_train_nb},
    'test': {'accuracy': accuracy_test_nb, 'precision': precision_test_nb,
             'recall': recall_test_nb, 'f1': f1_test_nb, 'roc_auc': roc_auc_test_nb}
    }

    metrics_svm = {
    'train': {'accuracy': accuracy_train_svm, 'precision': precision_train_svm,
              'recall': recall_train_svm, 'f1': f1_train_svm, 'roc_auc': roc_auc_train_svm},
    'test': {'accuracy': accuracy_test_svm, 'precision': precision_test_svm,
             'recall': recall_test_svm, 'f1': f1_test_svm, 'roc_auc': roc_auc_test_svm}
    }

    metrics_mlp = {
    'train': {'accuracy': accuracy_train_mlp, 'precision': precision_train_mlp,
              'recall': recall_train_mlp, 'f1': f1_train_mlp, 'roc_auc': roc_auc_train_mlp},
    'test': {'accuracy': accuracy_test_mlp, 'precision': precision_test_mlp,
             'recall': recall_test_mlp, 'f1': f1_test_mlp, 'roc_auc': roc_auc_test_mlp}
    }

    plot_confusion_matrices(lda,logreg,cm_train_lda, cm_test_lda, metrics_lda,
                        cm_train_logreg, cm_test_logreg, metrics_logreg, fold)
    plot_confusion_matrices(dt, rf, cm_train_dt, cm_test_dt, metrics_dt,
                        cm_train_rf, cm_test_rf, metrics_rf, fold)
    plot_confusion_matrices(knn,nb,cm_train_knn, cm_test_knn, metrics_knn,
                        cm_train_nb, cm_test_nb, metrics_nb, fold)
    plot_confusion_matrices(svm, mlp, cm_train_svm, cm_test_svm, metrics_svm,
                        cm_train_mlp, cm_test_mlp, metrics_mlp, fold)


    # For each classifier and dataset (train/test), record the information
    for clf, clf_name, clf_name_actual in [(lda, "lda", "Linear Discrimination Analysis"), (logreg, "logreg", "Logistic Regression"),
                                           (dt, "dt", "Decision Tree"), (rf, "rf", "Random Forest"),
                                           (knn, "knn", "KNN"), (nb, "nb", "Naive Bayes"),
                                           (svm, "svm", "Support Vector Machine"), (mlp, "mlp", "Multilayer Perceptron")]:
        # Define the predictions and probabilities based on the classifier name
        if clf_name == "lda":
            y_pred_train = y_pred_train_lda
            y_prob_train = y_prob_train_lda
            y_pred_test = y_pred_test_lda
            y_prob_test = y_prob_test_lda
        elif clf_name == "logreg":
            y_pred_train = y_pred_train_logreg
            y_prob_train = y_prob_train_logreg
            y_pred_test = y_pred_test_logreg
            y_prob_test = y_prob_test_logreg
        elif clf_name == "dt":
            y_pred_train = y_pred_train_dt
            y_prob_train = y_prob_train_dt
            y_pred_test = y_pred_test_dt
            y_prob_test = y_prob_test_dt
        elif clf_name == "rf":
            y_pred_train = y_pred_train_rf
            y_prob_train = y_prob_train_rf
            y_pred_test = y_pred_test_rf
            y_prob_test = y_prob_test_rf
        elif clf_name == "knn":
            y_pred_train = y_pred_train_knn
            y_prob_train = y_prob_train_knn
            y_pred_test = y_pred_test_knn
            y_prob_test = y_prob_test_knn
        elif clf_name == "nb":
            y_pred_train = y_pred_train_nb
            y_prob_train = y_prob_train_nb
            y_pred_test = y_pred_test_nb
            y_prob_test = y_prob_test_nb
        elif clf_name == "svm":
            y_pred_train = y_pred_train_svm
            y_prob_train = y_prob_train_svm
            y_pred_test = y_pred_test_svm
            y_prob_test = y_prob_test_svm
        elif clf_name == "mlp":
            y_pred_train = y_pred_train_mlp
            y_prob_train = y_prob_train_mlp
            y_pred_test = y_pred_test_mlp
            y_prob_test = y_prob_test_mlp
        else:
            raise ValueError(f"Classifier {clf_name_actual} not recognized.")

        for dataset, X, y, set_name in [
            (train_set, X_train, y_train, "Training"),
            (test_set, X_test, y_test, "Test"),
            ]:

            # Generate predictions and probabilities based on the classifier
            y_pred = clf.predict(X)
            y_prob = clf.predict_proba(X)[:, 1]  # Assuming that the positive class is at index 1

            # Calculate confusion matrix and ROC AUC
            cm = confusion_matrix(y, y_pred)
            roc_auc = round(roc_auc_score(y, y_prob), 2)  # Round to 2 decimals

            # Append results to the DataFrame
            new_row = {
                "Classifier Name": clf_name_actual,
                "Dataset": set_name,
                "Balance": "Unbalanced",
                #if len(train_set[train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 0.0]) > train_healthy_lim else "Unbalanced",
                "Number of Training Samples": len(train_set) if set_name=="Training" else len(test_set) ,
                "Number of Bankrupt Companies": dataset[dataset["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 1.0].shape[0],
                "TP": cm[1, 1],
                "TN": cm[0, 0],
                "FP": cm[0, 1],
                "FN": cm[1, 0],
                "ROC-AUC": roc_auc
            }
            # Append new row using concat
            unbalanced_results_df = pd.concat([unbalanced_results_df, pd.DataFrame([new_row])], ignore_index=True)

# Print the DataFrame using tabulate after the folds
print(tabulate(unbalanced_results_df, headers='keys', tablefmt='psql', showindex=False))

# Set a seed for reproducibility
np.random.seed(42)

# Create an empty DataFrame to store the results
balanced_results_df = pd.DataFrame(columns=[
    "Classifier Name",
    "Dataset",
    "Balance",
    "Number of Training Samples",
    "Number of Bankrupt Companies",
    "TP",
    "TN",
    "FP",
    "FN",
    "ROC-AUC"
])

# Iterate through each fold
skf = StratifiedKFold(n_splits=4)

for fold, (train_index, test_index) in enumerate(skf.split(data, data["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"])):
    # Retrieve the train and test sets
    train_set = data.iloc[train_index]
    test_set = data.iloc[test_index]

    # Calculating the number of healthy and bankrupt companies in the train and test sets
    train_healthy = train_set[train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 0.0]
    train_bankrupt = train_set[train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 1.0]

    if len(train_healthy) > train_healthy_lim:
        # Select train_healthy_lim indices from healthy companies to keep in the training set
        keep_indices = np.random.choice(train_healthy.index, train_healthy_lim, replace=False)

        # Find the indices of healthy companies to move to the test set
        move_indices = train_healthy.index.difference(keep_indices)

        # Move the excess healthy companies to the test set
        test_set = pd.concat([test_set, train_set.loc[move_indices]], axis=0)

        # Keep the limited number of healthy companies in the training set
        train_set = pd.concat([train_set.loc[keep_indices], train_bankrupt], axis=0)

    # ==========================
    X_train = train_set.drop(columns=["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"])
    y_train = train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"]

    X_test = test_set.drop(columns=["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"])
    y_test = test_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"]

    # Create a Linear Discriminant Analysis Classifier
    lda = LinearDiscriminantAnalysis(solver="lsqr")
    # Create a Logistic Regression model
    logreg = LogisticRegression(solver="liblinear", max_iter=200)
    # Create a Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth = 5)
    # Create a Random Forest Classifier
    rf = RandomForestClassifier(min_samples_split = 4, max_depth = 8)
    # Create a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors = 3)
    # Create a Naive Bayes Classifier
    nb = GaussianNB()
    # Create an SVM Classifier
    svm = SVC(random_state=42, C=10, kernel='rbf', probability=True)
    # Create an MLP Classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(12,),
        activation='relu',
        solver='lbfgs',
        alpha=0.0001,
        max_iter=10000,
        early_stopping=True,
        random_state=42
    )
    # Fit the LDA to the data
    lda.fit(X_train, y_train)
    # Fit the Logistic Regression to the data
    logreg.fit(X_train, y_train)
    # Fit the Decision Tree to the data
    dt.fit(X_train, y_train)
    # Fit the Random Forest to the data
    rf.fit(X_train, y_train)
    # Fit the KNN to the data
    knn.fit(X_train, y_train)
    # Fit the Naive Bayes to the data
    nb.fit(X_train, y_train)
    # Fit the SVM to the data
    svm.fit(X_train, y_train)
    # Fit the MLP to the data
    mlp.fit(X_train, y_train)

    # Predictions and probabilities for the test set
    y_pred_test_lda = lda.predict(X_test)
    y_prob_test_lda = lda.predict_proba(X_test)[:, 1]
    # Predictions and probabilities for the test set
    y_pred_test_logreg = logreg.predict(X_test)
    y_prob_test_logreg = logreg.predict_proba(X_test)[:, 1]
    # Decision Tree | Predictions and probabilities for the test set
    y_pred_test_dt = dt.predict(X_test)
    y_prob_test_dt = dt.predict_proba(X_test)[:,1]
    # Random Forest | Predictions and probabilities for the test set
    y_pred_test_rf = rf.predict(X_test)
    y_prob_test_rf = rf.predict_proba(X_test)[:,1]
    # KNN | Predictions and probabilities for the test set
    y_pred_test_knn = knn.predict(X_test)
    y_prob_test_knn = knn.predict_proba(X_test)[:, 1]
    # Naive Bayes | Predictions and probabilities for the test set
    y_pred_test_nb = nb.predict(X_test)
    y_prob_test_nb = nb.predict_proba(X_test)[:, 1]
    # SVM | Predictions and probabilities for the test set
    y_pred_test_svm = svm.predict(X_test)
    y_prob_test_svm = svm.predict_proba(X_test)[:, 1]
    # MLP | Predictions and probabilities for the test set
    y_pred_test_mlp = mlp.predict(X_test)
    y_prob_test_mlp = mlp.predict_proba(X_test)[:, 1]

    # LDA | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_lda = lda.predict(X_train)
    y_prob_train_lda = lda.predict_proba(X_train)[:, 1]
    # Logistic Regression | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_logreg = logreg.predict(X_train)
    y_prob_train_logreg = logreg.predict_proba(X_train)[:, 1]
    # Decision Tree | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_dt = dt.predict(X_train)
    y_prob_train_dt = dt.predict_proba(X_train)[:, 1]
    # Random Forest | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_rf = rf.predict(X_train)
    y_prob_train_rf = rf.predict_proba(X_train)[:, 1]
    # KNN | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_knn = knn.predict(X_train)
    y_prob_train_knn = knn.predict_proba(X_train)[:, 1]
    # Naive Bayes | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_nb = nb.predict(X_train)
    y_prob_train_nb = nb.predict_proba(X_train)[:, 1]
    # SVM | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_svm = svm.predict(X_train)
    y_prob_train_svm = svm.predict_proba(X_train)[:, 1]
    # MLP | Predictions and probabilities for the train set (for evaluating overfitting)
    y_pred_train_mlp = mlp.predict(X_train)
    y_prob_train_mlp = mlp.predict_proba(X_train)[:, 1]

    # Calculate confusion matrices
    cm_train_lda = confusion_matrix(y_train, y_pred_train_lda)
    cm_test_lda = confusion_matrix(y_test, y_pred_test_lda)
    cm_train_logreg = confusion_matrix(y_train, y_pred_train_logreg)
    cm_test_logreg = confusion_matrix(y_test, y_pred_test_logreg)
    cm_train_dt = confusion_matrix(y_train, y_pred_train_dt)
    cm_test_dt = confusion_matrix(y_test, y_pred_test_dt)
    cm_train_rf = confusion_matrix(y_train, y_pred_train_rf)
    cm_test_rf = confusion_matrix(y_test, y_pred_test_rf)
    cm_train_knn = confusion_matrix(y_train, y_pred_train_knn)
    cm_test_knn = confusion_matrix(y_test, y_pred_test_knn)
    cm_train_nb = confusion_matrix(y_train, y_pred_train_nb)
    cm_test_nb = confusion_matrix(y_test, y_pred_test_nb)
    cm_train_svm = confusion_matrix(y_train, y_pred_train_svm)
    cm_test_svm = confusion_matrix(y_test, y_pred_test_svm)
    cm_train_mlp = confusion_matrix(y_train, y_pred_train_mlp)
    cm_test_mlp = confusion_matrix(y_test, y_pred_test_mlp)

    # LDA | Calculate performance metrics for the test set
    accuracy_test_lda = accuracy_score(y_test,y_pred_test_lda)
    precision_test_lda = precision_score(y_test,y_pred_test_lda, average='macro')
    recall_test_lda = recall_score(y_test,y_pred_test_lda, average='macro')
    f1_test_lda = f1_score(y_test,y_pred_test_lda, average='macro')
    roc_auc_test_lda = roc_auc_score(y_test, y_prob_test_lda, average='macro')

    # LDA | Calculate performance metrics for the train set
    accuracy_train_lda = accuracy_score(y_train, y_pred_train_lda)
    precision_train_lda = precision_score(y_train, y_pred_train_lda, average='macro')
    recall_train_lda = recall_score(y_train, y_pred_train_lda, average='macro')
    f1_train_lda = f1_score(y_train, y_pred_train_lda, average='macro')
    roc_auc_train_lda = roc_auc_score(y_train, y_prob_train_lda, average='macro')

    # Logistic Regression | Calculate performance metrics for the test set
    accuracy_test_logreg = accuracy_score(y_test,y_pred_test_logreg)
    precision_test_logreg = precision_score(y_test,y_pred_test_logreg, average='macro')
    recall_test_logreg = recall_score(y_test,y_pred_test_logreg, average='macro')
    f1_test_logreg = f1_score(y_test,y_pred_test_logreg, average='macro')
    roc_auc_test_logreg = roc_auc_score(y_test, y_prob_test_logreg, average='macro')

    # Logistic Regression | Calculate performance metrics for the train set
    accuracy_train_logreg = accuracy_score(y_train, y_pred_train_logreg)
    precision_train_logreg = precision_score(y_train, y_pred_train_logreg, average='macro')
    recall_train_logreg = recall_score(y_train, y_pred_train_logreg, average='macro')
    f1_train_logreg = f1_score(y_train, y_pred_train_logreg, average='macro')
    roc_auc_train_logreg = roc_auc_score(y_train, y_prob_train_logreg, average='macro')

    # Decision Tree | Calculate performance metrics for the test set
    accuracy_test_dt = accuracy_score(y_test, y_pred_test_dt)
    precision_test_dt = precision_score(y_test, y_pred_test_dt, average='macro')
    recall_test_dt = recall_score(y_test, y_pred_test_dt, average='macro')
    f1_test_dt = f1_score(y_test, y_pred_test_dt, average='macro')
    roc_auc_test_dt = roc_auc_score(y_test, y_prob_test_dt, average='macro')

    # Decision Tree | Calculate performance metrics for the train set
    accuracy_train_dt = accuracy_score(y_train, y_pred_train_dt)
    precision_train_dt = precision_score(y_train, y_pred_train_dt, average='macro')
    recall_train_dt = recall_score(y_train, y_pred_train_dt, average='macro')
    f1_train_dt = f1_score(y_train, y_pred_train_dt, average='macro')
    roc_auc_train_dt = roc_auc_score(y_train, y_prob_train_dt, average='macro')

    # Random Forest | Calculate performance metrics for the test set
    accuracy_test_rf = accuracy_score(y_test, y_pred_test_rf)
    precision_test_rf = precision_score(y_test, y_pred_test_rf, average='macro')
    recall_test_rf = recall_score(y_test, y_pred_test_rf, average='macro')
    f1_test_rf = f1_score(y_test, y_pred_test_rf, average='macro')
    roc_auc_test_rf = roc_auc_score(y_test, y_prob_test_rf, average='macro')

    # Random Forest | Calculate performance metrics for the train set
    accuracy_train_rf = accuracy_score(y_train, y_pred_train_rf)
    precision_train_rf = precision_score(y_train, y_pred_train_rf, average='macro')
    recall_train_rf = recall_score(y_train, y_pred_train_rf, average='macro')
    f1_train_rf = f1_score(y_train, y_pred_train_rf, average='macro')
    roc_auc_train_rf = roc_auc_score(y_train, y_prob_train_rf, average='macro')

    # KNN | Calculate performance metrics for the test set
    accuracy_test_knn = accuracy_score(y_test, y_pred_test_knn)
    precision_test_knn = precision_score(y_test, y_pred_test_knn, average='macro')
    recall_test_knn = recall_score(y_test, y_pred_test_knn, average='macro')
    f1_test_knn = f1_score(y_test, y_pred_test_knn, average='macro')
    roc_auc_test_knn = roc_auc_score(y_test, y_prob_test_knn, average='macro')

    # KNN | Calculate performance metrics for the train set
    accuracy_train_knn = accuracy_score(y_train, y_pred_train_knn)
    precision_train_knn = precision_score(y_train, y_pred_train_knn, average='macro')
    recall_train_knn = recall_score(y_train, y_pred_train_knn, average='macro')
    f1_train_knn = f1_score(y_train, y_pred_train_knn, average='macro')
    roc_auc_train_knn = roc_auc_score(y_train, y_prob_train_knn, average='macro')

    # Naive Bayes | Calculate performance metrics for the test set
    accuracy_test_nb = accuracy_score(y_test, y_pred_test_nb)
    precision_test_nb = precision_score(y_test, y_pred_test_nb, average='macro')
    recall_test_nb = recall_score(y_test, y_pred_test_nb, average='macro')
    f1_test_nb = f1_score(y_test, y_pred_test_nb, average='macro')
    roc_auc_test_nb = roc_auc_score(y_test, y_prob_test_nb, average='macro')

    # Naive Bayes | Calculate performance metrics for the train set
    accuracy_train_nb = accuracy_score(y_train, y_pred_train_nb)
    precision_train_nb = precision_score(y_train, y_pred_train_nb, average='macro')
    recall_train_nb = recall_score(y_train, y_pred_train_nb, average='macro')
    f1_train_nb = f1_score(y_train, y_pred_train_nb, average='macro')
    roc_auc_train_nb = roc_auc_score(y_train, y_prob_train_nb, average='macro')

    # SVM | Calculate performance metrics for the test set
    accuracy_test_svm = accuracy_score(y_test, y_pred_test_svm)
    precision_test_svm = precision_score(y_test, y_pred_test_svm, average='macro')
    recall_test_svm = recall_score(y_test, y_pred_test_svm, average='macro')
    f1_test_svm = f1_score(y_test, y_pred_test_svm, average='macro')
    roc_auc_test_svm = roc_auc_score(y_test, y_prob_test_svm, average='macro')
    # SVM | Calculate performance metrics for the train set
    accuracy_train_svm = accuracy_score(y_train, y_pred_train_svm)
    precision_train_svm = precision_score(y_train, y_pred_train_svm, average='macro')
    recall_train_svm = recall_score(y_train, y_pred_train_svm, average='macro')
    f1_train_svm = f1_score(y_train, y_pred_train_svm, average='macro')
    roc_auc_train_svm = roc_auc_score(y_train, y_prob_train_svm, average='macro')

    # MLP | Calculate performance metrics for the test set
    accuracy_test_mlp = accuracy_score(y_test, y_pred_test_mlp)
    precision_test_mlp = precision_score(y_test, y_pred_test_mlp, average='macro')
    recall_test_mlp = recall_score(y_test, y_pred_test_mlp, average='macro')
    f1_test_mlp = f1_score(y_test, y_pred_test_mlp, average='macro')
    roc_auc_test_mlp = roc_auc_score(y_test, y_prob_test_mlp, average='macro')
    # MLP | Calculate performance metrics for the train set
    accuracy_train_mlp = accuracy_score(y_train, y_pred_train_mlp)
    precision_train_mlp = precision_score(y_train, y_pred_train_mlp, average='macro')
    recall_train_mlp = recall_score(y_train, y_pred_train_mlp, average='macro')
    f1_train_mlp = f1_score(y_train, y_pred_train_mlp, average='macro')
    roc_auc_train_mlp = roc_auc_score(y_train, y_prob_train_mlp, average='macro')

    # ==========================

    # Recalculate the counts after moving the companies
    train_healthy_count = train_set[train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 0.0].shape[0]
    train_bankrupt_count = train_set[train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 1.0].shape[0]
    test_healthy_count = test_set[test_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 0.0].shape[0]
    test_bankrupt_count = test_set[test_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 1.0].shape[0]

    # Print the results
    print(f"==== Fold {fold + 1} (After Balancing) ====")
    print(f"Train: {train_bankrupt_count} bankrupt, {train_healthy_count} healthy, Total: {len(train_set)}")
    print(f"Test: {test_bankrupt_count} bankrupt, {test_healthy_count} healthy, Total: {len(test_set)}")

    metrics_lda = {
    'train': {'accuracy': accuracy_train_lda, 'precision': precision_train_lda,
              'recall': recall_train_lda, 'f1': f1_train_lda, 'roc_auc': roc_auc_train_lda},
    'test': {'accuracy': accuracy_test_lda, 'precision': precision_test_lda,
             'recall': recall_test_lda, 'f1': f1_test_lda, 'roc_auc': roc_auc_test_lda}
    }

    metrics_logreg = {
    'train': {'accuracy': accuracy_train_logreg, 'precision': precision_train_logreg,
              'recall': recall_train_logreg, 'f1': f1_train_logreg, 'roc_auc': roc_auc_train_logreg},
    'test': {'accuracy': accuracy_test_logreg, 'precision': precision_test_logreg,
             'recall': recall_test_logreg, 'f1': f1_test_logreg, 'roc_auc': roc_auc_test_logreg}
    }

    metrics_dt = {
    'train': {'accuracy': accuracy_train_dt, 'precision': precision_train_dt,
              'recall': recall_train_dt, 'f1': f1_train_dt, 'roc_auc': roc_auc_train_dt},
    'test': {'accuracy': accuracy_test_dt, 'precision': precision_test_dt,
             'recall': recall_test_dt, 'f1': f1_test_dt, 'roc_auc': roc_auc_test_dt}
    }

    metrics_rf = {
    'train': {'accuracy': accuracy_train_rf, 'precision': precision_train_rf,
              'recall': recall_train_rf, 'f1': f1_train_rf, 'roc_auc': roc_auc_train_rf},
    'test': {'accuracy': accuracy_test_rf, 'precision': precision_test_rf,
             'recall': recall_test_rf, 'f1': f1_test_rf, 'roc_auc': roc_auc_test_rf}
    }

    metrics_knn = {
    'train': {'accuracy': accuracy_train_knn, 'precision': precision_train_knn,
              'recall': recall_train_knn, 'f1': f1_train_knn, 'roc_auc': roc_auc_train_knn},
    'test': {'accuracy': accuracy_test_knn, 'precision': precision_test_knn,
             'recall': recall_test_knn, 'f1': f1_test_knn, 'roc_auc': roc_auc_test_knn}
    }

    metrics_nb = {
    'train': {'accuracy': accuracy_train_nb, 'precision': precision_train_nb,
              'recall': recall_train_nb, 'f1': f1_train_nb, 'roc_auc': roc_auc_train_nb},
    'test': {'accuracy': accuracy_test_nb, 'precision': precision_test_nb,
             'recall': recall_test_nb, 'f1': f1_test_nb, 'roc_auc': roc_auc_test_nb}
    }

    metrics_svm = {
    'train': {'accuracy': accuracy_train_svm, 'precision': precision_train_svm,
              'recall': recall_train_svm, 'f1': f1_train_svm, 'roc_auc': roc_auc_train_svm},
    'test': {'accuracy': accuracy_test_svm, 'precision': precision_test_svm,
             'recall': recall_test_svm, 'f1': f1_test_svm, 'roc_auc': roc_auc_test_svm}
    }

    metrics_mlp = {
    'train': {'accuracy': accuracy_train_mlp, 'precision': precision_train_mlp,
              'recall': recall_train_mlp, 'f1': f1_train_mlp, 'roc_auc': roc_auc_train_mlp},
    'test': {'accuracy': accuracy_test_mlp, 'precision': precision_test_mlp,
             'recall': recall_test_mlp, 'f1': f1_test_mlp, 'roc_auc': roc_auc_test_mlp}
    }

    plot_confusion_matrices(lda,logreg,cm_train_lda, cm_test_lda, metrics_lda,
                        cm_train_logreg, cm_test_logreg, metrics_logreg, fold)
    plot_confusion_matrices(dt, rf, cm_train_dt, cm_test_dt, metrics_dt,
                        cm_train_rf, cm_test_rf, metrics_rf, fold)
    plot_confusion_matrices(knn,nb,cm_train_knn, cm_test_knn, metrics_knn,
                        cm_train_nb, cm_test_nb, metrics_nb, fold)
    plot_confusion_matrices(svm, mlp, cm_train_svm, cm_test_svm, metrics_svm,
                        cm_train_mlp, cm_test_mlp, metrics_mlp, fold)


    # For each classifier and dataset (train/test), record the information
    for clf, clf_name, clf_name_actual in [(lda, "lda", "Linear Discrimination Analysis"), (logreg, "logreg", "Logistic Regression"),
                                           (dt, "dt", "Decision Tree"), (rf, "rf", "Random Forest"),
                                           (knn, "knn", "KNN"), (nb, "nb", "Naive Bayes"),
                                           (svm, "svm", "Support Vector Machine"), (mlp, "mlp", "Multilayer Perceptron")]:
        # Define the predictions and probabilities based on the classifier name
        if clf_name == "lda":
            y_pred_train = y_pred_train_lda
            y_prob_train = y_prob_train_lda
            y_pred_test = y_pred_test_lda
            y_prob_test = y_prob_test_lda
        elif clf_name == "logreg":
            y_pred_train = y_pred_train_logreg
            y_prob_train = y_prob_train_logreg
            y_pred_test = y_pred_test_logreg
            y_prob_test = y_prob_test_logreg
        elif clf_name == "dt":
            y_pred_train = y_pred_train_dt
            y_prob_train = y_prob_train_dt
            y_pred_test = y_pred_test_dt
            y_prob_test = y_prob_test_dt
        elif clf_name == "rf":
            y_pred_train = y_pred_train_rf
            y_prob_train = y_prob_train_rf
            y_pred_test = y_pred_test_rf
            y_prob_test = y_prob_test_rf
        elif clf_name == "knn":
            y_pred_train = y_pred_train_knn
            y_prob_train = y_prob_train_knn
            y_pred_test = y_pred_test_knn
            y_prob_test = y_prob_test_knn
        elif clf_name == "nb":
            y_pred_train = y_pred_train_nb
            y_prob_train = y_prob_train_nb
            y_pred_test = y_pred_test_nb
            y_prob_test = y_prob_test_nb
        elif clf_name == "svm":
            y_pred_train = y_pred_train_svm
            y_prob_train = y_prob_train_svm
            y_pred_test = y_pred_test_svm
            y_prob_test = y_prob_test_svm
        elif clf_name == "mlp":
            y_pred_train = y_pred_train_mlp
            y_prob_train = y_prob_train_mlp
            y_pred_test = y_pred_test_mlp
            y_prob_test = y_prob_test_mlp
        else:
            raise ValueError(f"Classifier {clf_name_actual} not recognized.")

        for dataset, X, y, set_name in [
            (train_set, X_train, y_train, "Training"),
            (test_set, X_test, y_test, "Test"),
            ]:

            # Generate predictions and probabilities based on the classifier
            y_pred = clf.predict(X)
            y_prob = clf.predict_proba(X)[:, 1]  # Assuming that the positive class is at index 1

            # Calculate confusion matrix and ROC AUC
            cm = confusion_matrix(y, y_pred)
            roc_auc = round(roc_auc_score(y, y_prob), 2)  # Round to 2 decimals

            # Append results to the DataFrame
            new_row = {
                "Classifier Name": clf_name_actual,
                "Dataset": set_name,
                "Balance": "Balanced",
                #if len(train_set[train_set["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 0.0]) > train_healthy_lim else "Unbalanced",
                "Number of Training Samples": len(train_set) if set_name=="Training" else len(test_set) ,
                "Number of Bankrupt Companies": dataset[dataset["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] == 1.0].shape[0],
                "TP": cm[1, 1],
                "TN": cm[0, 0],
                "FP": cm[0, 1],
                "FN": cm[1, 0],
                "ROC-AUC": roc_auc
            }
            # Append new row using concat
            balanced_results_df = pd.concat([balanced_results_df, pd.DataFrame([new_row])], ignore_index=True)

# Print the DataFrame using tabulate after the folds
print(tabulate(balanced_results_df, headers='keys', tablefmt='psql', showindex=False))

# Export the DataFrame to a CSV file
balanced_results_df.to_csv('balancedDataOutcomes.csv', index=False)
