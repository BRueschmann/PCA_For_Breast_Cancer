# Practical 1: Medical Imaging Practical
# GD5302 - Health Data Science Practice
# Helper Functions

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.pipeline
import sklearn.impute
import sklearn.compose
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo 
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay



def split_data(df, target, test_size=0.2, validation_size=0.2, random_state=42):
    """ Split raw data into training, validation, and test sets """
    # Reserve a test set and split remainder into training and validation
    # Stratify splitting on the target y
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    train, val = train_test_split(train_val, test_size=validation_size, random_state=random_state, stratify=train_val[target])
    return train, val, test

def map_y(df):
    """ Mapping for "y" column: "B" becomes 0, "M" becomes 1"""
    mapping = {"B": 0, "M": 1}
    df['y'] = df['y'].map(mapping)
    return df

def plot_histograms(df):
    """Plots histograms for the distribution of values of each column in the DataFrame in a grid layout"""
    n_columns = 3
    n_rows = math.ceil(len(df.columns) / n_columns)
    
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, column in enumerate(df.columns):
        ax = axes[idx]
        sns.histplot(df[column], kde=False, ax=ax, color='purple')
        ax.set_title(column, fontsize=16)
        ax.set_xlabel('', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        
    # Turn off any extra subplots if there are more than needed
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def classify_skewness(df, target='y'):
    """
    Classifies all features except the target based on their skewness.
    Returns 3 lists containing feature names categorised as:
        - no_transformation (skewness below 0.5)
        - mild_transformation (skewness between 0.5 and 1)
        - strong_transformation (skewness over 1)
    """
    no_transformation = []
    mild_transformation = []
    strong_transformation = []
    
    for col in df.select_dtypes(include=['number']).columns:
        if col == target:
            continue  # Skip target variable
        
        skew_value = stats.skew(df[col])
        if abs(skew_value) < 0.5:
            no_transformation.append(col)
        elif 0.5 <= abs(skew_value) < 1.0:
            mild_transformation.append(col)
        else:
            strong_transformation.append(col)
    
    return no_transformation, mild_transformation, strong_transformation


def transform_features(df, columns, method='sqrt'):
    """ Transform heavy-tailed features using either log or sqrt to shrink large values """
    df_transformed = df.copy()
    for column in columns:
        if method == 'log':
            df_transformed[column] = np.log1p(df_transformed[column])
        elif method == 'sqrt':
            df_transformed[column] = np.sqrt(df_transformed[column])
    return df_transformed


def scale_features(train_df, other_df_list, target='y', scaler=MinMaxScaler()):
    """
    Fits scaler on all columns in training set except 'y'. Then also transforms valdation and test set
    Args.: train_df, other_df_list, target, scaler=MinMaxScaler()
    """
    # Determine columns to scale: all except the target
    columns = [col for col in train_df.columns if col != target]
    
    # Fit scaler on the training set and transform the training set.
    train_df_scaled = train_df.copy()
    train_df_scaled[columns] = scaler.fit_transform(train_df[columns])
    
    # Apply the same transformation to the other datasets.
    transformed_list = []
    for df in other_df_list:
        df_temp = df.copy()
        df_temp[columns] = scaler.transform(df_temp[columns])
        transformed_list.append(df_temp)
    
    return train_df_scaled, transformed_list 


def winsorize(train_df, val_df, test_df, target='y', n_std=3):
    """
    Winsorizes all features (except the target) in train, validation, and test sets.
    The winsorization bounds are learned only from the train set and then applied to val and test.
    The lower and upper bound are calculated from n_std * std.
    Returns 3 winsozied dfs.
    """
    # Make copies so original dataframes remain unchanged
    train_w = train_df.copy()
    val_w = val_df.copy()
    test_w = test_df.copy()

    # Identify numerical columns in train set (excluding target)
    num_cols = train_w.select_dtypes(include=['number']).columns
    for col in num_cols:
        if col == target:
            continue

        # Calculate mean and standard deviation from train set
        col_mean = train_w[col].mean()
        col_std = train_w[col].std()
        lower_bound = col_mean - n_std * col_std
        upper_bound = col_mean + n_std * col_std

        # Apply clipping / winsorizing to all 3 sets
        train_w[col] = train_w[col].clip(lower_bound, upper_bound)
        val_w[col] = val_w[col].clip(lower_bound, upper_bound)
        test_w[col] = test_w[col].clip(lower_bound, upper_bound)
    
    return train_w, val_w, test_w


def create_pipeline(model, n_components):
    """
    Creates a pipeline with PCA (with n_components) followed by given model.
    """
    return Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('model', model)
    ])

def run_random_search(pipeline, param_distributions, X, y, cv=5, n_iter=50, scoring=make_scorer(fbeta_score, beta=2)):
    """
    Runs randomised grid search on the provided pipeline.
    """
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        cv=cv,
        n_iter=n_iter,
        scoring=scoring,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X, y)
    return search

def compute_metrics(model, X, y):
    """
    Computes f2, accuracy, precision, and recall for the model on given data
    """
    y_pred = model.predict(X)
    return {
        'f2': fbeta_score(y, y_pred, beta=2),
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred)
    }

def run_dimensionality_experiment(X_train, y_train, X_val, y_val, X_test, y_test, model, param_distributions, dim_range, cv=5, n_iter=50, scoring=make_scorer(fbeta_score, beta=2)):
    """
    For each PCA dimension in dim_range, creates a pipeline (PCA + model),
    runs randomized search on the training set (optimized for f2),
    then evaluates and records metrics on the train, validation, and test sets.
    
    Returns results dictionary with PCA dimension as keys and the corresponding metrics and best parameters.
    """
    results = {}
    
    for n_dim in range(30, 0, -1):
        # Build pipeline with current PCA dimensionality
        pipeline = create_pipeline(model, n_components=n_dim)
        
        # Tune hyperparameters using randomized search (on training set)
        search = run_random_search(pipeline, param_distributions, X_train, y_train, cv=cv, n_iter=n_iter, scoring=scoring)
        best_model = search.best_estimator_
        
        # Compute metrics on training, validation, and test sets
        train_metrics = compute_metrics(best_model, X_train, y_train)
        val_metrics = compute_metrics(best_model, X_val, y_val)
        test_metrics = compute_metrics(best_model, X_test, y_test)
        
        # Save the metrics and best hyperparameters for current PCA dimension
        results[n_dim] = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'best_params': search.best_params_
        }
        
        # Print performance for current PCA dimension
        print(f"PCA components: {n_dim}, Train F2: {train_metrics['f2']:.4f}, Val F2: {val_metrics['f2']:.4f}, Test F2: {test_metrics['f2']:.4f}")
        
    return results

def plot_results(results, ylim=(0, 1)):
    """Plots evaluation scores from a dimensionality experiment"""
     # Get & sort all PCA dimensions.
    dims = sorted(results.keys())
    # list of metrics to plot
    metrics = ['f2', 'accuracy', 'precision', 'recall']
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    # Flatten the grid to 1D array
    axes = axes.flatten()
    
    # Loop over the metrics to create a plot for each one
    for i, metric in enumerate(metrics):
        # Extract the training, validation, & test scores for the current metric from all PCA dimensions
        train_scores = [results[d]['train_metrics'][metric] for d in dims]
        val_scores   = [results[d]['val_metrics'][metric] for d in dims]
        test_scores  = [results[d]['test_metrics'][metric] for d in dims]
        
        # Select current axis to plot
        ax = axes[i]
        # Plot the training, validation, & test scores
        ax.plot(dims, train_scores, marker='o', linestyle='-', color='purple', markersize=8, linewidth=2, label='Train')
        ax.plot(dims, val_scores, marker='s', linestyle='--', color='orange', markersize=8, linewidth=2, label='Validation')
        ax.plot(dims, test_scores, marker='s', linestyle='--', color='red', markersize=8, linewidth=2, label='Test')
        
        # set title of subplot
        ax.set_title(metric.capitalize(), fontsize=20)
        # Label and limit axis, make ticks bigger, have legend
        ax.set_xlabel('Number of PCA Components', fontsize=20)
        ax.set_ylabel('Score', fontsize=20)
        ax.set_ylim(ylim[0], ylim[1])
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(True)
        ax.legend(fontsize=20)
    
    # Adjust layout so subplots don't overlap
    plt.tight_layout()
    plt.show()


# Helper functions to display decision boundary

def get_pca_data(X, n_components=2):
    """Apply PCA, return transformed data and PCA object"""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca

def train_classifiers(X, y):
    """Train four classifiers with best hyperparameters and return a dictionary of models"""
    models = {
        # All with best hyperparameters for 2 PCA components found in last set
        'Logistic Regression': LogisticRegression(C=10),
        'SVM': SVC(probability=True, kernel='linear', gamma='auto', C=100),
        'Random Forest': RandomForestClassifier(n_estimators=100, min_samples_split=5, max_depth=20),
        'MLP': MLPClassifier(max_iter=500, learning_rate_init=0.005,
                             hidden_layer_sizes=(10,), alpha=0.001, activation='relu')
    }
    for model in models.values():
        model.fit(X, y)
    return models

def decision_boundary(X_train, y_train, X_test, y_test):
    """Plot decision boundaries for 4 classifiers on 2D PCA-transformed training data using DecisionBoundaryDisplay,
    and overlay the test data points."""
    # Reduce training data to 2 PCA components and get the PCA transformer
    X_pca_train, pca = get_pca_data(X_train, n_components=2)
    # Transform test data using PCA transformer
    X_pca_test = pca.transform(X_test)
    # Train classifiers on the PCA-transformed training data
    classifiers = train_classifiers(X_pca_train, y_train)
    
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'Logistic Regression': 'red',
        'SVM': 'blue',
        'Random Forest': 'green',
        'MLP': 'purple'
    }
    
    # Plot decision boundaries for each classifier using the training data
    for name, clf in classifiers.items():
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            X_pca_train,
            response_method='predict_proba',
            plot_method='contour',
            levels=[0.5],
            colors=[colors[name]],
            linewidths=2,
            alpha=0.8,
            ax=ax
        )
        # Add line for the legend
        ax.plot([], [], color=colors[name], label=name)
    
    # Plot test data points
    markers = {0: 'o', 1: 's'}
    point_colors = {0: 'lightblue', 1: 'purple'}
    for label in np.unique(y_test):
        idx = (y_test == label)
        ax.scatter(X_pca_test[idx, 0], X_pca_test[idx, 1],
                   marker=markers[label], s=80, edgecolor='k',
                   color=point_colors[label], label=f'Class {label}')
    
    ax.set_xlabel('PCA Component 1', fontsize=22)
    ax.set_ylabel('PCA Component 2', fontsize=22)
    ax.tick_params(axis='both', labelsize=22)
    ax.grid(True)
    ax.legend(fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_cumulative_explained_variance(X_train, threshold=0.95):
    """Plot cumulative explained variance and mark a threshold line"""
    n_components = X_train.shape[1]
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    
    cum_explained = np.cumsum(pca.explained_variance_ratio_)
    components = np.arange(1, n_components + 1)
    
    plt.figure(figsize=(10, 8))
    # Plot the cumulative explained variance curve
    plt.plot(components, cum_explained, 'o-', color='purple', markersize=8, linewidth=2, label='Cumulative Explained Variance')
    
    plt.xlabel('Number of Components', fontsize=22)
    plt.ylabel('Cumulative Explained Variance', fontsize=22)
    # Show every second number on the x-axis because of big font size
    ticks = np.arange(2, n_components + 1, 2)
    plt.xticks(ticks, fontsize=22)
    plt.yticks(fontsize=22)
    plt.axhline(y=threshold, color='b', linestyle='--', linewidth=2, label=f'{int(threshold*100)}% Threshold')
    plt.grid(True)
    plt.legend(fontsize=22)
    plt.show()


def rank_results(results_LR, results_SVM, results_RF, results_MLP, top_n=20):
    """
    Combines results dictionaries from different models
    Creates table ranking all models based on f2 performance on test set
    
    Args: results_LR, results_SVM, results_RF, results_MLP, top_n: Number of models in table 
    Returns a Df with ranked models
    """
    
    # Create empty list to store rows for final table
    rows = []
    
    # Combine results from all models with their corresponding model names.
    for model_name, results in zip(
        ["Logistic Regression", "SVM", "Random Forest", "MLP"],
        [results_LR, results_SVM, results_RF, results_MLP]
    ):
        # Loop over all PCA dimensions in results
        for n_dim, info in results.items():
            # Create row with model name, number of PCA components, best hyperparameters, and test f2 score
            row = {
                "Model": model_name,
                "PCA-Components": n_dim,
                "F2-Score": info['test_metrics']['f2'],
                "Recall": info['test_metrics']['recall'],
                "Accuracy": info['test_metrics']['accuracy'],
                "Precision": info['test_metrics']['precision']
            }
            rows.append(row)
    
    # Create Data Frame from list of rows
    df = pd.DataFrame(rows)
    
    # Sort DataFrame by F2-Score in descending order
    df = df.sort_values(by="F2-Score", ascending=False)
    # Rank number of models as first column
    df.insert(0, "Rank", range(1, len(df) + 1))

    # Return the top_n rows of the sorted DataFrame.
    return df.head(top_n)