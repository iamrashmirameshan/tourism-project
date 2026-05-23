# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import mlflow
import mlflow.sklearn
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    precision_score,
    f1_score
)
import warnings
warnings.filterwarnings("ignore")
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from mlflow.models.signature import infer_signature

api = HfApi()

Xtrain_path = "hf://datasets/rashmipr/tourism-project/Xtrain.csv"
Xtest_path = "hf://datasets/rashmipr/tourism-project/Xtest.csv"
ytrain_path = "hf://datasets/rashmipr/tourism-project/ytrain.csv"
ytest_path = "hf://datasets/rashmipr/tourism-project/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Define numeric and categorical features
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch',
    'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched','MaritalStatus','Designation'
]

# Set the clas weight to handle class imbalance
class_weight = (
    ytrain['ProdTaken'].value_counts()[0] /
    ytrain['ProdTaken'].value_counts()[1]
)
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],    # number of tree to build
    'xgbclassifier__max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():

    # ==========================
    # Hyperparameter tuning
    # ==========================

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        n_jobs=-1
    )

    grid_search.fit(
        Xtrain,
        ytrain.values.ravel()
    )

    # ==========================
    # Best Model
    # ==========================

    best_model = grid_search.best_estimator_

    print("\nBest Parameters:\n")
    print(grid_search.best_params_)

    # Log parameters
    mlflow.log_params(
        grid_search.best_params_
    )

    # ==========================
    # Prediction Threshold
    # ==========================

    classification_threshold = 0.45

    # ==========================
    # Training Predictions
    # ==========================

    y_pred_train_proba = (
        best_model.predict_proba(Xtrain)[:, 1]
    )

    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)

    # ==========================
    # Test Predictions
    # ==========================

    y_pred_test_proba = (
        best_model.predict_proba(Xtest)[:, 1]
    )

    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)

    # ==========================
    # Classification Reports
    # ==========================

    print("\nTraining Classification Report:\n")

    print(
        classification_report(
            ytrain,
            y_pred_train
        )
    )

    print("\nTesting Classification Report:\n")

    print(
        classification_report(
            ytest,
            y_pred_test
        )
    )

    # ==========================
    # TRAIN METRICS
    # ==========================

    train_accuracy = accuracy_score(
        ytrain,
        y_pred_train
    )

    train_precision = precision_score(
        ytrain,
        y_pred_train
    )

    train_recall = recall_score(
        ytrain,
        y_pred_train
    )

    train_f1 = f1_score(
        ytrain,
        y_pred_train
    )

    # ==========================
    # TEST METRICS
    # ==========================

    test_accuracy = accuracy_score(
        ytest,
        y_pred_test
    )

    test_precision = precision_score(
        ytest,
        y_pred_test
    )

    test_recall = recall_score(
        ytest,
        y_pred_test
    )

    test_f1 = f1_score(
        ytest,
        y_pred_test
    )

    # ==========================
    # LOG TRAIN METRICS
    # ==========================

    mlflow.log_metric(
        "train_accuracy",
        train_accuracy
    )

    mlflow.log_metric(
        "train_precision",
        train_precision
    )

    mlflow.log_metric(
        "train_recall",
        train_recall
    )

    mlflow.log_metric(
        "train_f1_score",
        train_f1
    )

    # ==========================
    # LOG TEST METRICS
    # ==========================

    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )

    mlflow.log_metric(
        "test_precision",
        test_precision
    )

    mlflow.log_metric(
        "test_recall",
        test_recall
    )

    mlflow.log_metric(
        "test_f1_score",
        test_f1
    )

    print("\nMetrics logged to MLflow.")

    # ==========================
    # SAVE MODEL
    # ==========================

    joblib.dump(
        best_model,
        "tourism_prediction.joblib"
    )

    print("\nModel saved locally.")

    # ==========================
    # LOG MODEL TO MLflow
    # ==========================

    # Infer model signature
    signature = infer_signature(
    Xtrain,
    best_model.predict(Xtrain)
    )

    # Log model to MLflow
    mlflow.sklearn.log_model(
    sk_model=best_model,
    name="xgboost_model",
    signature=signature,
    input_example=Xtrain.iloc[:5]
    )

    print("\nModel logged to MLflow.")

    # ==========================
    # HUGGING FACE MODEL REGISTRY
    # ==========================

    repo_id = "rashmipr/tourism-prediction-model"

    repo_type = "model"

    api = HfApi(
        token=os.getenv("MY_MLOps_Token")
    )

    try:

        api.repo_info(
            repo_id=repo_id,
            repo_type=repo_type
        )

        print(
            f"Model Space '{repo_id}' already exists. Using it."
        )

    except RepositoryNotFoundError:

        print(
            f"Model Space '{repo_id}' not found. Creating new space..."
        )

        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=False
        )

        print(
            f"Model Space '{repo_id}' created."
        )

    # Upload model
    api.upload_file(
        path_or_fileobj="tourism_prediction.joblib",

        path_in_repo="tourism_prediction.joblib",

        repo_id=repo_id,

        repo_type=repo_type,
    )

    print(
        "\nModel uploaded to Hugging Face successfully."
    )
