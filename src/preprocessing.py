from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataset."""
    df = df.copy()

    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1e-5)
    df['EMPLOYED_AGE_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1e-5)

    df['EXT_SOURCES_AVG'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['DOCUMENT_COUNT'] = df.filter(like='FLAG_DOCUMENT_').sum(axis=1)

    return df

def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data for modeling.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarray
        val : np.ndarray
        test : np.ndarray
    """

    # 1. Add engineered features
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    test_df = add_features(test_df)

    # 2. Correct outliers in 'DAYS_EMPLOYED'
    for df in [train_df, val_df, test_df]:
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # 3. Encode categorical variables
    cat_cols = train_df.select_dtypes(include='object').columns
    binary_cols = [col for col in cat_cols if train_df[col].nunique() == 2]
    multi_cols = [col for col in cat_cols if train_df[col].nunique() > 2]

    # OrdinalEncoder for binary columns
    if binary_cols:
        ordinal_encoder = OrdinalEncoder()
        train_df[binary_cols] = ordinal_encoder.fit_transform(train_df[binary_cols])
        val_df[binary_cols] = ordinal_encoder.transform(val_df[binary_cols])
        test_df[binary_cols] = ordinal_encoder.transform(test_df[binary_cols])

    # OneHotEncoder for multi-category columns
    if multi_cols:
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        train_ohe = onehot_encoder.fit_transform(train_df[multi_cols])
        val_ohe = onehot_encoder.transform(val_df[multi_cols])
        test_ohe = onehot_encoder.transform(test_df[multi_cols])

        ohe_columns = onehot_encoder.get_feature_names_out(multi_cols)
        train_ohe_df = pd.DataFrame(train_ohe, columns=ohe_columns, index=train_df.index)
        val_ohe_df = pd.DataFrame(val_ohe, columns=ohe_columns, index=val_df.index)
        test_ohe_df = pd.DataFrame(test_ohe, columns=ohe_columns, index=test_df.index)

        train_df = train_df.drop(columns=multi_cols).join(train_ohe_df)
        val_df = val_df.drop(columns=multi_cols).join(val_ohe_df)
        test_df = test_df.drop(columns=multi_cols).join(test_ohe_df)

    # 4. Impute missing values
    imputer = SimpleImputer(strategy='median')
    train_df[:] = imputer.fit_transform(train_df)
    val_df[:] = imputer.transform(val_df)
    test_df[:] = imputer.transform(test_df)

    # 5. Scale features
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    return train_scaled, val_scaled, test_scaled

# def preprocess_data(
#     train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Pre processes data for modeling. Receives train, val and test dataframes
#     and returns numpy ndarrays of cleaned up dataframes with feature engineering
#     already performed.

#     Arguments:
#         train_df : pd.DataFrame
#         val_df : pd.DataFrame
#         test_df : pd.DataFrame

#     Returns:
#         train : np.ndarrary
#         val : np.ndarrary
#         test : np.ndarrary
#     """
#     # Print shape of input data
#     print("Input train data shape: ", train_df.shape)
#     print("Input val data shape: ", val_df.shape)
#     print("Input test data shape: ", test_df.shape, "\n")

#     # Make a copy of the dataframes
#     working_train_df = train_df.copy()
#     working_val_df = val_df.copy()
#     working_test_df = test_df.copy()

#     # 1. Correct outliers/anomalous values in numerical
#     # columns (`DAYS_EMPLOYED` column).
#     working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
#     working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
#     working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

#     # 2. TODO Encode string categorical features (dytpe `object`):
#     #     - If the feature has 2 categories encode using binary encoding,
#     #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
#     #       from the dataset should have 2 categories.
#     #     - If it has more than 2 categories, use one-hot encoding, please use
#     #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
#     #       from the dataset should have more than 2 categories.
#     # Take into account that:
#     #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
#     #     working_test_df).
#     #   - In order to prevent overfitting and avoid Data Leakage you must use only
#     #     working_train_df DataFrame to fit the OrdinalEncoder and
#     #     OneHotEncoder classes, then use the fitted models to transform all the
#     #     datasets.


#     # 3. TODO Impute values for all columns with missing data or, just all the columns.
#     # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
#     # Again, take into account that:
#     #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
#     #     working_test_df).
#     #   - In order to prevent overfitting and avoid Data Leakage you must use only
#     #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
#     #     model to transform all the datasets.


#     # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
#     # Please use sklearn.preprocessing.MinMaxScaler().
#     # Again, take into account that:
#     #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
#     #     working_test_df).
#     #   - In order to prevent overfitting and avoid Data Leakage you must use only
#     #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
#     #     model to transform all the datasets.


#     return None

