# fhmv/features/feature_engineering_module.py

import pandas as pd
import numpy as np

class FeatureEngineeringModule:
    """
    FHMV Feature Engineering Module.

    This module is responsible for standardizing (z-score scaling) the 10 core
    features received from the DataIngestionPreprocessingModule.
    Scaling parameters (mean and standard deviation) are learned exclusively
    from a training dataset and then applied consistently to prevent
    lookahead bias, as per FHMV_2.docx (Compendium, Section IV.D.2)[cite: 372, 416].
    """

    def __init__(self, config: dict = None):
        """
        Initializes the FeatureEngineeringModule.

        Args:
            config (dict, optional): Configuration dictionary. May contain
                                     parameters like "scaling_method" if more
                                     methods are supported in the future.
                                     Defaults to None.
        """
        self.scaling_method = config.get("scaling_method", "standardize") if config else "standardize"
        if self.scaling_method != "standardize":
            # For now, only 'standardize' is fully detailed.
            # Other methods like 'normalize' (min-max) could be added.
            raise NotImplementedError(f"Scaling method '{self.scaling_method}' is not yet implemented. "
                                      "Only 'standardize' is supported.")
        
        self.feature_means_ = None
        self.feature_stds_ = None
        self.fitted_columns_ = None
        print("FeatureEngineeringModule Initialized with config.")

    def fit(self, features_df_train: pd.DataFrame):
        """
        Learns the scaling parameters (mean and standard deviation) for each
        feature column from the training data for standardization.

        Args:
            features_df_train (pd.DataFrame): DataFrame containing the core
                                              features for the training period.
                                              NaNs will be ignored in mean/std calculation.
        
        Raises:
            TypeError: If features_df_train is not a Pandas DataFrame.
            ValueError: If features_df_train is empty or if mean/std cannot be computed
                        for some columns (e.g., all-NaN columns).
        """
        if not isinstance(features_df_train, pd.DataFrame):
            raise TypeError("features_df_train must be a Pandas DataFrame.")
        if features_df_train.empty:
            raise ValueError("Input DataFrame features_df_train is empty.")

        self.feature_means_ = features_df_train.mean(axis=0, skipna=True)
        self.feature_stds_ = features_df_train.std(axis=0, skipna=True)
        self.fitted_columns_ = features_df_train.columns

        # Handle cases where standard deviation is zero (e.g., constant feature)
        # to avoid division by zero during transformation. Transformed feature will be zero.
        if self.feature_stds_ is not None:
            zero_std_mask = (self.feature_stds_ == 0) | self.feature_stds_.isna()
            self.feature_stds_[zero_std_mask] = 1.0 # Avoid division by zero

        if self.feature_means_.isnull().any() or self.feature_stds_.isnull().any():
            # This check is now partially handled by setting std=1 for zero/NaN std columns.
            # However, if a mean is NaN (all-NaN column), it's still an issue.
            nan_cols_mean = self.feature_means_[self.feature_means_.isnull()].index.tolist()
            if nan_cols_mean:
                raise ValueError(f"Mean could not be computed for all columns "
                                 f"(likely due to all-NaN columns): {nan_cols_mean}. "
                                 f"Please ensure training data has valid values.")
        print("FeatureEngineeringModule fitted.")
                             
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standardization to the input features DataFrame using the
        parameters learned during the fit method.

        Args:
            features_df (pd.DataFrame): DataFrame containing the core features
                                        to be scaled.

        Returns:
            pd.DataFrame: DataFrame with the scaled features.
        
        Raises:
            RuntimeError: If the fit method has not been called yet.
            TypeError: If features_df is not a Pandas DataFrame.
            ValueError: If the columns in features_df do not match fitted columns.
        """
        if self.feature_means_ is None or self.feature_stds_ is None or self.fitted_columns_ is None:
            raise RuntimeError("The 'fit' method must be called before 'transform'.")

        if not isinstance(features_df, pd.DataFrame):
            raise TypeError("features_df must be a Pandas DataFrame.")
        
        if not self.fitted_columns_.equals(features_df.columns):
            missing_cols = self.fitted_columns_.difference(features_df.columns).tolist()
            extra_cols = features_df.columns.difference(self.fitted_columns_).tolist()
            error_msg = ""
            if missing_cols: error_msg += f"Missing columns in input DataFrame: {missing_cols}. "
            if extra_cols: error_msg += f"Extra columns in input DataFrame: {extra_cols}. "
            if error_msg:
                 raise ValueError(f"Column mismatch: {error_msg}Expected columns: {self.fitted_columns_.tolist()}")

        scaled_features_df = (features_df - self.feature_means_) / self.feature_stds_
        return scaled_features_df

    def fit_transform(self, features_df_train: pd.DataFrame) -> pd.DataFrame:
        """
        A convenience method that calls fit and then transform on the same data.

        Args:
            features_df_train (pd.DataFrame): DataFrame containing the core
                                              features for training and scaling.

        Returns:
            pd.DataFrame: DataFrame with the scaled features.
        """
        self.fit(features_df_train)
        return self.transform(features_df_train)