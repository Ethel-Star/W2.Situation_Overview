import pandas as pd
import numpy as np
from scipy import stats

class DataUtils:
    def __init__(self, df):
        self.df = df

    def check_missing_values(self):
        """Checks and summarizes missing values in the DataFrame."""
        missing_summary = self.df.isnull().sum()
        missing_percentage = (missing_summary / len(self.df)) * 100
        missing_table = pd.DataFrame({
            'Missing Values': missing_summary,
            'Percentage': missing_percentage.map("{:.1f}%".format),
            'Dtype': self.df.dtypes
        })
        missing_table = missing_table[missing_table['Missing Values'] > 0]
        missing_table = missing_table.sort_values(by='Percentage', ascending=False)
        
        print(f"Total columns with missing values: {missing_table.shape[0]}")
        print(f"Top 5 columns with the most missing values:\n{missing_table.head()}")
        
        return missing_table

    def handle_missing_values(self):
        """Handles missing values by filling them or dropping columns with excessive missing values."""
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                if self.df[col].skew() > 1:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
            elif self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        # Drop columns with more than 50% missing values
        threshold = 0.5 * len(self.df)
        high_missing_cols = self.df.isnull().sum()[self.df.isnull().sum() > threshold].index
        self.df = self.df.drop(columns=high_missing_cols)
        
        print("Missing values handled and high missing value columns dropped.")
        return self.df

    def detect_outliers(self, z_thresh=3):
        """Detects outliers based on Z-score threshold for numeric columns."""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        outlier_dict = {}
        for column in numeric_columns:
            column_data = self.df[column]
            z_scores = np.abs(stats.zscore(column_data, nan_policy='omit'))
            z_scores = pd.Series(z_scores, index=self.df.index)
            outliers = self.df.index[z_scores > z_thresh]
            outlier_dict[column] = outliers

        print("Outlier detection complete.")
        return outlier_dict

    def fix_outliers(self, method='median', quantile_val=0.95):
        """Fix outliers in the DataFrame for numeric columns."""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if method == 'median':
                self.df[column] = np.where(
                    self.df[column] > self.df[column].quantile(quantile_val), 
                    self.df[column].median(), 
                    self.df[column]
                )
            elif method == 'mean':
                self.df[column] = np.where(
                    self.df[column] > self.df[column].quantile(quantile_val), 
                    self.df[column].mean(), 
                    self.df[column]
                )
        return self.df

    def remove_outliers(self, z_thresh=3):
        """Remove outliers based on Z-score threshold for numeric columns."""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            z_scores = np.abs(stats.zscore(self.df[column].dropna()))
            self.df[column] = np.where(z_scores > z_thresh, np.nan, self.df[column])
        
        self.df.dropna(inplace=True)
        
        print("Outliers removed.")
        return self.df

    def bytes_to_megabytes(self, bytes_value):
        """Converts bytes to megabytes."""
        return bytes_value / (1024 * 1024)

    def convert_bytes_to_megabytes(self):
        """Converts columns with 'Bytes' in their names to megabytes."""
        byte_columns = [col for col in self.df.columns if '(Bytes)' in col]
        
        for column in byte_columns:
            new_column_name = column.replace('(Bytes)', '(MB)')
            self.df[new_column_name] = self.df[column].apply(self.bytes_to_megabytes)
        
        print("Converted byte columns to megabytes.")
        return self.df
