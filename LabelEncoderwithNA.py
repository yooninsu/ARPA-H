import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Union, List, Dict

class LabelEncoderWithNA:
    """
    Label Encoder that handles NA values by encoding them as 0
    and other values as integers starting from 1
    """
    def __init__(self):
        self.label_encoders = {}
        self.mapping_dict = {}
    
    def fit_transform(self, df: pd.DataFrame, columns: Union[List[str], str] = None) -> pd.DataFrame:
        """
        Fit and transform specified columns or all object columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe 
        columns : list or str, optional
            Columns to encode. If None, all object columns will be encoded
        
        Returns:
        --------
        pd.DataFrame
            Transformed dataframe
        """
        # Copy dataframe to avoid modifying original
        df_encoded = df.copy()
        
        if columns is None:
            columns = [col for col in df_encoded.columns 
                       if not self._is_numeric_column(df_encoded[col])]
        elif isinstance(columns, str):
            columns = [columns]
            
        for col in columns:
            if col in df_encoded.columns:
                # Skip if column is actually numeric
                if self._is_numeric_column(df_encoded[col]):
                    print(f"Skipping '{col}' - contains numeric values")
                    # Convert to proper numeric type
                    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                    continue
                    
                df_encoded[col] = self._encode_column(df_encoded[col], col)

        
        return df_encoded
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """
        Check if a column contains numeric values (even if dtype is object)
        
        Parameters:
        -----------
        series : pd.Series
            Column to check
            
        Returns:
        --------
        bool
            True if column contains numeric values
        """
        # Remove NA values for checking
        non_na_values = series.dropna()
        
        if len(non_na_values) == 0:
            return False
        
        # Try to convert to numeric
        try:
            pd.to_numeric(non_na_values, errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def _encode_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Encode a single column with NA as 0
        
        Parameters:
        -----------
        series : pd.Series
            Column to encode
        col_name : str
            Column name for tracking encoder
        
        Returns:
        --------
        pd.Series
            Encoded column
        """
        # Create a copy and handle NA values
        series_copy = series.copy()
        
        # Remember which values were NA
        na_mask = series_copy.isna()
        
        # Temporarily fill NA with a placeholder
        series_copy = series_copy.fillna('__NA_PLACEHOLDER__')
        
        # Initialize and fit label encoder
        le = LabelEncoder()
        encoded = le.fit_transform(series_copy)
        
        # Create mapping dictionary
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        
        # Adjust encoding: shift all non-NA values by 1
        if '__NA_PLACEHOLDER__' in mapping:
            # Get the encoded value for our NA placeholder
            na_encoded_value = mapping['__NA_PLACEHOLDER__']
            
            # Shift all values that were >= na_encoded_value
            encoded = np.where(encoded > na_encoded_value, encoded, encoded + 1)
            encoded = np.where(encoded == na_encoded_value, 0, encoded)
            
            # Update mapping
            new_mapping = {}
            for key, value in mapping.items():
                if key == '__NA_PLACEHOLDER__':
                    new_mapping['NA'] = 0
                elif value > na_encoded_value:
                    new_mapping[key] = value
                else:
                    new_mapping[key] = value + 1
            mapping = new_mapping
        else:
            # If no NA values, just shift everything by 1
            encoded = encoded + 1
            mapping = {key: value + 1 for key, value in mapping.items()}
            mapping['NA'] = 0
        
        # Store encoder and mapping
        self.label_encoders[col_name] = le
        self.mapping_dict[col_name] = mapping
        
        # Convert to pandas Series
        result = pd.Series(encoded, index=series.index)
        
        # Ensure NA values are 0
        result[na_mask] = 0
        
        return result
    
    def transform(self, df: pd.DataFrame, columns: Union[List[str], str] = None) -> pd.DataFrame:
        """
        Transform new data using fitted encoders
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list or str, optional
            Columns to encode. If None, use columns from fit
        
        Returns:
        --------
        pd.DataFrame
            Transformed dataframe
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = list(self.label_encoders.keys())
        elif isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col in df_encoded.columns and col in self.label_encoders:
                # Check if column contains numeric values
                if self._is_numeric_column(df_encoded[col]):
                    print(f"Skipping '{col}' - contains numeric values")
                    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                    continue
                df_encoded[col] = self._transform_column(df_encoded[col], col)
        
        return df_encoded
    
    def _transform_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Transform a single column using fitted encoder
        """
        series_copy = series.copy()
        na_mask = series_copy.isna()
        
        # Handle unknown categories
        result = []
        for idx, value in series_copy.items():
            if pd.isna(value):
                result.append(0)
            elif value in self.mapping_dict[col_name]:
                result.append(self.mapping_dict[col_name][value])
            else:
                # Unknown category - you can handle this differently if needed
                # Option 1: Assign a new code
                # Option 2: Assign to 0 (same as NA)
                # Option 3: Raise an error
                result.append(0)  # Treating unknown as NA
        
        return pd.Series(result, index=series.index)
    
    def get_mapping(self, column: str = None) -> Union[Dict, Dict[str, Dict]]:
        """
        Get the mapping dictionary for encoded values
        
        Parameters:
        -----------
        column : str, optional
            Column name. If None, return all mappings
        
        Returns:
        --------
        dict
            Mapping of original values to encoded values
        """
        if column:
            return self.mapping_dict.get(column, {})
        return self.mapping_dict
    
    def inverse_transform(self, df: pd.DataFrame, columns: Union[List[str], str] = None) -> pd.DataFrame:
        """
        Inverse transform encoded values back to original
        
        Parameters:
        -----------
        df : pd.DataFrame
            Encoded dataframe
        columns : list or str, optional
            Columns to decode
        
        Returns:
        --------
        pd.DataFrame
            Decoded dataframe
        """
        df_decoded = df.copy()
        
        if columns is None:
            columns = list(self.label_encoders.keys())
        elif isinstance(columns, str):
            columns = [columns]
        
        for col in columns:
            if col in df_decoded.columns and col in self.mapping_dict:
                # Create inverse mapping
                inverse_mapping = {v: k for k, v in self.mapping_dict[col].items()}
                
                # Apply inverse mapping
                df_decoded[col] = df_decoded[col].map(inverse_mapping)
                
                # Replace 'NA' string with actual NaN
                df_decoded[col] = df_decoded[col].replace('NA', np.nan)
        
        return df_decoded

def get_value_label_dict(self, custom_labels: Dict[str, Dict] = None) -> Dict[str, Dict]:
    """ 
    Get a dictionary mapping column names to their value-label mappings.
    
    Parameters:
    -----------
    custom_labels : dict, optional
        Custom labels to override default mappings
    
    Returns:
    --------
    dict
        Dictionary with column names as keys and value-label mappings as values
    """
    
    value_label_dict = {}
    
    for col in self.mapping_dict:
        if col in custom_labels:
            value_label_dict[col] = custom_labels[col]
        else:
            value_label_dict[col] = self.mapping_dict[col]
    
    return value_label_dict
    