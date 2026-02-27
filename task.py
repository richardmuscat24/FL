"""Data loading and utilities for federated XGBoost"""
from imblearn.over_sampling import ADASYN, SMOTE
import xgboost as xgb
import numpy as np
from pathlib import Path
import logging

# Configure logging at module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


SAMPLING_STRATEGIES = {
    "1677": 0.05468166602085926,
    "2": 0.2,
    "4870": 0.09,
    "4": 0.09,
    "146": 0.0550474,
    "m741": 0.021701533945381567,
    "m1818": 0.017292752001369337,
    "m2310": 0.032103571433416155,
    "m544": 0.15755520301000245
}

def get_sampling_strategy(bank_id):
    """Get bank-specific ADASYN sampling strategy"""
    bank_key = str(bank_id)
    if bank_key not in SAMPLING_STRATEGIES:
        raise ValueError(f"No sampling strategy found for bank {bank_id}")
    return SAMPLING_STRATEGIES[bank_key]


def load_data(partition_id, num_partitions, 
              bank_ids=None,
              imbalance_strategy='none', 
              sampling_strategy=0.1):
    """
    Load partition data from CSV files with optional upsampling.
    
    Args:
        partition_id: Client ID (0-indexed)
        num_partitions: Total number of partitions
        bank_ids: List of bank IDs (optional, uses default if None)
        imbalance_strategy: 'none', 'smote', 'adasyn', or 'scale_pos_weight'
        sampling_strategy: Sampling ratio for SMOTE/ADASYN
        
    Returns:
        train_dmatrix, valid_dmatrix, num_train, num_val
    """
    
    # Default to your 5 banks if not specified
    if bank_ids is None:
        bank_ids = ['1677', '4', '2', '146', '4870'] #small
        # bank_ids = ['m741', 'm1818', 'm2310','m544'] #medium 
    
    if partition_id >= len(bank_ids):
        raise ValueError(f"Invalid partition {partition_id}, max is {len(bank_ids)-1}")
    
    bank_id = bank_ids[partition_id]
    csv_path = f"../data/enriched/ebank_{bank_id}.csv"
    
    logger.info(f"Partition {partition_id} → Bank {bank_id}: {csv_path}")
    
    # Check file exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Load CSV
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    logger.info(f"  Loaded shape: {data.shape}")
    
    # Split features and labels (last column is label)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    # Remove ID columns
    remove_cols = [0, 1, 2, 10, 11]
    X = np.delete(X, remove_cols, axis=1)
    logger.info(f"  After removing ID cols: {X.shape}")
    
    # Temporal split (80/20 chronological - NO SHUFFLING)
    n_test = int(len(X) * 0.2)
    X_train = X[:-n_test]
    y_train = y[:-n_test]
    X_test = X[-n_test:]
    y_test = y[-n_test:]
    
    logger.info(f"  Before upsampling: Train={len(y_train)}, Test={len(y_test)}")
    logger.info(f"  Fraud rate: train={y_train.mean():.4f}, test={y_test.mean():.4f}")
    
    # ========================================
    # APPLY UPSAMPLING (before DMatrix conversion)
    # ========================================
    
    original_train_size = len(y_train)
    original_class_dist = np.bincount(y_train)

    # sampling_strategy = get_sampling_strategy(bank_id)
    
    try:
        if imbalance_strategy == 'smote':
            logger.info(f"  Applying SMOTE (sampling_strategy={sampling_strategy})")
            logger.info(f"    Before: {original_train_size} samples, dist={original_class_dist}")
            
            smote = SMOTE(
                random_state=42, 
                k_neighbors=5, 
                sampling_strategy=sampling_strategy
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            new_class_dist = np.bincount(y_train)
            logger.info(f"    After:  {len(y_train)} samples, dist={new_class_dist}")
            
        elif imbalance_strategy == 'adasyn':
            logger.info(f"  Applying ADASYN (sampling_strategy={sampling_strategy})")
            logger.info(f"    Before: {original_train_size} samples, dist={original_class_dist}")
            
            adasyn = ADASYN(
                random_state=42, 
                n_neighbors=5, 
                sampling_strategy=sampling_strategy
            )
            X_train, y_train = adasyn.fit_resample(X_train, y_train)
            
            new_class_dist = np.bincount(y_train)
            logger.info(f"    After:  {len(y_train)} samples, dist={new_class_dist}")
            
        elif imbalance_strategy == 'scale_pos_weight':
            logger.info(f"  Using scale_pos_weight (configured in XGBoost params)")
            # No data-level resampling, handled by XGBoost
            
        elif imbalance_strategy == 'none':
            logger.info(f"  No imbalance handling applied")
            
        else:
            logger.warning(f"  Unknown imbalance_strategy: {imbalance_strategy}")
            
    except ValueError as e:
        logger.warning(f"  ⚠️  Upsampling failed: {e}")
        logger.warning(f"  Continuing with original training data (no upsampling)")
        # X_train, y_train remain unchanged
    
    # Update counts after potential upsampling
    num_train = len(X_train)
    num_val = len(X_test)
    
    logger.info(f"  Final: Train={num_train}, Test={num_val}")
    logger.info(f"  Final fraud rate: train={y_train.mean():.4f}, test={y_test.mean():.4f}")
    
    # Convert to DMatrix for xgboost (AFTER upsampling!)
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(X_test, label=y_test)
    
    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
