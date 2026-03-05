"""
Standalone inference script using saved FL model.
Replicates the exact evaluation logic from client_app.py
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_recall_curve, roc_auc_score, 
    precision_score, recall_score, auc
)
from pathlib import Path
import argparse


def load_bank_data(bank_id):
    """
    Load and prepare bank data exactly as in task.py load_data()
    with imbalance_strategy='none' (for evaluation)
    """
    csv_path = f"../data/enriched/ebank_{bank_id}.csv"
    
    # Check file exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Load CSV
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    print(f"Loaded {csv_path}: shape {data.shape}")
    
    # Split features and labels (last column is label)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    # Remove ID columns (same as task.py)
    remove_cols = [0, 1, 2, 10, 11]
    X = np.delete(X, remove_cols, axis=1)
    print(f"After removing ID cols: {X.shape}")
    
    # Temporal split (80/20 chronological - NO SHUFFLING)
    # This MUST match what's done in task.py
    n_test = int(len(X) * 0.2)
    X_train = X[:-n_test]
    y_train = y[:-n_test]
    X_test = X[-n_test:]
    y_test = y[-n_test:]
    
    print(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples")
    print(f"Test fraud rate: {y_test.mean():.4f}")
    
    # Convert to DMatrix (for XGBoost)
    test_dmatrix = xgb.DMatrix(X_test, label=y_test)
    
    return test_dmatrix, y_test, len(y_test)


def evaluate_model(model_path, bank_id):
    """
    Evaluate saved model on bank's test data.
    Replicates exact evaluation logic from client_app.py evaluate()
    """
    print(f"\n{'='*70}")
    print(f"Evaluating Model: {model_path}")
    print(f"Bank: {bank_id}")
    print(f"{'='*70}\n")
    
    # Load test data (no upsampling, just like FL evaluation)
    test_dmatrix, y_true, num_test = load_bank_data(bank_id)
    
    # Load the saved model
    # Match the exact loading from client_app.py evaluate()
    bst = xgb.Booster()
    bst.load_model(model_path)
    
    print(f"Model loaded: {bst.num_boosted_rounds()} trees")
    
    # Predict (same as client_app.py)
    y_pred_proba = bst.predict(test_dmatrix)
    
    print(f"Predictions - Min: {y_pred_proba.min():.4f}, Max: {y_pred_proba.max():.4f}, Mean: {y_pred_proba.mean():.4f}")
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Find F1-optimal threshold (same logic as client_app.py)
    prec, rec, thr = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate AUC-PR
    auc_pr = auc(rec, prec)
    
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.argmax(f1s)
    f1_optimal = f1s[best_idx]
    best_thr = thr[best_idx]
    precision_optimal = prec[best_idx]
    recall_optimal = rec[best_idx]
    
    # Calculate metrics at 0.5 threshold
    y_pred_05 = (y_pred_proba >= 0.5).astype(int)
    f1_05 = f1_score(y_true, y_pred_05, zero_division=0)
    precision_05 = precision_score(y_true, y_pred_05, zero_division=0)
    recall_05 = recall_score(y_true, y_pred_05, zero_division=0)
    
    # Print results (same format as client_app.py)
    print(f"\n{'='*70}")
    print(f"RESULTS - Bank {bank_id}")
    print(f"{'='*70}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"\nAt threshold = 0.5:")
    print(f"  F1:        {f1_05:.4f}")
    print(f"  Precision: {precision_05:.4f}")
    print(f"  Recall:    {recall_05:.4f}")
    print(f"\nAt optimal threshold = {best_thr:.4f}:")
    print(f"  F1:        {f1_optimal:.4f}")
    print(f"  Precision: {precision_optimal:.4f}")
    print(f"  Recall:    {recall_optimal:.4f}")
    print(f"\nTest samples: {num_test}")
    print(f"{'='*70}\n")
    
    return {
        'roc_auc': roc_auc,
        'auc_pr': auc_pr,
        'f1_05': f1_05,
        'precision_05': precision_05,
        'recall_05': recall_05,
        'f1_optimal': f1_optimal,
        'precision_optimal': precision_optimal,
        'recall_optimal': recall_optimal,
        'best_threshold': best_thr,
        'num_examples': num_test,
        'predictions': y_pred_proba  # For debugging
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate saved FL model on bank test data')
    parser.add_argument('--model', '-m', type=str, required=True, help='Path to saved model JSON file')
    parser.add_argument('--bank', '-b', type=str, required=True, help='Bank ID (e.g., 1677, m741)')
    parser.add_argument('--save-predictions', '-s', action='store_true', help='Save predictions to file for debugging')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(args.model, args.bank)
    
    # Optionally save predictions for debugging
    if args.save_predictions:
        pred_file = f"predictions_{args.bank}_inference.npy"
        np.save(pred_file, results['predictions'])
        print(f"Predictions saved to: {pred_file}")
