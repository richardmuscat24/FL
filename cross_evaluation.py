"""
Leave-one-out cross-evaluation for federated XGBoost models.
Evaluates a global FL model on concatenated test sets with one bank held out.
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_recall_curve, roc_auc_score, 
    precision_score, recall_score, auc
)
from pathlib import Path
import argparse
import csv


# Federation definitions
FEDERATIONS = {
    'small': ['1677', '4', '2', '146', '4870'],
    'medium': ['m741', 'm1818', 'm2310', 'm544']
}


def load_bank_test_data(bank_id):
    """
    Load test data for a single bank.
    Replicates exact logic from task.py load_data() with imbalance_strategy='none'
    """
    csv_path = f"../data/enriched/ebank_{bank_id}.csv"
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Load CSV
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    
    # Split features and labels (last column is label)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    # Remove ID columns (same as task.py)
    remove_cols = [0, 1, 2, 10, 11]
    X = np.delete(X, remove_cols, axis=1)
    
    # Temporal split (80/20 chronological - NO SHUFFLING)
    n_test = int(len(X) * 0.2)
    X_test = X[-n_test:]
    y_test = y[-n_test:]
    
    return X_test, y_test


def concatenate_test_sets(bank_list):
    """
    Concatenate test sets from multiple banks.
    Returns combined X, y and dict with per-bank sample counts.
    """
    X_combined = []
    y_combined = []
    bank_sizes = {}
    
    for bank_id in bank_list:
        X_test, y_test = load_bank_test_data(bank_id)
        X_combined.append(X_test)
        y_combined.append(y_test)
        bank_sizes[bank_id] = len(y_test)
    
    X_combined = np.vstack(X_combined)
    y_combined = np.concatenate(y_combined)
    
    return X_combined, y_combined, bank_sizes


def evaluate_on_combination(model_path, eval_banks, left_out_bank):
    """
    Evaluate model on concatenated test sets.
    """
    print(f"\n{'='*70}")
    print(f"Left-out Bank: {left_out_bank}")
    print(f"Evaluating on: {eval_banks}")
    print(f"{'='*70}")
    
    # Load concatenated test data
    X_test, y_true, bank_sizes = concatenate_test_sets(eval_banks)
    
    print(f"Combined test set:")
    for bank_id, size in bank_sizes.items():
        fraud_rate = load_bank_test_data(bank_id)[1].mean()
        print(f"  Bank {bank_id}: {size} samples (fraud rate: {fraud_rate:.4f})")
    print(f"  TOTAL: {len(y_true)} samples (fraud rate: {y_true.mean():.4f})")
    
    # Convert to DMatrix
    test_dmatrix = xgb.DMatrix(X_test, label=y_true)
    
    # Load the saved FL model (simple loading, no params override)
    bst = xgb.Booster()
    bst.load_model(model_path)
    
    print(f"\nModel: {bst.num_boosted_rounds()} trees")
    
    # Predict
    y_pred_proba = bst.predict(test_dmatrix)
    
    print(f"Predictions - Min: {y_pred_proba.min():.4f}, Max: {y_pred_proba.max():.4f}, Mean: {y_pred_proba.mean():.4f}")
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Find F1-optimal threshold
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
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS")
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
    print(f"{'='*70}\n")
    
    return {
        'left_out_bank': left_out_bank,
        'eval_banks': ','.join(eval_banks),
        'num_banks_eval': len(eval_banks),
        'roc_auc': roc_auc,
        'auc_pr': auc_pr,
        'f1_05': f1_05,
        'precision_05': precision_05,
        'recall_05': recall_05,
        'f1_optimal': f1_optimal,
        'precision_optimal': precision_optimal,
        'recall_optimal': recall_optimal,
        'best_threshold': best_thr,
        'num_examples': len(y_true),
        'predictions': y_pred_proba,
        'y_true': y_true,
    }


def run_cross_evaluation(model_path, federation_type, output_csv=None, save_predictions=False):
    """
    Run leave-one-out cross-evaluation for all banks in the federation.
    """
    if federation_type not in FEDERATIONS:
        raise ValueError(f"Invalid federation type. Choose from: {list(FEDERATIONS.keys())}")
    
    bank_list = FEDERATIONS[federation_type]
    
    print(f"\n{'#'*70}")
    print(f"# LEAVE-ONE-OUT CROSS-EVALUATION")
    print(f"# Model: {model_path}")
    print(f"# Federation: {federation_type} - {bank_list}")
    print(f"{'#'*70}\n")
    
    all_results = []
    
    # Iterate through each bank as left-out
    for left_out_bank in bank_list:
        # Create evaluation set (all banks except left-out)
        eval_banks = [b for b in bank_list if b != left_out_bank]
        
        # Run evaluation
        results = evaluate_on_combination(model_path, eval_banks, left_out_bank)
        all_results.append(results)
        
        # Optionally save predictions
        if save_predictions:
            pred_file = f"exp2_federated_{left_out_bank}.npz"
            np.savez(
                pred_file,
                y_prob=results['predictions'],
                y_test=results['y_true'],
                optimal_threshold=np.float32(results['best_threshold'])
            )
            print(f"  Predictions saved to: {pred_file}\n")
    
    # Save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = [
                'left_out_bank', 'eval_banks', 'num_banks_eval',
                'f1_05', 'precision_05', 'recall_05',
                'f1_optimal', 'precision_optimal', 'recall_optimal',
                'roc_auc', 'auc_pr', 'best_threshold', 'num_examples'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in all_results:
                row = {k: v for k, v in result.items() if k in fieldnames}
                writer.writerow(row)
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}\n")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY - {federation_type.upper()} FEDERATION")
    print(f"{'='*70}")
    print(f"{'Left-Out':<10} {'AUC-PR':<8} {'F1@opt':<8} {'F1@0.5':<8} {'Threshold':<10}")
    print(f"{'-'*70}")
    for result in all_results:
        print(f"{result['left_out_bank']:<10} "
              f"{result['auc_pr']:<8.4f} "
              f"{result['f1_optimal']:<8.4f} "
              f"{result['f1_05']:<8.4f} "
              f"{result['best_threshold']:<10.4f}")
    
    # Calculate averages
    avg_auc_pr = np.mean([r['auc_pr'] for r in all_results])
    avg_f1_opt = np.mean([r['f1_optimal'] for r in all_results])
    avg_f1_05  = np.mean([r['f1_05'] for r in all_results])
    
    print(f"{'-'*70}")
    print(f"{'AVERAGE':<10} {avg_auc_pr:<8.4f} {avg_f1_opt:<8.4f} {avg_f1_05:<8.4f}")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Leave-one-out cross-evaluation for federated XGBoost models'
    )
    parser.add_argument(
        '--model', '-m', 
        type=str, 
        required=True, 
        help='Path to saved FL model JSON file'
    )
    parser.add_argument(
        '--federation', '-f', 
        type=str, 
        required=True,
        choices=['small', 'medium'],
        help='Federation type: small or medium'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str,
        default=None,
        help='Output CSV file path (default: auto-generated in metrics3/)'
    )
    parser.add_argument(
        '--save-predictions', '-s', 
        action='store_true',
        help='Save npz prediction files for PR curve generation'
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        model_name = Path(args.model).stem
        args.output = f"metrics3/cross_eval_{model_name}_{args.federation}.csv"
    
    run_cross_evaluation(
        model_path=args.model,
        federation_type=args.federation,
        output_csv=args.output,
        save_predictions=args.save_predictions
    )