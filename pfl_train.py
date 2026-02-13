"""
Personalized FL: Fine-tune the global federated model locally per client.
Run this AFTER federated training completes to create personalized models.

Usage:
    python pfl_train.py --model final_model.json --epochs 50
"""
import argparse
import xgboost as xgb
from task import load_data
from pathlib import Path
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
import numpy as np
import csv

# XGBoost parameters (from pyproject.toml)
PARAMS = {
    'objective': 'binary:logistic',
    'eta': 0.09,
    'max_depth': 10,
    'reg_lambda': 82.0,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'seed': 42,
    'nthread': 16
}

# Data configuration
IMBALANCE_STRATEGY = 'adasyn'
SAMPLING_STRATEGY = 0.01
NUM_CLIENTS = 4
BANK_IDS = ['m741', 'm1818', 'm2310', 'm544']

def personalize_client(client_id, num_clients, global_model_path, params, epochs, imbalance_strategy, sampling_strategy, bank_id):
    """Fine-tune global model for one client and evaluate."""
    
    print(f"\n{'='*60}")
    print(f"Personalizing for Client {client_id} ({bank_id})")
    print(f"{'='*60}")
    
    # Load client's local data (train + test)
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
        client_id,
        num_clients,
        imbalance_strategy=imbalance_strategy,
        sampling_strategy=sampling_strategy
    )
    
    # Load global model and evaluate baseline
    bst = xgb.Booster(params=params)
    bst.load_model(global_model_path)
    print(f"Loaded global model: {bst.num_boosted_rounds()} trees")
    
    # Evaluate global model before personalization
    y_true = valid_dmatrix.get_label()
    y_pred_proba_global = bst.predict(valid_dmatrix)
    auc_global = roc_auc_score(y_true, y_pred_proba_global)
    
    # F1@0.5 (fixed threshold)
    y_pred_global = (y_pred_proba_global >= 0.5).astype(int)
    f1_global = f1_score(y_true, y_pred_global)
    
    # F1-optimal (best threshold)
    prec, rec, thr = precision_recall_curve(y_true, y_pred_proba_global)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.argmax(f1s)
    f1_optimal_global = f1s[best_idx]
    
    print(f"Global model - F1@0.5: {f1_global:.4f}, F1-optimal: {f1_optimal_global:.4f}, AUC: {auc_global:.4f}")
    
    # Fine-tune locally
    print(f"Fine-tuning for {epochs} epochs on {num_train} samples...")
    bst = xgb.train(
        params,
        train_dmatrix,
        num_boost_round=epochs,
        xgb_model=bst  # Continue from global model
    )
    
    # Evaluate personalized model
    y_pred_proba_personalized = bst.predict(valid_dmatrix)
    auc_personalized = roc_auc_score(y_true, y_pred_proba_personalized)
    
    # F1@0.5 (fixed threshold)
    y_pred_personalized = (y_pred_proba_personalized >= 0.5).astype(int)
    f1_personalized = f1_score(y_true, y_pred_personalized)
    
    # F1-optimal (best threshold)
    prec, rec, thr = precision_recall_curve(y_true, y_pred_proba_personalized)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.argmax(f1s)
    f1_optimal_personalized = f1s[best_idx]
    
    improvement = f1_personalized - f1_global
    improvement_optimal = f1_optimal_personalized - f1_optimal_global
    
    print(f"Personalized model - F1@0.5: {f1_personalized:.4f}, F1-optimal: {f1_optimal_personalized:.4f}, AUC: {auc_personalized:.4f}")
    print(f"Improvement: F1@0.5={improvement:+.4f}, F1-optimal={improvement_optimal:+.4f}")
    
    # Save personalized model
    output_path = f"personalized_model_client_{client_id}.json"
    bst.save_model(output_path)
    print(f"✓ Saved: {output_path} ({bst.num_boosted_rounds()} trees)")
    
    return {
        'client_id': client_id,
        'f1_global': f1_global,
        'f1_personalized': f1_personalized,
        'f1_optimal_global': f1_optimal_global,
        'f1_optimal_personalized': f1_optimal_personalized,
        'auc_global': auc_global,
        'auc_personalized': auc_personalized,
        'improvement': improvement,
        'improvement_optimal': improvement_optimal,
        'num_examples': num_val
    }

def main():
    parser = argparse.ArgumentParser(description="Personalized FL fine-tuning")
    parser.add_argument("--model", default="final_model.json", help="Path to global model")
    parser.add_argument("--epochs", type=int, default=50, help="Fine-tuning epochs per client")
    parser.add_argument("--run-tag", default="pfl", help="Run tag for output filename")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PERSONALIZED FL")
    print("="*60)
    print(f"Global model: {args.model}")
    print(f"Fine-tuning epochs: {args.epochs}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Imbalance strategy: {IMBALANCE_STRATEGY}")
    print(f"Sampling strategy: {SAMPLING_STRATEGY}")
    print("="*60)
    
    # Personalize for each client
    results = []
    for client_id in range(NUM_CLIENTS):
        result = personalize_client(
            client_id,
            NUM_CLIENTS,
            args.model,
            PARAMS,
            args.epochs,
            IMBALANCE_STRATEGY,
            SAMPLING_STRATEGY,
            BANK_IDS[client_id]
        )
        result['bank_id'] = BANK_IDS[client_id]
        results.append(result)
    
    # Save results to CSV
    metrics_dir = Path("metrics3")
    metrics_dir.mkdir(exist_ok=True)
    csv_file = metrics_dir / f"pfl_results_{args.run_tag}_{args.epochs}epochs_{IMBALANCE_STRATEGY}_{SAMPLING_STRATEGY}.csv"
    
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['client_id', 'bank_id', 
                      'f1_global', 'f1_personalized', 'improvement',
                      'f1_optimal_global', 'f1_optimal_personalized', 'improvement_optimal',
                      'auc_global', 'auc_personalized', 'num_examples']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print("\n" + "="*60)
    print("✓ Personalization complete!")
    print(f"Results saved to: {csv_file}")
    print("="*60)
    print("\nSummary (F1@0.5):")
    for r in results:
        print(f"  {r['bank_id']}: F1 {r['f1_global']:.4f} → {r['f1_personalized']:.4f} ({r['improvement']:+.4f})")
    print("\nSummary (F1-optimal):")
    for r in results:
        print(f"  {r['bank_id']}: F1 {r['f1_optimal_global']:.4f} → {r['f1_optimal_personalized']:.4f} ({r['improvement_optimal']:+.4f})")

if __name__ == "__main__":
    main()
