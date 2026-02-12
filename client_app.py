"""Flower ClientApp for XGBoost with imbalance handling support"""
import warnings
import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from task import load_data, replace_keys
import csv
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# Flower ClientApp
app = ClientApp()


def _local_boost(bst_input, num_local_round, train_dmatrix):
    """Update trees based on local training data."""
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())
    
    # Bagging: extract the last N=num_local_round trees for server aggregation
    bst = bst_input[
        bst_input.num_boosted_rounds()
        - num_local_round : bst_input.num_boosted_rounds()
    ]
    return bst


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train local XGBoost model with optional imbalance handling."""
    
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # ✅ Get imbalance configuration from run config
    imbalance_strategy = context.run_config.get("imbalance-strategy", "none")
    sampling_strategy = context.run_config.get("sampling-strategy", 0.1)
   
    
    print(f"\n{'='*60}")
    print(f"Client {partition_id} Training")
    print(f"  Imbalance strategy: {imbalance_strategy}")
    if imbalance_strategy in ['smote', 'adasyn']:
        print(f"  Sampling strategy: {sampling_strategy}")
    print(f"{'='*60}\n")
    
    # Load data WITH imbalance handling
    train_dmatrix, _, num_train, _ = load_data(
        partition_id, 
        num_partitions,
        imbalance_strategy=imbalance_strategy,
        sampling_strategy=sampling_strategy
    )
    
    # Read training config
    num_local_round = context.run_config["local-epochs"]
    
    # Flatten config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    
    # Handle scale_pos_weight if using that strategy
    if imbalance_strategy == "scale_pos_weight":
        # Scale pos weight should already be in params from config
        scale_weight = params.get('scale_pos_weight', 1.0)
        print(f"  Using scale_pos_weight={scale_weight}")
    
    global_round = msg.content["config"]["server-round"]
    
    if global_round == 1:
        # First round local training
        print(f"  Round 1: Training from scratch ({num_local_round} rounds)")
        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=num_local_round,
        )
    else:
        # Load global model and continue training
        print(f"  Round {global_round}: Continuing from global model")
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
        bst.load_model(global_model)
        
        # Local training (bagging)
        bst = _local_boost(bst, num_local_round, train_dmatrix)
    
    # Save model
    local_model = bst.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)
    
    print(f"  ✓ Training complete: {num_train} samples, {bst.num_boosted_rounds()} trees\n")
    
    # Construct reply message
    model_record = ArrayRecord([model_np])
    metrics = {
        "num-examples": num_train
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate global model on local test data."""
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # ✅ NO upsampling for test data! Always use 'none'
    _, valid_dmatrix, _, num_val = load_data(
        partition_id, 
        num_partitions,
        imbalance_strategy='none'  # Important: no upsampling for evaluation
    )
    
    # Load model
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    params["objective"] = "binary:logistic"   # ensure correct objective
    
    bst = xgb.Booster(params=params)
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)
    
    # Predict
    y_true = valid_dmatrix.get_label()
    y_pred_proba = bst.predict(valid_dmatrix)
    
    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Find F1-optimal threshold
    prec, rec, thr = precision_recall_curve(y_true, y_pred_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.argmax(f1s)
    f1_optimal = f1s[best_idx]
    best_thr = thr[best_idx]
    
    # Also calculate F1 at 0.5 threshold for comparison
    y_pred_05 = (y_pred_proba >= 0.5).astype(int)
    f1_05 = f1_score(y_true, y_pred_05, zero_division=0)
    
    server_round = msg.content["config"]["server-round"]
    
    print(f"Client {partition_id} | Round {server_round:2d} | "
          f"AUC: {auc:.4f} | F1(opt): {f1_optimal:.4f} @ {best_thr:.3f} | "
          f"F1(0.5): {f1_05:.4f} | "
          f"pred∈[{y_pred_proba.min():.3f},{y_pred_proba.max():.3f}]")
    
    # ========================================
    # 💾 SAVE METRICS TO CSV
    # ========================================
    metrics_dir = Path("metrics3")
    metrics_dir.mkdir(exist_ok=True)
    
    # Get imbalance strategy from config for filename
    imbalance_strategy = context.run_config.get("imbalance-strategy", "none")
    imbalance_sampling = context.run_config.get("sampling-strategy", "none")
    run_tag = context.run_config.get("run-tag", "")

    csv_file = metrics_dir / f"client_metrics_{run_tag}_{imbalance_strategy}_{imbalance_sampling}.csv"
    
    # Write header if file doesn't exist
    file_exists = csv_file.exists()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'round','client_id', 'f1_05', 'f1_optimal','auc', 
            'best_threshold', 'num_examples', 'imbalance_strategy', 'imbalance_sampling'
        ])
        
        if not file_exists:
            writer.writeheader()

        # bank_ids = ['1677', '4', '2', '146', '4870'] #small
        bank_ids = ['m741', 'm1818', 'm2310','m544'] #medium 
        
                
        writer.writerow({
            'round': server_round,
            'client_id': bank_ids[partition_id],
            'f1_05': float(f1_05),
            'f1_optimal': float(f1_optimal),
            'auc': float(auc),
            'best_threshold': float(best_thr),
            'num_examples': num_val,
            'imbalance_strategy': imbalance_strategy,
            'imbalance_sampling': imbalance_sampling
        })
    
    # Construct reply message
    metrics = {
        "auc": float(auc), 
        "f1_optimal": float(f1_optimal),
        "f1_05": float(f1_05),
        "best_threshold": float(best_thr),
        "num-examples": num_val
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    return Message(content=content, reply_to=msg)
