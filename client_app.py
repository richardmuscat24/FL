"""Flower ClientApp for XGBoost with imbalance handling support"""
import warnings
import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, precision_score, recall_score, auc
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
    # if imbalance_strategy in ['smote', 'adasyn']:
    #     print(f"  Sampling strategy: {sampling_strategy}")
    # print(f"{'='*60}\n")
    
    # Load data WITH imbalance handling
    train_dmatrix, _, num_train, _ = load_data(
        partition_id, 
        num_partitions,
        imbalance_strategy=imbalance_strategy,
        sampling_strategy=sampling_strategy
    )

    # bank_ids = ['1677', '4', '2', '146', '4870'] #small
    bank_ids = ['m741', 'm1818', 'm2310','m544'] #medium 

    bank_id = bank_ids[partition_id]  # Map partition_id to bank_id

    NUM_ROUNDS = {
        "1677": 482,
        "2": 580,
        "4870": 185,
        "4": 807,
        "146": 314,
        "m741": 717,
        "m1818": 443,
        "m2310": 333,
        "m544": 381
    }

    
    # Read training config
    # num_local_round = context.run_config["local-epochs"]
    
    num_local_round = NUM_ROUNDS[bank_id]  # Map partition_id to bank_id for rounds

    # Flatten config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    BANK_HYPERPARAMS = {
        "1677": {
            "max_depth": 10,
            "eta": 0.019403187804511373,
            "min_child_weight": 9,
            "subsample": 0.5247519822601464,
            "colsample_bytree": 0.8588953819457197,
            "gamma": 0.0,  # Add if missing
            "reg_lambda": 1.0,  # Add if missing
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": 42,
            "nthread": 16
        },
        "2": {
            "max_depth": 14,
            "eta": 0.09133070167890314,
            "min_child_weight": 5,
            "subsample": 0.7830044552882892,
            "colsample_bytree": 0.9233293259886008,
            "gamma": 1.137754362329786,
            "reg_lambda": 3.552725534174049,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": 42,
            "nthread": 16
        },
        "146": {
            "max_depth": 10,
            "eta": 0.07507182413546831,
            "min_child_weight": 10,
            "subsample": 0.9330880728874675,
            "colsample_bytree": 0.9753571532049581,
            "gamma": 1.4639878836228102,
            "reg_lambda": 1.3372928062124765,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": 42,
            "nthread": 16
        },
        "4870": {
            "max_depth": 10,
            "eta": 0.09,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "reg_lambda": 82.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": 42,
            "nthread": 16
        },
        "4": {
            "max_depth": 10,
            "eta": 0.0899212045236615,
            "min_child_weight": 5,
            "subsample": 0.823296768065808,
            "colsample_bytree": 0.7352540496296012,
            "gamma": 1.2070576863780036,
            "reg_lambda": 0.13814964816593606,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": 42,
            "nthread": 16
        },
        "m741": {
            "max_depth": 13,
            "eta": 0.07234131022059624,
            "min_child_weight": 8,
            "subsample": 0.6092310343361069,
            "colsample_bytree": 0.7467268297589102,
            "gamma": 2.2855723450500065,
            "reg_lambda": 70.9619275247569,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": 42,
            "nthread": 16
        },
        "m1818": {
            "max_depth": 10,
            "eta": 0.04527052030292707,
            "min_child_weight": 9,
            "subsample": 0.7646839077806987,
            "colsample_bytree": 0.6873248697791162,
            "gamma": 1.1758383388449716,
            "reg_lambda": 14.174384184375949,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": 42,
            "nthread": 16
        },
        "m2310": {
            "max_depth": 4,
            "eta": 0.050821957961647915,
            "min_child_weight": 2,
            "subsample": 0.9947933960703935,
            "colsample_bytree": 0.9314893414198531,
            "gamma": 0.4846150141038674,
            "reg_lambda": 1.3729365499483956,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": 42,
            "nthread": 16
        },
        "m544": {
            "max_depth": 5,
            "eta": 0.055377757618561646,
            "min_child_weight": 2,
            "subsample": 0.7591753462663356,
            "colsample_bytree": 0.6691879991376378,
            "gamma": 4.548133242047752,
            "reg_lambda": 1.6346740590130946,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": 42,
            "nthread": 16
        }
    }
    
    # 🎯 Get bank-specific hyperparameters
    if bank_id in BANK_HYPERPARAMS:
        # Override params with bank-specific hyperparameters
        params = BANK_HYPERPARAMS[bank_id]
        print(f"  Using bank-specific params for Bank {bank_id}")
    else:
        print(f"  Warning: No specific params for Bank {bank_id}, using default params")
    
    print(f"  XGBoost params: {params}")
    
    
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
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Find F1-optimal threshold
    prec, rec, thr = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate AUC-PR (area under precision-recall curve)
    auc_pr = auc(rec, prec)
    
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.argmax(f1s)
    f1_optimal = f1s[best_idx]
    best_thr = thr[best_idx]
    precision_optimal = prec[best_idx]
    recall_optimal = rec[best_idx]
    
    # Also calculate metrics at 0.5 threshold for comparison
    y_pred_05 = (y_pred_proba >= 0.5).astype(int)
    f1_05 = f1_score(y_true, y_pred_05, zero_division=0)
    precision_05 = precision_score(y_true, y_pred_05, zero_division=0)
    recall_05 = recall_score(y_true, y_pred_05, zero_division=0)
    
    server_round = msg.content["config"]["server-round"]
    
    print(f"Client {partition_id} | Round {server_round:2d} | "
          f"ROC-AUC: {roc_auc:.4f} | AUC-PR: {auc_pr:.4f} | "
          f"F1(opt): {f1_optimal:.4f} @ {best_thr:.3f} | F1(0.5): {f1_05:.4f} | "
          f"pred∈[{y_pred_proba.min():.3f},{y_pred_proba.max():.3f}]")
    
    # ========================================
    # 💾 SAVE PR CURVE DATA (Final Round Only)
    # ========================================
    num_server_rounds = context.run_config.get("num-server-rounds", 20)
    if server_round == num_server_rounds:
        metrics_dir = Path("metrics3")
        metrics_dir.mkdir(exist_ok=True)
        
        run_tag = context.run_config.get("run-tag", "")
        imbalance_strategy = context.run_config.get("imbalance-strategy", "none")
        imbalance_sampling = context.run_config.get("sampling-strategy", "none")
        
        # bank_ids = ['1677', '4', '2', '146', '4870'] #small
        bank_ids = ['m741', 'm1818', 'm2310','m544'] #medium
        bank_id = bank_ids[partition_id]
        
        # Clean up sampling for filename: if 0, use "client_specific"
        sampling_str = "client_specific" if str(imbalance_sampling) == "0" else str(imbalance_sampling)
        
        # Save PR curve data
        pr_file = metrics_dir / f"pr_curve_{run_tag}_{bank_id}_{imbalance_strategy}_{sampling_str}.npz"
        np.savez(pr_file, precision=prec, recall=rec, thresholds=thr, 
                 optimal_threshold=best_thr, f1_scores=f1s)
        print(f"  💾 Saved PR curve data to {pr_file}")

        pred_file = metrics_dir / f"predictions_{run_tag}_{bank_id}_{imbalance_strategy}_{sampling_str}.npz"
        np.savez(pred_file, 
                    y_test=y_true,        # ground truth — same as local
                    y_prob=y_pred_proba,  # federated model probabilities
                    optimal_threshold=best_thr)
        print(f"  💾 Saved predictions to {pred_file}")
    
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
            'round', 'client_id', 'f1_05', 'precision_05', 'recall_05',
            'f1_optimal', 'precision_optimal', 'recall_optimal', 
            'roc_auc', 'auc_pr', 'best_threshold', 'num_examples', 
            'imbalance_strategy', 'imbalance_sampling'
        ])
        
        if not file_exists:
            writer.writeheader()

        # bank_ids = ['1677', '4', '2', '146', '4870'] #small
        bank_ids = ['m741', 'm1818', 'm2310','m544'] #medium 
        
                
        writer.writerow({
            'round': server_round,
            'client_id': bank_ids[partition_id],
            'f1_05': float(f1_05),
            'precision_05': float(precision_05),
            'recall_05': float(recall_05),
            'f1_optimal': float(f1_optimal),
            'precision_optimal': float(precision_optimal),
            'recall_optimal': float(recall_optimal),
            'roc_auc': float(roc_auc),
            'auc_pr': float(auc_pr),
            'best_threshold': float(best_thr),
            'num_examples': num_val,
            'imbalance_strategy': imbalance_strategy,
            'imbalance_sampling': imbalance_sampling
        })
    
    # Construct reply message
    metrics = {
        "roc_auc": float(roc_auc),
        "auc_pr": float(auc_pr),
        "f1_optimal": float(f1_optimal),
        "f1_05": float(f1_05),
        "best_threshold": float(best_thr),
        "num-examples": num_val
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    return Message(content=content, reply_to=msg)
