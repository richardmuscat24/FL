"""Flower ServerApp for XGBoost"""
import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging
from task import replace_keys

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    
    # Flatten config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    
    # Init global model (empty at start)
    global_model = b""
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])
    
    # Initialize FedXgbBagging strategy
    strategy = FedXgbBagging(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )
    
    # Start strategy, run FedXgbBagging for `num_rounds`
    print(f"\n{'='*60}")
    print(f"Starting Federated XGBoost Training")
    print(f"{'='*60}")
    print(f"Rounds: {num_rounds}")
    print(f"Clients: 5 bank clients")
    print(f"{'='*60}\n")
    
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    
    # Save final model to disk
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Check if result has arrays
    if result and hasattr(result, 'arrays') and result.arrays:
        # Try to get the model - handle different key formats
        try:
            # Try index 0 first
            if 0 in result.arrays:
                global_model_array = result.arrays[0]
            elif "0" in result.arrays:
                global_model_array = result.arrays["0"]
            else:
                # Get first available key
                first_key = list(result.arrays.keys())[0]
                print(f"Using key: {first_key}")
                global_model_array = result.arrays[first_key]
            
            # Convert to bytes
            global_model = bytearray(global_model_array.numpy().tobytes())
            
            # Load into booster
            bst = xgb.Booster(params=params)
            bst.load_model(global_model)
            
            # Save model
            print("Saving final model to disk...")
            bst.save_model("final_model.json")
            print("✓ Model saved to: final_model.json")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
            print(f"Result arrays keys: {list(result.arrays.keys()) if result.arrays else 'None'}")
    else:
        print("Warning: No model returned from training")
    
    print("="*60 + "\n")