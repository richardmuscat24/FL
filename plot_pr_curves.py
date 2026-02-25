"""Plot Precision-Recall curves for all clients after FL training"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict

def load_pr_file(pr_file):
    """Load a single PR curve file and return its data"""
    data = np.load(pr_file)
    precision = data['precision']
    recall = data['recall']
    optimal_threshold = float(data['optimal_threshold'])
    f1_scores = data['f1_scores']
    
    # Find optimal point
    best_idx = np.argmax(f1_scores)
    opt_prec = precision[best_idx]
    opt_rec = recall[best_idx]
    
    return {
        'precision': precision,
        'recall': recall,
        'optimal_threshold': optimal_threshold,
        'opt_prec': opt_prec,
        'opt_rec': opt_rec,
        'f1_scores': f1_scores
    }


def extract_metadata(filename):
    """Extract experiment metadata from filename"""
    # Format: pr_curve_{run_tag}_{bank_id}_{strategy}_{sampling}.npz
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) < 5:
        return None
    
    # Handle cases where run_tag or other parts have underscores
    bank_id = parts[-3]  # third from end
    strategy = parts[-2]  # second from end
    sampling = parts[-1]  # last
    run_tag = '_'.join(parts[2:-3])  # everything between pr_curve and bank_id
    
    return {
        'run_tag': run_tag,
        'bank_id': bank_id,
        'strategy': strategy,
        'sampling': sampling
    }


def group_pr_files(metrics_dir="metrics3"):
    """Group PR curve files by experiment (run_tag, strategy, sampling)"""
    metrics_path = Path(metrics_dir)
    pr_files = list(metrics_path.glob("pr_curve_*.npz"))
    
    if not pr_files:
        return {}
    
    # Group files by experiment
    experiments = defaultdict(list)
    
    for pr_file in pr_files:
        meta = extract_metadata(pr_file.name)
        if meta:
            exp_key = (meta['run_tag'], meta['strategy'], meta['sampling'])
            experiments[exp_key].append({
                'file': pr_file,
                'bank_id': meta['bank_id']
            })
    
    return experiments


def plot_experiment(file_list, output_name=None, title_suffix=""):
    """
    Plot PR curves for a list of files (one experiment).
    
    Args:
        file_list: List of dicts with 'file' and 'bank_id' keys
        output_name: Output filename (if None, auto-generate)
        title_suffix: Additional text for title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Sort by bank_id for consistent ordering
    file_list = sorted(file_list, key=lambda x: x['bank_id'])
    
    for idx, item in enumerate(file_list):
        pr_file = item['file']
        bank_id = item['bank_id']
        
        try:
            data = load_pr_file(pr_file)
            
            # Plot PR curve
            ax.plot(data['recall'], data['precision'], 
                    label=f'Bank {bank_id}', 
                    linewidth=2, 
                    color=colors[idx % len(colors)])
            
            # Mark optimal point
            ax.scatter(data['opt_rec'], data['opt_prec'], 
                      s=100, 
                      color=colors[idx % len(colors)],
                      marker='*',
                      edgecolors='black',
                      linewidths=1.5,
                      zorder=5)
            
            print(f"  ✓ Bank {bank_id}: Prec={data['opt_prec']:.3f}, Rec={data['opt_rec']:.3f}, Thr={data['optimal_threshold']:.3f}")
            
        except Exception as e:
            print(f"  ⚠️  Error loading {pr_file.name}: {e}")
            continue
    
    # Formatting
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    
    title = f'Precision-Recall Curves (Round 20){title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add diagonal reference line
    ax.plot([0, 1], [0.5, 0.5], 'k--', alpha=0.3, linewidth=1)
    
    # Save figure
    if output_name is None:
        output_name = f"pr_curves_{file_list[0]['file'].stem.replace('pr_curve_', '')}.png"
    
    output_file = Path("metrics3") / output_name
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved: {output_file}\n")
    
    plt.show()


def list_experiments(metrics_dir="metrics3"):
    """List all available experiments"""
    experiments = group_pr_files(metrics_dir)
    
    if not experiments:
        print("❌ No PR curve files found in metrics3/")
        return []
    
    print(f"\n{'='*70}")
    print(f"Found {len(experiments)} experiment(s):")
    print(f"{'='*70}")
    
    exp_list = []
    for idx, (exp_key, files) in enumerate(sorted(experiments.items()), 1):
        run_tag, strategy, sampling = exp_key
        bank_ids = [f['bank_id'] for f in files]
        
        print(f"{idx}. {run_tag} | {strategy} | sampling={sampling}")
        print(f"   Banks: {', '.join(sorted(bank_ids))}")
        print(f"   Files: {len(files)}")
        
        exp_list.append({
            'key': exp_key,
            'files': files,
            'run_tag': run_tag,
            'strategy': strategy,
            'sampling': sampling
        })
    
    print(f"{'='*70}\n")
    return exp_list


def plot_specific_file(file_path):
    """Plot a single PR curve file"""
    pr_file = Path(file_path)
    
    if not pr_file.exists():
        print(f"❌ File not found: {pr_file}")
        return
    
    meta = extract_metadata(pr_file.name)
    if not meta:
        print(f"❌ Invalid filename format: {pr_file.name}")
        return
    
    print(f"\n📊 Plotting: {pr_file.name}")
    print(f"   Bank: {meta['bank_id']}")
    print(f"   Experiment: {meta['run_tag']} | {meta['strategy']} | {meta['sampling']}\n")
    
    plot_experiment(
        [{'file': pr_file, 'bank_id': meta['bank_id']}],
        output_name=f"pr_plot_{pr_file.stem}.png",
        title_suffix=f"\n{meta['run_tag']} | {meta['strategy']} | sampling={meta['sampling']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Precision-Recall curves from FL experiments')
    parser.add_argument('--file', '-f', type=str, help='Path to specific PR curve file to plot')
    parser.add_argument('--list', '-l', action='store_true', help='List all available experiments')
    parser.add_argument('--all', '-a', action='store_true', help='Plot all experiments')
    parser.add_argument('--experiment', '-e', type=int, help='Plot specific experiment number (use --list to see numbers)')
    
    args = parser.parse_args()
    
    if args.file:
        # Plot specific file
        plot_specific_file(args.file)
    
    elif args.list:
        # Just list experiments
        list_experiments()
    
    elif args.all:
        # Plot all experiments
        print("🔍 Finding all experiments...")
        experiments = group_pr_files()
        
        if not experiments:
            print("❌ No PR curve files found!")
        else:
            for exp_key, files in experiments.items():
                run_tag, strategy, sampling = exp_key
                print(f"\n{'='*70}")
                print(f"Plotting: {run_tag} | {strategy} | sampling={sampling}")
                print(f"{'='*70}")
                
                title_suffix = f"\n{run_tag} | {strategy} | sampling={sampling}"
                output_name = f"pr_curves_{run_tag}_{strategy}_{sampling}.png"
                
                plot_experiment(files, output_name=output_name, title_suffix=title_suffix)
    
    elif args.experiment is not None:
        # Plot specific experiment by number
        exp_list = list_experiments()
        
        if not exp_list:
            exit(1)
        
        if args.experiment < 1 or args.experiment > len(exp_list):
            print(f"❌ Invalid experiment number. Choose 1-{len(exp_list)}")
            exit(1)
        
        exp = exp_list[args.experiment - 1]
        
        print(f"\n{'='*70}")
        print(f"Plotting experiment {args.experiment}: {exp['run_tag']} | {exp['strategy']} | {exp['sampling']}")
        print(f"{'='*70}")
        
        title_suffix = f"\n{exp['run_tag']} | {exp['strategy']} | sampling={exp['sampling']}"
        output_name = f"pr_curves_{exp['run_tag']}_{exp['strategy']}_{exp['sampling']}.png"
        
        plot_experiment(exp['files'], output_name=output_name, title_suffix=title_suffix)
    
    else:
        # Default: list experiments and prompt
        exp_list = list_experiments()
        
        if exp_list:
            print("💡 Usage:")
            print("   python plot_pr_curves.py --all                    # Plot all experiments")
            print("   python plot_pr_curves.py -e 1                     # Plot experiment 1")
            print("   python plot_pr_curves.py -f metrics3/pr_curve... # Plot specific file")

