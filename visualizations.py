"""
Visualization Module for PDA State Diagram and Experimental Results

Authors: Salma Bellaou, Asma Khamri
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import json


def create_pda_state_diagram(output_path: str = "/home/claude/av_block_pda_project/visualizations/pda_state_diagram.png"):
    """
    Create a visual diagram of the PDA states and transitions.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define state positions
    states = {
        'q0': (5, 8, 'Initial State\n(q₀)'),
        'qfirst': (2, 5, 'First-Degree\nAV Block\n(qfirst)'),
        'qmobitz1': (5, 5, 'Mobitz Type I\n(Wenckebach)\n(qmobitz1)'),
        'qmobitz2': (8, 5, 'Mobitz Type II\n(qmobitz2)'),
        'qthird': (5, 2, 'Third-Degree\nComplete Block\n(qthird)')
    }
    
    # Draw states
    for state_name, (x, y, label) in states.items():
        if state_name == 'q0':
            # Initial state - single circle
            circle = plt.Circle((x, y), 0.4, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
        else:
            # Accepting states - double circle
            circle1 = plt.Circle((x, y), 0.45, color='lightgreen', ec='black', linewidth=2)
            circle2 = plt.Circle((x, y), 0.35, fill=False, ec='black', linewidth=2)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
        
        ax.text(x, y, label, ha='center', va='center', fontsize=9, weight='bold')
    
    # Define transitions with labels
    transitions = [
        # To First-Degree
        ('q0', 'qfirst', '≥3 consecutive\nPRlong\n(constant PR)',
         {'connectionstyle': 'arc3,rad=.3'}),
        
        # To Mobitz I
        ('q0', 'qmobitz1', '≥2 PRincrease\n+ P-P pattern\n(progressive PR)',
         {'connectionstyle': 'arc3,rad=0'}),
        
        # To Mobitz II
        ('q0', 'qmobitz2', 'Constant PRlong\n+ P-P pattern\n(sudden drop)',
         {'connectionstyle': 'arc3,rad=-.3'}),
        
        # To Third-Degree
        ('q0', 'qthird', 'Variable PR\n+ P-P or R-R\n(dissociation)',
         {'connectionstyle': 'arc3,rad=.3'}),
    ]
    
    # Draw transitions
    for from_state, to_state, label, style in transitions:
        x1, y1, _ = states[from_state]
        x2, y2, _ = states[to_state]
        
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.4,head_length=0.8',
            color='darkblue',
            linewidth=2,
            **style
        )
        ax.add_patch(arrow)
        
        # Add label
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        offset_x = 0.3 if abs(x2 - x1) > 0.5 else 0
        offset_y = 0.3 if abs(y2 - y1) > 1 else 0
        
        ax.text(mid_x + offset_x, mid_y + offset_y, label,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                fontsize=8)
    
    # Add title and legend
    ax.text(5, 9.5, 'Pushdown Automaton State Diagram for AV Block Classification',
            ha='center', va='center', fontsize=14, weight='bold')
    
    # Add stack alphabet info
    stack_info = """Stack Alphabet (Γ):
• Z₀: Initial stack symbol
• LONG: PR prolongation marker
• PREV_PR: Previous PR reference
• P_MARK: P-wave marker
• R_MARK: R-wave marker"""
    
    ax.text(0.5, 6, stack_info,
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.9),
            fontsize=8, family='monospace')
    
    # Add input alphabet info
    input_info = """Input Alphabet (Σ):
• P: P-wave
• R: QRS complex
• PRnormal: PR ≤ 200ms
• PRlong: PR > 200ms (constant)
• PRincrease: PR > previous PR
• drop: Dropped QRS"""
    
    ax.text(9.5, 6, input_info,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.9),
            fontsize=8, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"State diagram saved to: {output_path}")
    plt.close()


def create_results_visualization(results_path: str,
                                 output_dir: str = "/home/claude/av_block_pda_project/visualizations/"):
    """
    Create visualizations of experimental results.
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract model names and metrics
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] * 100 for model in models]
    precisions = [results[model]['macro_precision'] * 100 for model in models]
    recalls = [results[model]['macro_recall'] * 100 for model in models]
    f1_scores = [results[model]['macro_f1'] * 100 for model in models]
    
    # Create comparison bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#2ecc71')
    ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#3498db')
    ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#e74c3c')
    ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#f39c12')
    
    ax.set_ylabel('Score (%)', fontsize=12, weight='bold')
    ax.set_title('Model Performance Comparison: PDA vs Baselines', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for i, model in enumerate(models):
        ax.text(i - 1.5*width, accuracies[i] + 2, f'{accuracies[i]:.1f}%',
                ha='center', va='bottom', fontsize=8)
        ax.text(i - 0.5*width, precisions[i] + 2, f'{precisions[i]:.1f}%',
                ha='center', va='bottom', fontsize=8)
        ax.text(i + 0.5*width, recalls[i] + 2, f'{recalls[i]:.1f}%',
                ha='center', va='bottom', fontsize=8)
        ax.text(i + 1.5*width, f1_scores[i] + 2, f'{f1_scores[i]:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Model comparison saved to: {output_dir}/model_comparison.png")
    plt.close()
    
    # Create per-class performance for PDA
    pda_metrics = results['PDA (Ours)']['class_metrics']
    classes = list(pda_metrics.keys())
    class_precisions = [pda_metrics[c]['precision'] * 100 for c in classes]
    class_recalls = [pda_metrics[c]['recall'] * 100 for c in classes]
    class_f1s = [pda_metrics[c]['f1_score'] * 100 for c in classes]
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax.bar(x - width, class_precisions, width, label='Precision', color='#3498db')
    ax.bar(x, class_recalls, width, label='Recall', color='#e74c3c')
    ax.bar(x + width, class_f1s, width, label='F1-Score', color='#f39c12')
    
    ax.set_ylabel('Score (%)', fontsize=12, weight='bold')
    ax.set_title('PDA Per-Class Performance', fontsize=14, weight='bold')
    ax.set_xticks(x)
    # Shorten labels for display
    short_labels = [c.replace(' (Wenckebach)', '').replace(' (Complete Heart Block)', '') 
                   for c in classes]
    ax.set_xticklabels(short_labels, fontsize=10, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pda_per_class.png", dpi=300, bbox_inches='tight')
    print(f"Per-class performance saved to: {output_dir}/pda_per_class.png")
    plt.close()


if __name__ == "__main__":
    import os
    
    # Create visualizations directory
    viz_dir = "/home/claude/av_block_pda_project/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create state diagram
    print("Creating PDA state diagram...")
    create_pda_state_diagram()
    
    # Create results visualizations
    print("\nCreating results visualizations...")
    results_path = "/home/claude/av_block_pda_project/results/synthetic_ecg_dataset_results.json"
    create_results_visualization(results_path)
    
    print("\n" + "="*60)
    print("All visualizations created successfully!")
    print("="*60)
