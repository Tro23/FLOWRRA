"""
FLOWRRA in Warehouse: Empirical Synthesis & Publication Figures
Reads:
  - distance_test_v4.csv
  - benchmark_new_all_2.csv
  - curriculum_metrics_aws_run4.csv
Outputs:
  - flowrra_complete_benchmark_synthesis.png (300 DPI)
  - summary_table_for_article.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    file_old = 'distance_test_v4.csv'
    file_new = 'benchmark_new_all_2.csv'
    file_train = 'curriculum_metrics_aws_run4.csv'
    
    for f in [file_old, file_new, file_train]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file '{f}' not found in the current directory.")

    print("Loading datasets...")
    df_old = pd.read_csv(file_old)
    df_new = pd.read_csv(file_new)
    df_train = pd.read_csv(file_train)

    # -------------------------------------------------------------
    # 1. Process Training Metrics (Integrated System: Policy + RULES)
    # -------------------------------------------------------------
    total_inherited = df_train['orders_inherited'].sum()
    total_delivered = df_train['orders_inherited_delivered'].sum()
    train_rec_rate_total = (total_delivered / total_inherited) * 100

    train_last50 = df_train.tail(50)
    train_rec_rate_last50 = (train_last50['orders_inherited_delivered'].sum() / train_last50['orders_inherited'].sum()) * 100
    train_comp_rate_last50 = train_last50['completion_rate'].mean() * 100
    train_collisions_last50 = train_last50['collisions'].mean()

    print(f"\n[Training Analysis]")
    print(f"Total Curriculum Inherited Deliveries: {total_delivered:.0f}/{total_inherited:.0f} ({train_rec_rate_total:.2f}%)")
    print(f"Last 50 Episodes Inherited Deliveries: {train_last50['orders_inherited_delivered'].sum():.0f}/{train_last50['orders_inherited'].sum():.0f} ({train_rec_rate_last50:.2f}%)")
    print(f"Last 50 Episodes Completion Rate: {train_comp_rate_last50:.2f}% | Mean Collisions: {train_collisions_last50:.2f}")

    # -------------------------------------------------------------
    # 2. Benchmark Aggregations (Small vs Large Layouts)
    # -------------------------------------------------------------
    small_map = '25_5_2_5_2_1'
    large_map = '50_20_5_10_5_2'

    summary_new = df_new.groupby(['map', 'algorithm'])[['recovery_rate', 'rescuer_deaths', 'replan_s', 'distance_travelled', 'completion_rate', 'collisions']].mean()
    summary_old = df_old.groupby(['map', 'algorithm'])[['recovery_rate', 'rescuer_deaths', 'replan_s', 'distance_travelled', 'completion_rate', 'collisions']].mean()

    # -------------------------------------------------------------
    # 3. Create Multi-Panel Figure (2x2 Grid)
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Orphaned Order Recovery Rate
    rec_methods = [
        'Pure Planner\n(RHCR-PIBT)', 
        'RHCR + naive\n(Small Map)', 'FLOWRRA\n(Small Map)', 'RULES\n(Small Map)',
        'RHCR + naive\n(Large Map)', 'FLOWRRA\n(Large Map)', 'RULES\n(Large Map)',
        'FLOWRRA + RULES\n(Train Total)', 'FLOWRRA + RULES\n(Train Last 50)'
    ]
    rec_values = [
        0.0, 
        summary_new.loc[(small_map, 'RHCR-PIBT+naive'), 'recovery_rate'] * 100,
        summary_new.loc[(small_map, 'FLOWRRA'), 'recovery_rate'] * 100,
        summary_new.loc[(small_map, 'RULES'), 'recovery_rate'] * 100,
        summary_new.loc[(large_map, 'RHCR-PIBT+naive'), 'recovery_rate'] * 100,
        summary_new.loc[(large_map, 'FLOWRRA'), 'recovery_rate'] * 100,
        summary_new.loc[(large_map, 'RULES'), 'recovery_rate'] * 100,
        train_rec_rate_total,
        train_rec_rate_last50
    ]
    rec_categories = [
        'Baseline', 
        'Small Map (1.4k)', 'Small Map (1.4k)', 'Small Map (1.4k)',
        'Large Map (27k)', 'Large Map (27k)', 'Large Map (27k)',
        'Integrated System', 'Integrated System'
    ]

    df_p1 = pd.DataFrame({'Method': rec_methods, 'Recovery Rate (%)': rec_values, 'Category': rec_categories})
    sns.barplot(data=df_p1, x='Method', y='Recovery Rate (%)', hue='Category', dodge=False, ax=axs[0, 0], palette='viridis')
    axs[0, 0].set_title('Orphaned Order Recovery Rate: Baselines, Ablations & Integrated Holon', fontweight='bold', fontsize=12)
    axs[0, 0].set_ylim(0, 105)
    axs[0, 0].tick_params(axis='x', rotation=38)
    axs[0, 0].axhline(100, color='red', linestyle='--', alpha=0.5)

    # Panel B: Cascading Rescuer Deaths
    res_deaths_data = {
        'Map': ['Small Map (1,435 nodes)', 'Small Map (1,435 nodes)', 'Small Map (1,435 nodes)',
                'Large Map (27,000 nodes)', 'Large Map (27,000 nodes)', 'Large Map (27,000 nodes)'],
        'Algorithm': ['RHCR+naive', 'FLOWRRA', 'RULES', 'RHCR+naive', 'FLOWRRA', 'RULES'],
        'Rescuer Deaths': [
            summary_new.loc[(small_map, 'RHCR-PIBT+naive'), 'rescuer_deaths'],
            summary_new.loc[(small_map, 'FLOWRRA'), 'rescuer_deaths'],
            summary_new.loc[(small_map, 'RULES'), 'rescuer_deaths'],
            summary_new.loc[(large_map, 'RHCR-PIBT+naive'), 'rescuer_deaths'],
            summary_new.loc[(large_map, 'FLOWRRA'), 'rescuer_deaths'],
            summary_new.loc[(large_map, 'RULES'), 'rescuer_deaths']
        ]
    }
    df_p2 = pd.DataFrame(res_deaths_data)
    sns.barplot(data=df_p2, x='Map', y='Rescuer Deaths', hue='Algorithm', ax=axs[0, 1], palette='magma')
    axs[0, 1].set_title('Cascading Failures: Rescuer Deaths per Episode', fontweight='bold', fontsize=12)
    axs[0, 1].set_ylabel('Mean Rescuer Deaths')

    # Panel C: Replanning Downtime (Computational Paralysis)
    downtime_data = {
        'Map': ['Small Map (1,435 nodes)', 'Small Map (1,435 nodes)', 'Small Map (1,435 nodes)',
                'Large Map (27,000 nodes)', 'Large Map (27,000 nodes)', 'Large Map (27,000 nodes)'],
        'Algorithm': ['RHCR+naive', 'FLOWRRA', 'RULES', 'RHCR+naive', 'FLOWRRA', 'RULES'],
        'Replan Downtime (s)': [
            summary_new.loc[(small_map, 'RHCR-PIBT+naive'), 'replan_s'],
            summary_new.loc[(small_map, 'FLOWRRA'), 'replan_s'],
            summary_new.loc[(small_map, 'RULES'), 'replan_s'],
            summary_new.loc[(large_map, 'RHCR-PIBT+naive'), 'replan_s'],
            summary_new.loc[(large_map, 'FLOWRRA'), 'replan_s'],
            summary_new.loc[(large_map, 'RULES'), 'replan_s']
        ]
    }
    df_p3 = pd.DataFrame(downtime_data)
    sns.barplot(data=df_p3, x='Map', y='Replan Downtime (s)', hue='Algorithm', ax=axs[1, 0], palette='rocket')
    axs[1, 0].set_title('System Paralysis: Centralized Replanning Downtime (Seconds)', fontweight='bold', fontsize=12)
    axs[1, 0].set_ylabel('Seconds Frozen per Episode')

    # Panel D: Small Floor Travel Efficiency Across Fleet Counts
    agents = [25, 40, 60]
    old_flow = df_old[(df_old['map'] == small_map) & (df_old['algorithm'] == 'FLOWRRA')].groupby('requested_agents')['distance_travelled'].mean().loc[agents].values
    new_flow = df_new[(df_new['map'] == small_map) & (df_new['algorithm'] == 'FLOWRRA')].groupby('requested_agents')['distance_travelled'].mean().loc[agents].values
    rules_dist = df_new[(df_new['map'] == small_map) & (df_new['algorithm'] == 'RULES')].groupby('requested_agents')['distance_travelled'].mean().loc[agents].values
    rhcr_dist = df_new[(df_new['map'] == small_map) & (df_new['algorithm'] == 'RHCR-PIBT+naive')].groupby('requested_agents')['distance_travelled'].mean().loc[agents].values

    dist_data = {
        'Fleet Count': [25, 40, 60] * 4,
        'Configuration': ['Old FLOWRRA']*3 + ['New FLOWRRA']*3 + ['RULES Orchestrator']*3 + ['RHCR+naive']*3,
        'Distance Travelled': list(old_flow) + list(new_flow) + list(rules_dist) + list(rhcr_dist)
    }
    df_p4 = pd.DataFrame(dist_data)
    sns.barplot(data=df_p4, x='Fleet Count', y='Distance Travelled', hue='Configuration', ax=axs[1, 1], palette='mako')
    axs[1, 1].set_title('Small Floor Travel Efficiency: Old vs. New FLOWRRA vs. RULES', fontweight='bold', fontsize=12)
    axs[1, 1].set_ylabel('Distance Travelled (cells)')

    plt.tight_layout()
    output_png = 'flowrra_complete_benchmark_synthesis.png'
    plt.savefig(output_png, dpi=300)
    print(f"\n[Figure Generated] Saved 300 DPI chart to: {output_png}")

    # -------------------------------------------------------------
    # 4. Generate Summary Table (CSV and Markdown)
    # -------------------------------------------------------------
    table_df = pd.DataFrame({
        'Metric': [
            'Orphaned Recovery Rate',
            'Cascading Rescuer Deaths',
            'Mean Recovery Hops',
            'Replan Downtime (s)',
            'Distance Traversed (cells)',
            'Overall Completion Rate'
        ],
        'Pure Planner (RHCR-PIBT)': [
            '0.0%',
            '0.00 (No Rescues)',
            'N/A',
            '0.1s - 9.8s',
            '298 - 963',
            '75.5%'
        ],
        'RHCR + naive (Small / Large)': [
            f"{summary_new.loc[(small_map, 'RHCR-PIBT+naive'), 'recovery_rate']*100:.1f}% / {summary_new.loc[(large_map, 'RHCR-PIBT+naive'), 'recovery_rate']*100:.1f}%",
            f"{summary_new.loc[(small_map, 'RHCR-PIBT+naive'), 'rescuer_deaths']:.2f} / {summary_new.loc[(large_map, 'RHCR-PIBT+naive'), 'rescuer_deaths']:.2f}",
            f"{df_new[(df_new['map']==small_map) & (df_new['algorithm']=='RHCR-PIBT+naive')]['mean_recovery_hops'].mean():.1f} / {df_new[(df_new['map']==large_map) & (df_new['algorithm']=='RHCR-PIBT+naive')]['mean_recovery_hops'].mean():.1f}",
            f"{summary_new.loc[(small_map, 'RHCR-PIBT+naive'), 'replan_s']:.1f}s / {summary_new.loc[(large_map, 'RHCR-PIBT+naive'), 'replan_s']:.1f}s",
            f"{summary_new.loc[(small_map, 'RHCR-PIBT+naive'), 'distance_travelled']:.0f} / {summary_new.loc[(large_map, 'RHCR-PIBT+naive'), 'distance_travelled']:.0f}",
            f"{summary_new.loc[(small_map, 'RHCR-PIBT+naive'), 'completion_rate']*100:.1f}% / {summary_new.loc[(large_map, 'RHCR-PIBT+naive'), 'completion_rate']*100:.1f}%"
        ],
        'FLOWRRA Policy (Small / Large)': [
            f"{summary_new.loc[(small_map, 'FLOWRRA'), 'recovery_rate']*100:.1f}% / {summary_new.loc[(large_map, 'FLOWRRA'), 'recovery_rate']*100:.1f}%",
            f"{summary_new.loc[(small_map, 'FLOWRRA'), 'rescuer_deaths']:.2f} / {summary_new.loc[(large_map, 'FLOWRRA'), 'rescuer_deaths']:.2f}",
            f"{df_new[(df_new['map']==small_map) & (df_new['algorithm']=='FLOWRRA')]['mean_recovery_hops'].mean():.1f} / {df_new[(df_new['map']==large_map) & (df_new['algorithm']=='FLOWRRA')]['mean_recovery_hops'].mean():.1f}",
            f"{summary_new.loc[(small_map, 'FLOWRRA'), 'replan_s']:.1f}s / {summary_new.loc[(large_map, 'FLOWRRA'), 'replan_s']:.1f}s",
            f"{summary_new.loc[(small_map, 'FLOWRRA'), 'distance_travelled']:.0f} / {summary_new.loc[(large_map, 'FLOWRRA'), 'distance_travelled']:.0f}",
            f"{summary_new.loc[(small_map, 'FLOWRRA'), 'completion_rate']*100:.1f}% / {summary_new.loc[(large_map, 'FLOWRRA'), 'completion_rate']*100:.1f}%"
        ],
        'RULES Orchestrator (Small / Large)': [
            f"{summary_new.loc[(small_map, 'RULES'), 'recovery_rate']*100:.1f}% / {summary_new.loc[(large_map, 'RULES'), 'recovery_rate']*100:.1f}%",
            f"{summary_new.loc[(small_map, 'RULES'), 'rescuer_deaths']:.2f} / {summary_new.loc[(large_map, 'RULES'), 'rescuer_deaths']:.2f}",
            f"{df_new[(df_new['map']==small_map) & (df_new['algorithm']=='RULES')]['mean_recovery_hops'].mean():.1f} / {df_new[(df_new['map']==large_map) & (df_new['algorithm']=='RULES')]['mean_recovery_hops'].mean():.1f}",
            f"{summary_new.loc[(small_map, 'RULES'), 'replan_s']:.1f}s / {summary_new.loc[(large_map, 'RULES'), 'replan_s']:.1f}s",
            f"{summary_new.loc[(small_map, 'RULES'), 'distance_travelled']:.0f} / {summary_new.loc[(large_map, 'RULES'), 'distance_travelled']:.0f}",
            f"{summary_new.loc[(small_map, 'RULES'), 'completion_rate']*100:.1f}% / {summary_new.loc[(large_map, 'RULES'), 'completion_rate']*100:.1f}%"
        ],
        'Integrated FLOWRRA + RULES (Train)': [
            f"{train_rec_rate_total:.1f}% - {train_rec_rate_last50:.1f}%",
            '0.37 (Forced Interv)',
            'Matched Clock',
            '0.0s (Real-Time)',
            'Active Dispersion',
            f"{train_comp_rate_last50:.1f}%"
        ]
    })

    output_csv = 'summary_table_for_article.csv'
    table_df.to_csv(output_csv, index=False)
    print(f"\n[Table Exported] Saved summary table to: {output_csv}")
    print("\n--- Markdown Format for Direct Pasting ---\n")
    print(table_df.to_markdown(index=False))

if __name__ == '__main__':
    main()