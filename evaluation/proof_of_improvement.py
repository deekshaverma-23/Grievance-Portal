import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

v1 = pd.read_csv('RAG_Final_Optimized_Report.csv')
v2 = pd.read_csv('local_results_FINAL.csv')

metrics = ['faithfulness', 'context_precision', 'context_recall', 'answer_relevancy']

comparison = pd.DataFrame({
    'Metric': [m.replace('_', ' ').title() for m in metrics],
    'Baseline (V1)': v1[metrics].mean().values,
    'Optimized (V2)': v2[metrics].mean().values
})

comparison['Improvement (%)'] = ((comparison['Optimized (V2)'] - comparison['Baseline (V1)']) / comparison['Baseline (V1)'] * 100).round(1)

comparison.to_csv('Optimization_Summary.csv', index=False)

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, comparison['Baseline (V1)'], width, label='V1: Baseline (k=5)', color='salmon')
ax.bar(x + width/2, comparison['Optimized (V2)'], width, label='V2: Optimized (k=3)', color='skyblue')

ax.set_title('RAG Optimization Audit: Baseline vs Optimized')
ax.set_ylabel('Ragas Score (0-1)')
ax.set_xticks(x)
ax.set_xticklabels(comparison['Metric'])
ax.legend()
plt.savefig('optimization_proof.png')

print("Summary Table Created!")
print(comparison)