import matplotlib.pyplot as plt
import numpy as np

curvatures = [0.5, 1.0, 2.0]
color_scores = [np.float64(0.11281645108829252), np.float64(0.11281645108829252), np.float64(0.11281645108829252)]
shape_scores = [np.float64(0.023410887284080674), np.float64(0.023410887284080674), np.float64(0.021892289548668488)]
texture_scores = [np.float64(0.12104727348426088), np.float64(0.11837721509379519), np.float64(0.12068782119044236)]
avg_scores = [np.float64(0.08575820395221136), np.float64(0.0848681844887228), np.float64(0.08513218727580112)]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(curvatures, color_scores, 'o-', label='Color', color='#e74c3c', linewidth=2)
ax.plot(curvatures, shape_scores, 's-', label='Shape', color='#3498db', linewidth=2)
ax.plot(curvatures, texture_scores, '^-', label='Texture', color='#2ecc71', linewidth=2)
ax.plot(curvatures, avg_scores, 'D--', label='Average', color='#8e44ad', linewidth=2.5)

ax.set_xlabel('Curvature $c$', fontsize=14)
ax.set_ylabel('BLIP-VQA Score', fontsize=14)
ax.set_title('Curvature Sensitivity Analysis', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('paper_results_quick_v2/figures/curvature_plot.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"Curvature plot saved to paper_results_quick_v2/figures/curvature_plot.pdf")
