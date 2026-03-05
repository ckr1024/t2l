import matplotlib.pyplot as plt
import numpy as np

curvatures = [0.5, 1.0, 2.0]
color_scores = [np.float64(0.4028573092888109), np.float64(0.4015244628069922), np.float64(0.6003276043455117)]
shape_scores = [np.float64(0.4564258272293955), np.float64(0.45888524372130635), np.float64(0.32200521647464486)]
texture_scores = [np.float64(0.20437021555844695), np.float64(0.06596320550888776), np.float64(0.20513048118446023)]
avg_scores = [np.float64(0.35455111735888445), np.float64(0.3087909706790621), np.float64(0.37582110066820557)]

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
plt.savefig('./paper_results/figures/curvature_plot.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"Curvature plot saved to ./paper_results/figures/curvature_plot.pdf")
