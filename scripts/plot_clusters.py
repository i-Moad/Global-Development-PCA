import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.artist import Artist

from global_pca.clustering import fit_kmeans
from global_pca.data.pipeline import INDICATOR_CODES
from global_pca.analysis import plot_cluster_scatter
from global_pca.pca import (
    calculate_component_scores,
    calculate_covariance_matrix,
    calculate_eigenvalues_eigenvectors,
    standardize_data,
)

CLUSTER_NAMES = {
  0: "Intermediate development",
  1: "Lower development",
  2: "Higher development",
}

data = pd.read_csv("data/processed/country_development.csv")

standardized_data = standardize_data(
  data=data,
  indicator_codes=INDICATOR_CODES,
)

covariance_matrix = calculate_covariance_matrix(
  standardized_data=standardized_data,
)

eigenvalues, eigenvectors = calculate_eigenvalues_eigenvectors(
  covariance_matrix=covariance_matrix,
)

eigenvectors["PC1"] *= -1

scores = calculate_component_scores(
  standardized_data=standardized_data,
  eigenvectors=eigenvectors,
)

scores.index = data["country"].astype(str)

assignments, model = fit_kmeans(
  scores=scores,
  component_columns=["PC1", "PC2", "PC3", "PC4"],
  n_clusters=3,
)

fig, ax = plot_cluster_scatter(
    scores=scores,
    assignments=assignments,
    cluster_names=CLUSTER_NAMES,
)

for country, position in scores.iterrows():
    ax.annotate(
        str(country),
        (position["PC1"], position["PC2"]),
        fontsize=6,
        alpha=0.7,
    )

fig.savefig(
    "reports/figures/pca_clusters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
print("Cluster visualization saved successfully.")