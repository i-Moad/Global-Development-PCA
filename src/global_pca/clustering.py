import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def evaluate_cluster_counts(
    scores: pd.DataFrame,
    component_columns: list[str],
    k_values: range = range(2, 9),
    random_state: int = 42,
    n_init: int = 20,
) -> pd.DataFrame:
  """
  Evaluate candidate K-Means cluster counts using inertia and silhouette scores.
  """

  if scores.empty:
    raise ValueError(
      "Scores DataFrame cannot be empty."
    )

  if not component_columns:
    raise ValueError(
      "At least one principal component is required."
    )

  if len(set(component_columns)) != len(component_columns):
    raise ValueError(
      "Component columns must be unique."
    )

  missing_columns = set(component_columns) - set(scores.columns)
    
  if missing_columns:
    raise ValueError(
      f"Missing indicator columns: {sorted(missing_columns)}"
    )

  if not all(
    pd.api.types.is_numeric_dtype(scores[column])
    for column in component_columns
  ):
    raise ValueError(
      "Component scores must be numeric."
    )

  data = scores[component_columns].to_numpy(dtype=float)

  if not np.isfinite(data).all():
    raise ValueError(
      "Component scores must conyain finite values."
    )

  n_observations = len(data)

  if n_observations < 3:
    raise ValueError(
      "At least three observations are required."
    )

  if not k_values:
    raise ValueError(
      "At least one candidate cluster count is required."
    )

  if any(
    not isinstance(k, (int, np.integer))
    or isinstance(k, (bool, np.bool))
    or k < 2
    or k >= n_observations
    for k in k_values
  ):
    raise ValueError(
      "Each cluster count must be an integer between "
      f"2 and {n_observations - 1}."
    )

  results = []

  for k in k_values:
    model = KMeans(
      n_clusters=k,
      random_state=random_state,
      n_init=n_init,
    )

    labels = model.fit_predict(data)

    score = silhouette_score(data, labels)

    results.append({
      "n_clusters": k,
      "inertia": model.inertia_,
      "silhouette_score": score,
    })

  return pd.DataFrame(results)

def fit_kmeans(
    scores: pd.DataFrame,
    component_columns: list[str],
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 20,
) -> tuple[pd.Series, KMeans]:
  """
  Fit K-Means and return cluster assignments and the fitted model.
  """

  if scores.empty:
    raise ValueError(
      "Scores DataFrame cannot be empty."
    )

  if not component_columns:
    raise ValueError(
      "At least one principal component is required."
    )

  if len(component_columns) != len(set(component_columns)):
    raise ValueError(
      "Component columns must be unique."
    )

  missing_columns = set(component_columns) - set(scores.columns)
      
  if missing_columns:
    raise ValueError(
      f"Missing indicator columns: {sorted(missing_columns)}"
    )

  if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
    raise ValueError(
      "Number of clusters must be an integer."
    )

  if n_clusters < 2 or n_clusters >= len(scores):
    raise ValueError(
      "Number of clusters must be between 2 and "
      "n_observations - 1."
    )

  data = scores[component_columns].to_numpy(dtype=float)

  if not np.isfinite(data).all():
    raise ValueError(
      "Component scores must contain finite values."
    )

  model = KMeans(
    n_clusters=n_clusters,
    random_state=random_state,
    n_init=n_init,
  )

  labels = model.fit_predict(data)

  assignments = pd.Series(
    labels,
    index=scores.index,
    name="cluster",
  )

  return assignments, model
