import re

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fe.config import (
    CENTRALITY_WINDOW_FUNS,
    CENTRALITY_WINDOW_SIZES,
    CUTOFF_FREQUENCIES,
)

# from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
# from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# from sklearn_extra.cluster import KMedoids
# import gower


__all__ = (
    "add_centrality_window",
    "add_dominant_frequencies",
    "add_pca",
    "add_signal_cutoff",
)

# def gower_distance(df: pd.DataFrame):
#     gower_dist_matrix = gower.gower_matrix(df)
#     return gower_dist_matrix[:, 0]

# def add_gower_distance(df: pd.DataFrame, feature_columns: List[str]):
#     gower_distances = gower_distance(df[feature_columns])
#     df['gower_distance'] = np.nan
#     df.loc[df.index, 'gower_distance'] = gower_distances

# def add_clustering(df: pd.DataFrame, feature_columns: List[str], n_clusters: int = 5):
#
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     df['kmeans_cluster'] = kmeans.fit_predict(df[feature_columns])
#
#     kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
#     df['kmedoids_cluster'] = kmedoids.fit_predict(df[feature_columns])
#
#     linkage_matrix = linkage(df[feature_columns], method='ward')
#     df['div_cluster'] = np.nan
#     df.loc[df.index, 'div_cluster'] = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
#
#     spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
#     df['subspace_cluster'] = np.nan
#     df.loc[df.index, 'subspace_cluster'] = spectral.fit_predict(df[feature_columns])
#
#     agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
#     df['agg_cluster'] = np.nan
#     df.loc[df.index, 'agg_cluster'] = agg_clustering.fit_predict(df[feature_columns])
#


def get_fun_name(fun) -> str:
    return re.findall(r"function (\w*) at", str(fun)).pop()


def butterworth_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


def add_pca(
    df: pd.DataFrame, feature_columns: list[str], n_components: int = 15
) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(scaled_features)

    # Create a DataFrame for the PCA components
    pca_columns = [f"pca_{n_components}_component_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_components, columns=pca_columns)

    # Concatenate the PCA components with the original DataFrame
    df = pd.concat([df, pca_df], axis="columns")

    return df


def add_centrality_window(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Add centrality features to the DataFrame
    based on windows sizes
    """

    def rerturn_centrality_window_features(
        column: pd.Series, window: int
    ) -> list[pd.Series]:
        return [
            pd.Series(
                column.rolling(window=window).apply(fun),
                name=f"{column.name}_{get_fun_name(fun)}_{window}",
            )
            for fun in CENTRALITY_WINDOW_FUNS
        ]

    new_features = []
    for window in CENTRALITY_WINDOW_SIZES:
        for column in feature_columns:
            new_features.extend(rerturn_centrality_window_features(df[column], window))
    return pd.concat([df, pd.concat(new_features, axis=1)], axis=1)


def add_signal_cutoff(
    df: pd.DataFrame, feature_columns: list[str], fs: int = 100
) -> pd.DataFrame:
    for cutoff in CUTOFF_FREQUENCIES:
        for column in feature_columns:
            # TODO check if freq ?
            df[f"{column}_filtered_{cutoff}Hz"] = butterworth_filter(
                df[column], cutoff, fs
            )
    return df


def dominant_frequencies(df: pd.DataFrame, window_size: int, fs: int = 100):
    freqs = np.fft.fftfreq(window_size, d=1 / fs)
    half_freqs = freqs[: window_size // 2]

    def freq_with_max_amplitude(window):
        if len(window) < window_size:
            return np.nan
        fft_vals = np.fft.fft(window)
        half_fft_vals = np.abs(fft_vals[: window_size // 2])
        dominant_freq = half_freqs[np.argmax(half_fft_vals)]
        return dominant_freq

    return df.rolling(window=window_size).apply(freq_with_max_amplitude, raw=True)


def add_dominant_frequencies(
    df: pd.DataFrame, feature_columns: list[str], fs: int = 100
) -> pd.DataFrame:
    for window in CENTRALITY_WINDOW_SIZES:
        for column in feature_columns:
            df[f"{column}_dominant_freq_{window}"] = dominant_frequencies(
                df[column], window, fs
            )
    return df
