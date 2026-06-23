import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from src.utils import ensure_dir, OUTPUT_DIR


class DataAnalyzer:
    def __init__(self, images, labels, class_names, output_dir=None):
        self.images = images
        self.labels = labels
        self.class_names = class_names
        self.output_dir = output_dir or str(OUTPUT_DIR / "analysis")
        ensure_dir(self.output_dir)

    def plot_class_distribution(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("husl", len(self.class_names))
        count_list = [counts[list(unique).index(i)] if i in unique else 0 for i in range(len(self.class_names))]
        bars = plt.bar(range(len(self.class_names)), count_list, color=colors)
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        for bar, count in zip(bars, count_list):
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     str(count), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/class_distribution.png", dpi=150)
        plt.close()

    def plot_sample_images(self, n_per_class=5):
        fig, axes = plt.subplots(len(self.class_names), n_per_class,
                                 figsize=(n_per_class * 3, len(self.class_names) * 3))
        for idx, class_name in enumerate(self.class_names):
            mask = self.labels == idx
            class_images = self.images[mask]
            for j in range(n_per_class):
                ax = axes[idx][j]
                if j < len(class_images):
                    ax.imshow(class_images[j])
                ax.axis('off')
                if j == 0:
                    ax.set_title(class_name, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/sample_images.png", dpi=150)
        plt.close()

    def compute_channel_statistics(self):
        channel_names = ['Red', 'Green', 'Blue']
        stats_data = []
        for c, name in enumerate(channel_names):
            channel = self.images[:, :, :, c].flatten().astype(float)
            stats_data.append({
                'Channel': name,
                'Mean': np.mean(channel),
                'Std': np.std(channel),
                'Min': np.min(channel),
                'Max': np.max(channel),
                'Median': np.median(channel),
                'Skewness': float(stats.skew(channel)),
                'Kurtosis': float(stats.kurtosis(channel))
            })
        df = pd.DataFrame(stats_data)
        df.to_csv(f"{self.output_dir}/channel_statistics.csv", index=False)
        return df

    def compute_per_class_statistics(self):
        stats_data = []
        for idx, class_name in enumerate(self.class_names):
            mask = self.labels == idx
            class_imgs = self.images[mask].astype(float)
            for c, ch_name in enumerate(['Red', 'Green', 'Blue']):
                channel = class_imgs[:, :, :, c].flatten()
                stats_data.append({
                    'Class': class_name,
                    'Channel': ch_name,
                    'Mean': np.mean(channel),
                    'Std': np.std(channel),
                    'Skewness': float(stats.skew(channel)),
                    'Kurtosis': float(stats.kurtosis(channel))
                })
        df = pd.DataFrame(stats_data)

        pivot_mean = df.pivot(index='Class', columns='Channel', values='Mean')
        plt.figure(figsize=(10, 6))
        pivot_mean.plot(kind='bar', figsize=(10, 6))
        plt.title("Mean Pixel Value per Channel per Class")
        plt.ylabel("Mean Pixel Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/per_class_channel_stats.png", dpi=150)
        plt.close()

        df.to_csv(f"{self.output_dir}/per_class_statistics.csv", index=False)
        return df

    def plot_correlation_matrix(self, features, feature_names=None):
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(features.shape[1])]

        n_features = min(30, features.shape[1])
        selected_features = features[:, :n_features]
        selected_names = feature_names[:n_features]

        corr_matrix = np.corrcoef(selected_features.T)

        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, xticklabels=selected_names, yticklabels=selected_names,
                    cmap='coolwarm', center=0, square=True, linewidths=0.5)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_matrix.png", dpi=150)
        plt.close()

        return corr_matrix

    def plot_pca_scatter(self, features, labels):
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        plt.figure(figsize=(10, 8))
        colors = sns.color_palette("husl", len(self.class_names))
        for idx, class_name in enumerate(self.class_names):
            mask = labels == idx
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=[colors[idx]], label=class_name, alpha=0.6, s=20)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.title("PCA Visualization of Feature Space")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pca_scatter.png", dpi=150)
        plt.close()

        return pca

    def plot_pixel_distribution(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        colors_rgb = ['red', 'green', 'blue']

        for idx, class_name in enumerate(self.class_names):
            ax = axes[idx // 3][idx % 3]
            mask = self.labels == idx
            class_imgs = self.images[mask]
            for c, color in enumerate(colors_rgb):
                ax.hist(class_imgs[:, :, :, c].flatten(), bins=50, alpha=0.5,
                        color=color, label=color.upper(), density=True)
            ax.set_title(class_name)
            ax.set_xlabel("Pixel Value")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pixel_distributions.png", dpi=150)
        plt.close()

    def run_full_analysis(self, features=None, feature_names=None):
        print("Running preliminary data analysis...")

        print("  - Plotting class distribution...")
        self.plot_class_distribution()

        print("  - Plotting sample images...")
        self.plot_sample_images()

        print("  - Computing channel statistics...")
        channel_stats = self.compute_channel_statistics()
        print(channel_stats.to_string(index=False))

        print("  - Computing per-class statistics...")
        self.compute_per_class_statistics()

        print("  - Plotting pixel distributions...")
        self.plot_pixel_distribution()

        if features is not None:
            print("  - Plotting correlation matrix...")
            self.plot_correlation_matrix(features, feature_names)

            print("  - Plotting PCA scatter...")
            self.plot_pca_scatter(features, self.labels)

        print(f"  Analysis complete. Plots saved to {self.output_dir}/")
        return channel_stats
