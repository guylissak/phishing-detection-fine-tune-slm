"""
Visualization functions for phishing email detection experiment.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(cm, title: str, save_path: Path = None, ax=None):
    """
    Plot a single confusion matrix.

    Args:
        cm: Confusion matrix array
        title: Plot title
        save_path: Optional path to save the figure
        ax: Optional matplotlib axis (for subplots)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        standalone = True
    else:
        fig = ax.figure
        standalone = False

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Count', fontsize=10)

    classes = ["Safe", "Phishing"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_title(title, fontsize=11)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')

    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


def plot_comparison_confusion_matrices(results_list: list, titles: list,
                                        save_path: Path = None):
    """
    Plot multiple confusion matrices side by side.

    Args:
        results_list: List of result dictionaries containing 'confusion_matrix' and 'accuracy'
        titles: List of titles for each matrix
        save_path: Optional path to save the figure
    """
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5))

    if n == 1:
        axes = [axes]

    for ax, results, title in zip(axes, results_list, titles):
        cm = results["confusion_matrix"]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        # Add colorbar for each matrix
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Count', fontsize=9)

        ax.set_title(f'{title}\nAccuracy: {results["accuracy"]:.3f}', fontsize=11)

        classes = ["Safe", "Phishing"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight='bold')

        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

def plot_class_distribution(class_counts, save_path: Path = None):
    """
    Plot class distribution as bar chart and pie chart.

    Args:
        class_counts: pandas Series with class counts
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = ['#2ecc71', '#e74c3c']  # Green for safe, red for phishing

    # Bar chart
    axes[0].bar(class_counts.index, class_counts.values, color=colors)
    axes[0].set_title('Email Class Distribution', fontsize=12)
    axes[0].set_ylabel('Count')
    for i, (idx, val) in enumerate(class_counts.items()):
        axes[0].text(i, val + 50, str(val), ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                colors=colors, explode=(0.02, 0.02))
    axes[1].set_title('Email Class Proportion', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return class_counts.max() / class_counts.min()


# =============================================================================
# TEXT LENGTH ANALYSIS
# =============================================================================

def plot_text_length_analysis(df, save_path: Path = None):
    """
    Plot character and word count distributions by class.

    Args:
        df: DataFrame with 'Email Type', 'char_count', 'word_count' columns
        save_path: Optional path to save the figure
    """
    safe_df = df[df['Email Type'] == 'Safe Email']
    phishing_df = df[df['Email Type'] == 'Phishing Email']

    # Filter outliers for better visualization (99th percentile)
    char_99 = df['char_count'].quantile(0.99)
    word_99 = df['word_count'].quantile(0.99)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Text Length Analysis by Email Type', fontsize=16, fontweight='bold', y=1.02)

    safe_color = '#2ecc71'
    phishing_color = '#e74c3c'

    # Character count histograms - NORMALIZED
    axes[0, 0].hist(safe_df['char_count'], bins=30, alpha=0.6, label='Safe', color=safe_color,
                    density=True, range=(0, char_99), edgecolor='white', linewidth=0.5)
    axes[0, 0].hist(phishing_df['char_count'], bins=30, alpha=0.6, label='Phishing', color=phishing_color,
                    density=True, range=(0, char_99), edgecolor='white', linewidth=0.5)
    axes[0, 0].set_title('Character Count Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Characters')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.85))
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].text(0.98, 0.98, 'Brown = overlap', transform=axes[0, 0].transAxes,
                    ha='right', va='top', fontsize=8, style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Character count box plot
    safe_chars = safe_df['char_count'][safe_df['char_count'] <= char_99]
    phishing_chars = phishing_df['char_count'][phishing_df['char_count'] <= char_99]

    bp1 = axes[0, 1].boxplot([safe_chars, phishing_chars],
                              labels=['Safe', 'Phishing'], patch_artist=True)
    bp1['boxes'][0].set_facecolor(safe_color)
    bp1['boxes'][0].set_alpha(0.7)
    bp1['boxes'][1].set_facecolor(phishing_color)
    bp1['boxes'][1].set_alpha(0.7)
    for median in bp1['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    axes[0, 1].set_title('Character Count Box Plot', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Characters')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Add stats in top-left corner
    safe_char_stats = f"Safe:\n  Med: {safe_chars.median():.0f}\n  Q1: {safe_chars.quantile(0.25):.0f}\n  Q3: {safe_chars.quantile(0.75):.0f}"
    phish_char_stats = f"Phishing:\n  Med: {phishing_chars.median():.0f}\n  Q1: {phishing_chars.quantile(0.25):.0f}\n  Q3: {phishing_chars.quantile(0.75):.0f}"
    axes[0, 1].text(0.02, 0.98, safe_char_stats, transform=axes[0, 1].transAxes, fontsize=9,
                    verticalalignment='top', color=safe_color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0, 1].text(0.02, 0.58, phish_char_stats, transform=axes[0, 1].transAxes, fontsize=9,
                    verticalalignment='top', color=phishing_color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Word count histograms - NORMALIZED
    axes[1, 0].hist(safe_df['word_count'], bins=30, alpha=0.6, label='Safe', color=safe_color,
                    density=True, range=(0, word_99), edgecolor='white', linewidth=0.5)
    axes[1, 0].hist(phishing_df['word_count'], bins=30, alpha=0.6, label='Phishing', color=phishing_color,
                    density=True, range=(0, word_99), edgecolor='white', linewidth=0.5)
    axes[1, 0].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Words')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.85))
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].text(0.98, 0.98, 'Brown = overlap', transform=axes[1, 0].transAxes,
                    ha='right', va='top', fontsize=8, style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Word count box plot
    safe_words = safe_df['word_count'][safe_df['word_count'] <= word_99]
    phishing_words = phishing_df['word_count'][phishing_df['word_count'] <= word_99]

    bp2 = axes[1, 1].boxplot([safe_words, phishing_words],
                              labels=['Safe', 'Phishing'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(safe_color)
    bp2['boxes'][0].set_alpha(0.7)
    bp2['boxes'][1].set_facecolor(phishing_color)
    bp2['boxes'][1].set_alpha(0.7)
    for median in bp2['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    axes[1, 1].set_title('Word Count Box Plot', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Words')
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Add stats in top-left corner
    safe_word_stats = f"Safe:\n  Med: {safe_words.median():.0f}\n  Q1: {safe_words.quantile(0.25):.0f}\n  Q3: {safe_words.quantile(0.75):.0f}"
    phish_word_stats = f"Phishing:\n  Med: {phishing_words.median():.0f}\n  Q1: {phishing_words.quantile(0.25):.0f}\n  Q3: {phishing_words.quantile(0.75):.0f}"
    axes[1, 1].text(0.02, 0.98, safe_word_stats, transform=axes[1, 1].transAxes, fontsize=9,
                    verticalalignment='top', color=safe_color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1, 1].text(0.02, 0.58, phish_word_stats, transform=axes[1, 1].transAxes, fontsize=9,
                    verticalalignment='top', color=phishing_color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    # Print outlier information
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION (beyond 99th percentile)")
    print("=" * 60)
    print(f"\nCharacter count outliers (>{char_99:.0f} chars):")
    print(f"  Safe: {(safe_df['char_count'] > char_99).sum()} emails")
    print(f"  Phishing: {(phishing_df['char_count'] > char_99).sum()} emails")
    print(f"\nWord count outliers (>{word_99:.0f} words):")
    print(f"  Safe: {(safe_df['word_count'] > word_99).sum()} emails")
    print(f"  Phishing: {(phishing_df['word_count'] > word_99).sum()} emails")


# =============================================================================
# TOKEN DISTRIBUTION
# =============================================================================

def plot_token_distribution(df, max_training_tokens: int = 2048, save_path: Path = None):
    """
    Plot token count distribution by class.

    Args:
        df: DataFrame with 'Email Type' and 'token_count' columns
        max_training_tokens: Training limit to mark on plot
        save_path: Optional path to save the figure
    """
    token_99 = df['token_count'].quantile(0.99)

    safe_color = '#2ecc71'
    phishing_color = '#e74c3c'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Token Count Analysis by Email Type', fontsize=16, fontweight='bold', y=1.02)

    safe_tokens = df[df['Email Type'] == 'Safe Email']['token_count']
    phishing_tokens = df[df['Email Type'] == 'Phishing Email']['token_count']

    # Histogram - normalized and limited to 99th percentile
    axes[0].hist(safe_tokens, bins=30, alpha=0.6, label='Safe', color=safe_color,
                 density=True, range=(0, token_99), edgecolor='white', linewidth=0.5)
    axes[0].hist(phishing_tokens, bins=30, alpha=0.6, label='Phishing', color=phishing_color,
                 density=True, range=(0, token_99), edgecolor='white', linewidth=0.5)
    axes[0].axvline(x=max_training_tokens, color='orange', linestyle='--',
                    linewidth=2, label=f'Training limit ({max_training_tokens})')
    axes[0].set_title('Token Count Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tokens')
    axes[0].set_ylabel('Density')
    axes[0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.85))
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].text(0.98, 0.98, 'Brown = overlap', transform=axes[0].transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Box plot - filtered
    safe_tokens_filtered = safe_tokens[safe_tokens <= token_99]
    phishing_tokens_filtered = phishing_tokens[phishing_tokens <= token_99]

    bp = axes[1].boxplot([safe_tokens_filtered, phishing_tokens_filtered],
                          labels=['Safe', 'Phishing'], patch_artist=True)
    bp['boxes'][0].set_facecolor(safe_color)
    bp['boxes'][1].set_facecolor(phishing_color)
    for box in bp['boxes']:
        box.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    axes[1].axhline(y=max_training_tokens, color='orange', linestyle='--',
                    linewidth=2, label=f'Training limit ({max_training_tokens})')
    axes[1].set_title('Token Count by Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Tokens')
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', alpha=0.3)

    # Add stats in top-left corner with colored text
    safe_token_stats = f"Safe:\n  Med: {safe_tokens_filtered.median():.0f}\n  Q1: {safe_tokens_filtered.quantile(0.25):.0f}\n  Q3: {safe_tokens_filtered.quantile(0.75):.0f}"
    phish_token_stats = f"Phishing:\n  Med: {phishing_tokens_filtered.median():.0f}\n  Q1: {phishing_tokens_filtered.quantile(0.25):.0f}\n  Q3: {phishing_tokens_filtered.quantile(0.75):.0f}"
    axes[1].text(0.02, 0.98, safe_token_stats, transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='top', color=safe_color, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].text(0.02, 0.62, phish_token_stats, transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='top', color=phishing_color, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    # Print token limit information
    MAX_MODEL_TOKENS = 131072  # Qwen2.5 context limit
    exceeds_train = len(df[df['token_count'] > max_training_tokens])
    exceeds_model = len(df[df['token_count'] > MAX_MODEL_TOKENS])

    print(f"\nEmails exceeding training limit ({max_training_tokens} tokens): {exceeds_train} ({100*exceeds_train/len(df):.1f}%)")
    print(f"Emails exceeding model limit ({MAX_MODEL_TOKENS} tokens): {exceeds_model}")


# =============================================================================
# WORD CLOUDS
# =============================================================================

def plot_word_clouds(safe_text: str, phishing_text: str, save_path: Path = None):
    """
    Plot word clouds for safe and phishing emails.

    Args:
        safe_text: Combined text from safe emails
        phishing_text: Combined text from phishing emails
        save_path: Optional path to save the figure
    """
    stopwords = set(STOPWORDS)
    stopwords.update(['will', 'hi', 'ect', 'Â', 'â', 's'])

    def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return "rgb(34, 139, 34)"  # Forest green

    def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return "rgb(178, 34, 34)"  # Firebrick red

    safe_wc = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100,
        random_state=42,
        stopwords=stopwords,
        color_func=green_color_func
    ).generate(safe_text)

    phishing_wc = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100,
        random_state=42,
        stopwords=stopwords,
        color_func=red_color_func
    ).generate(phishing_text)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(safe_wc, interpolation='bilinear')
    axes[0].set_title('Safe Emails - Most Common Words', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(phishing_wc, interpolation='bilinear')
    axes[1].set_title('Phishing Emails - Most Common Words', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# =============================================================================
# PHISHING KEYWORD ANALYSIS
# =============================================================================

def plot_phishing_keyword_analysis(df, save_path: Path = None):
    """
    Plot phishing keyword count distribution by class.

    Args:
        df: DataFrame with 'Email Type' and 'phishing_keyword_count' columns
        save_path: Optional path to save the figure
    """
    safe_kw = df[df['Email Type'] == 'Safe Email']['phishing_keyword_count']
    phishing_kw = df[df['Email Type'] == 'Phishing Email']['phishing_keyword_count']

    safe_color = '#2ecc71'
    phishing_color = '#e74c3c'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phishing Keyword Analysis by Email Type', fontsize=16, fontweight='bold', y=1.02)

    # Histogram
    axes[0].hist(safe_kw, bins=range(0, 15), alpha=0.6, label='Safe', color=safe_color,
                 align='left', edgecolor='white', linewidth=0.5)
    axes[0].hist(phishing_kw, bins=range(0, 15), alpha=0.6, label='Phishing', color=phishing_color,
                 align='left', edgecolor='white', linewidth=0.5)
    axes[0].set_title('Keyword Count Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Phishing Keywords')
    axes[0].set_ylabel('Frequency')
    axes[0].text(0.98, 0.98, 'Brown = overlap', transform=axes[0].transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.85))
    axes[0].grid(axis='y', alpha=0.3)

    # Boxplot
    bp = axes[1].boxplot([safe_kw, phishing_kw], labels=['Safe', 'Phishing'], patch_artist=True)
    bp['boxes'][0].set_facecolor(safe_color)
    bp['boxes'][1].set_facecolor(phishing_color)
    for box in bp['boxes']:
        box.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    axes[1].set_title('Keyword Count by Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Phishing Keywords')
    axes[1].grid(axis='y', alpha=0.3)

    # Add stats in top-left corner with colored text
    safe_stats = f"Safe:\n  Med: {safe_kw.median():.0f}\n  Q1: {safe_kw.quantile(0.25):.0f}\n  Q3: {safe_kw.quantile(0.75):.0f}"
    phish_stats = f"Phishing:\n  Med: {phishing_kw.median():.0f}\n  Q1: {phishing_kw.quantile(0.25):.0f}\n  Q3: {phishing_kw.quantile(0.75):.0f}"
    axes[1].text(0.02, 0.98, safe_stats, transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='top', color=safe_color, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].text(0.02, 0.62, phish_stats, transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='top', color=phishing_color, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# =============================================================================
# TRAINING CURVES
# =============================================================================

def plot_training_curves(metrics_callback, save_path: Path = None):
    """
    Plot training and validation loss curves.

    Args:
        metrics_callback: TrainerCallback object with loss history
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training and Validation Loss
    ax1 = axes[0]
    if metrics_callback.train_losses:
        ax1.plot(metrics_callback.train_steps, metrics_callback.train_losses,
                'b-', alpha=0.7, label='Train Loss')

    if metrics_callback.eval_losses:
        ax1.scatter(metrics_callback.eval_steps, metrics_callback.eval_losses,
                   c='red', s=100, zorder=5, label='Val Loss')
        ax1.plot(metrics_callback.eval_steps, metrics_callback.eval_losses, 'r--', alpha=0.5)

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning Rate Schedule
    ax2 = axes[1]
    if metrics_callback.learning_rates:
        ax2.plot(metrics_callback.lr_steps, metrics_callback.learning_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_training_loss_smoothed(metrics_callback, window: int = 10, save_path: Path = None):
    """
    Plot training loss with smoothing.

    Args:
        metrics_callback: TrainerCallback object with loss history
        window: Moving average window size
        save_path: Optional path to save the figure
    """
    if not metrics_callback.train_losses:
        print("No training losses to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Raw loss
    ax.plot(metrics_callback.train_steps, metrics_callback.train_losses,
           'b-', alpha=0.3, label='Train Loss (raw)')

    # Smoothed loss using exponential moving average (starts from beginning)
    window = min(window, len(metrics_callback.train_losses) // 3) or 1
    if len(metrics_callback.train_losses) >= window:
        # Use exponential moving average for smooth curve from the start
        alpha = 2 / (window + 1)
        smoothed = []
        ema = metrics_callback.train_losses[0]
        for loss in metrics_callback.train_losses:
            ema = alpha * loss + (1 - alpha) * ema
            smoothed.append(ema)
        ax.plot(metrics_callback.train_steps, smoothed, 'b-', linewidth=2,
               label=f'Train Loss (smoothed, window={window})')

    # Validation loss
    if metrics_callback.eval_losses:
        ax.scatter(metrics_callback.eval_steps, metrics_callback.eval_losses,
                  c='red', s=100, zorder=5, label='Val Loss')
        ax.plot(metrics_callback.eval_steps, metrics_callback.eval_losses, 'r--', alpha=0.7)

    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# =============================================================================
# METRICS COMPARISON BAR CHARTS
# =============================================================================

def plot_metrics_comparison(results_dict: dict, save_path: Path = None):
    """
    Plot metrics comparison bar chart for multiple methods.

    Args:
        results_dict: Dictionary mapping method names to result dictionaries
        save_path: Optional path to save the figure
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)

    # Extract metrics
    metrics = ['Accuracy', 'Phishing\nPrecision', 'Phishing\nRecall', 'Phishing\nF1']
    values = {}
    for method, results in results_dict.items():
        values[method] = [
            results['accuracy'],
            results['report']['Phishing']['precision'],
            results['report']['Phishing']['recall'],
            results['report']['Phishing']['f1-score']
        ]

    x = np.arange(len(metrics))
    width = 0.8 / n_methods

    colors = ['forestgreen', 'steelblue', 'darkorange', 'purple', 'brown']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        offset = (i - n_methods/2 + 0.5) * width
        bars = ax.bar(x + offset, values[method], width, label=method, color=colors[i % len(colors)])

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_class_metrics_comparison(results_dict: dict, save_path: Path = None):
    """
    Plot both Phishing and Safe class metrics comparison.

    Args:
        results_dict: Dictionary mapping method names to result dictionaries
        save_path: Optional path to save the figure
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    metrics_names = ['Precision', 'Recall', 'F1-Score']
    colors = ['forestgreen', 'steelblue', 'darkorange', 'purple', 'brown']
    width = 0.8 / n_methods
    x = np.arange(len(metrics_names))

    # Phishing class
    ax1 = axes[0]
    for i, method in enumerate(methods):
        results = results_dict[method]
        values = [
            results['report']['Phishing']['precision'],
            results['report']['Phishing']['recall'],
            results['report']['Phishing']['f1-score']
        ]
        offset = (i - n_methods/2 + 0.5) * width
        bars = ax1.bar(x + offset, values, width, label=method, color=colors[i % len(colors)])

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    ax1.set_ylabel('Score')
    ax1.set_title('Phishing Class Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis='y', alpha=0.3)

    # Safe class
    ax2 = axes[1]
    for i, method in enumerate(methods):
        results = results_dict[method]
        values = [
            results['report']['Safe']['precision'],
            results['report']['Safe']['recall'],
            results['report']['Safe']['f1-score']
        ]
        offset = (i - n_methods/2 + 0.5) * width
        bars = ax2.bar(x + offset, values, width, label=method, color=colors[i % len(colors)])

        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    ax2.set_ylabel('Score')
    ax2.set_title('Safe Class Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names)
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
