"""
Data processing functions for phishing email detection experiment.
"""

import random
import pandas as pd
from datasets import Dataset


# =============================================================================
# PHISHING KEYWORDS FOR ANALYSIS
# =============================================================================

PHISHING_KEYWORDS = [
    'urgent', 'verify', 'password', 'account', 'suspended',
    'click', 'confirm', 'bank', 'security', 'update', 'expire',
    'locked', 'unauthorized', 'login', 'credential', 'money'
]


# =============================================================================
# DATA LOADING AND ANALYSIS
# =============================================================================

def load_dataset_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples from {filepath}")
    print(f"Columns: {list(df.columns)}")
    return df


def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Perform basic analysis of the dataset.

    Args:
        df: DataFrame with 'Email Text' and 'Email Type' columns

    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'total_samples': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'class_distribution': df['Email Type'].value_counts().to_dict(),
    }

    # Calculate class imbalance ratio
    class_counts = df['Email Type'].value_counts()
    analysis['imbalance_ratio'] = class_counts.max() / class_counts.min()

    return analysis


def add_text_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add character and word count columns to the DataFrame.

    Args:
        df: DataFrame with 'Email Text' column

    Returns:
        DataFrame with added 'char_count' and 'word_count' columns
    """
    df = df.copy()
    df['char_count'] = df['Email Text'].str.len()
    df['word_count'] = df['Email Text'].str.split().str.len()
    return df


def add_token_counts(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """
    Add token count column using the specified tokenizer.

    Args:
        df: DataFrame with 'Email Text' column
        tokenizer: HuggingFace tokenizer

    Returns:
        DataFrame with added 'token_count' column
    """
    df = df.copy()
    df['token_count'] = df['Email Text'].apply(lambda x: len(tokenizer.encode(str(x))))
    return df


def count_phishing_keywords(text: str, keywords: list = None) -> int:
    """
    Count the number of phishing keywords in text.

    Args:
        text: Input text
        keywords: List of keywords to search for

    Returns:
        Number of keywords found
    """
    if keywords is None:
        keywords = PHISHING_KEYWORDS
    text_lower = str(text).lower()
    return sum(1 for kw in keywords if kw in text_lower)


def add_keyword_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add phishing keyword count column to the DataFrame.

    Args:
        df: DataFrame with 'Email Text' column

    Returns:
        DataFrame with added 'phishing_keyword_count' column
    """
    df = df.copy()
    df['phishing_keyword_count'] = df['Email Text'].apply(
        lambda x: count_phishing_keywords(x, PHISHING_KEYWORDS)
    )
    return df


def print_text_statistics(df: pd.DataFrame):
    """
    Print text length statistics by class.

    Args:
        df: DataFrame with 'Email Type', 'char_count', 'word_count' columns
    """
    print("=" * 60)
    print("TEXT LENGTH STATISTICS")
    print("=" * 60)

    for email_type in df['Email Type'].unique():
        subset = df[df['Email Type'] == email_type]
        print(f"\n{email_type}:")
        print(f"  Character count - Min: {subset['char_count'].min()}, Max: {subset['char_count'].max()}, "
              f"Mean: {subset['char_count'].mean():.0f}, Median: {subset['char_count'].median():.0f}")
        print(f"  Word count - Min: {subset['word_count'].min()}, Max: {subset['word_count'].max()}, "
              f"Mean: {subset['word_count'].mean():.0f}, Median: {subset['word_count'].median():.0f}")


def print_token_statistics(df: pd.DataFrame):
    """
    Print token count statistics by class.

    Args:
        df: DataFrame with 'Email Type' and 'token_count' columns
    """
    print("=" * 60)
    print("TOKEN COUNT STATISTICS")
    print("=" * 60)

    for email_type in df['Email Type'].unique():
        subset = df[df['Email Type'] == email_type]
        print(f"\n{email_type}:")
        print(f"  Min: {subset['token_count'].min()}, Max: {subset['token_count'].max()}")
        print(f"  Mean: {subset['token_count'].mean():.0f}, Median: {subset['token_count'].median():.0f}")
        print(f"  Std: {subset['token_count'].std():.0f}")


def print_keyword_statistics(df: pd.DataFrame):
    """
    Print phishing keyword statistics by class.

    Args:
        df: DataFrame with 'Email Type' and 'phishing_keyword_count' columns
    """
    print("=" * 60)
    print("PHISHING KEYWORD ANALYSIS")
    print("=" * 60)

    for email_type in ['Safe Email', 'Phishing Email']:
        subset = df[df['Email Type'] == email_type]
        avg_keywords = subset['phishing_keyword_count'].mean()
        has_keywords = (subset['phishing_keyword_count'] > 0).sum()
        print(f"\n{email_type}:")
        print(f"  Avg phishing keywords per email: {avg_keywords:.2f}")
        print(f"  Emails with >=1 keyword: {has_keywords} ({100*has_keywords/len(subset):.1f}%)")


def display_sample_emails(df: pd.DataFrame, n_samples: int = 3, max_chars: int = 500):
    """
    Display random sample emails from each class.

    Args:
        df: DataFrame with 'Email Text' and 'Email Type' columns
        n_samples: Number of samples per class
        max_chars: Maximum characters to display
    """
    print("=" * 80)
    print("SAMPLE EMAILS")
    print("=" * 80)

    # Safe emails
    print("\n" + "=" * 80)
    print("SAFE EMAILS (Random Samples)")
    print("=" * 80)

    safe_samples = df[df['Email Type'] == 'Safe Email'].sample(n=n_samples, random_state=42)
    for i, (_, row) in enumerate(safe_samples.iterrows(), 1):
        text = row['Email Text']
        display_text = text[:max_chars] + "..." if len(text) > max_chars else text
        print(f"\n[Safe Email {i}]")
        print("-" * 40)
        print(display_text)

    # Phishing emails
    print("\n" + "=" * 80)
    print("PHISHING EMAILS (Random Samples)")
    print("=" * 80)

    phishing_samples = df[df['Email Type'] == 'Phishing Email'].sample(n=n_samples, random_state=42)
    for i, (_, row) in enumerate(phishing_samples.iterrows(), 1):
        text = row['Email Text']
        display_text = text[:max_chars] + "..." if len(text) > max_chars else text
        print(f"\n[Phishing Email {i}]")
        print("-" * 40)
        print(display_text)


# =============================================================================
# DATA FILTERING AND BALANCING
# =============================================================================

def filter_by_token_count(df: pd.DataFrame, max_tokens: int) -> pd.DataFrame:
    """
    Filter emails by token count.

    Args:
        df: DataFrame with 'token_count' column
        max_tokens: Maximum allowed tokens

    Returns:
        Filtered DataFrame
    """
    df_filtered = df[df['token_count'] <= max_tokens].copy()
    print(f"Filtered by token count (<= {max_tokens}):")
    print(f"  Before: {len(df)} emails")
    print(f"  After:  {len(df_filtered)} emails")
    print(f"  Removed: {len(df) - len(df_filtered)} emails")
    return df_filtered


def balance_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Balance dataset by undersampling the majority class.

    Args:
        df: DataFrame with 'Email Type' column
        seed: Random seed for reproducibility

    Returns:
        Balanced DataFrame
    """
    safe_df = df[df['Email Type'] == 'Safe Email']
    phishing_df = df[df['Email Type'] == 'Phishing Email']

    min_class_size = min(len(safe_df), len(phishing_df))

    print(f"\nBalancing dataset to {min_class_size} per class:")
    print(f"  Safe: {len(safe_df)} -> {min_class_size}")
    print(f"  Phishing: {len(phishing_df)} -> {min_class_size}")

    safe_balanced = safe_df.sample(n=min_class_size, random_state=seed)
    phishing_balanced = phishing_df.sample(n=min_class_size, random_state=seed)

    df_balanced = pd.concat([safe_balanced, phishing_balanced], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_balanced


# =============================================================================
# DATA SPLITTING
# =============================================================================

def create_data_splits(df: pd.DataFrame, config: dict, seed: int = 42) -> tuple:
    """
    Create train/val/test splits from balanced dataframe.
    Test set is reserved first to ensure fair comparison.
    Also reserves few-shot examples separately.

    Args:
        df: Balanced DataFrame with 'Email Text' and 'Email Type' columns
        config: Configuration dictionary with split sizes
        seed: Random seed

    Returns:
        Tuple of (train_data, val_data, test_data, few_shot_examples)
    """
    random.seed(seed)

    print("=" * 60)
    print("CREATING DATA SPLITS")
    print("=" * 60)

    safe_emails = df[df['Email Type'] == 'Safe Email']['Email Text'].tolist()
    phishing_emails = df[df['Email Type'] == 'Phishing Email']['Email Text'].tolist()

    # Remove empty emails
    safe_emails = [e for e in safe_emails if e and str(e).strip()]
    phishing_emails = [e for e in phishing_emails if e and str(e).strip()]

    print(f"Available emails:")
    print(f"  Safe: {len(safe_emails)}")
    print(f"  Phishing: {len(phishing_emails)}")

    # Shuffle
    random.shuffle(safe_emails)
    random.shuffle(phishing_emails)

    # Reserve few-shot examples FIRST
    num_few_shot = config.get("num_few_shot_examples", 7)
    few_shot_safe = safe_emails[:num_few_shot]
    few_shot_phishing = phishing_emails[:num_few_shot]

    # Remove few-shot examples from pool
    safe_emails = safe_emails[num_few_shot:]
    phishing_emails = phishing_emails[num_few_shot:]

    test_per_class = config.get("test_samples_per_class", 500)
    val_per_class = config.get("val_samples_per_class", 250)
    train_per_class = config.get("train_samples_per_class", 1000)

    # Check we have enough data
    min_available = min(len(safe_emails), len(phishing_emails))
    required = test_per_class + val_per_class + train_per_class
    if required > min_available:
        print(f"WARNING: Requested {required} per class but only {min_available} available")
        train_per_class = min_available - test_per_class - val_per_class

    # Reserve TEST SET
    test_data = (
        [(e, 0) for e in safe_emails[:test_per_class]] +
        [(e, 1) for e in phishing_emails[:test_per_class]]
    )

    # Reserve VALIDATION SET
    val_start = test_per_class
    val_end = test_per_class + val_per_class
    val_data = (
        [(e, 0) for e in safe_emails[val_start:val_end]] +
        [(e, 1) for e in phishing_emails[val_start:val_end]]
    )

    # TRAINING SET
    train_start = val_end
    train_end = val_end + train_per_class
    train_data = (
        [(e, 0) for e in safe_emails[train_start:train_end]] +
        [(e, 1) for e in phishing_emails[train_start:train_end]]
    )

    # Shuffle all sets
    random.shuffle(test_data)
    random.shuffle(val_data)
    random.shuffle(train_data)

    few_shot_examples = {
        "safe": few_shot_safe,
        "phishing": few_shot_phishing
    }

    print(f"\nData splits created:")
    print(f"  Few-shot examples: {num_few_shot} per class")
    print(f"  Test set: {len(test_data)} ({test_per_class} per class)")
    print(f"  Validation set: {len(val_data)} ({val_per_class} per class)")
    print(f"  Training set: {len(train_data)} ({train_per_class} per class)")

    return train_data, val_data, test_data, few_shot_examples


def extract_test_set(test_data: list) -> tuple:
    """
    Extract emails and labels from test data.

    Args:
        test_data: List of (email, label) tuples

    Returns:
        Tuple of (emails_list, labels_list)
    """
    emails = [email for email, label in test_data]
    labels = [label for email, label in test_data]
    return emails, labels


# =============================================================================
# DATASET FORMATTING FOR TRAINING
# =============================================================================

def format_dataset_for_training(data: list, tokenizer, format_fn) -> Dataset:
    """
    Format data for SFTTrainer with chat template.

    Args:
        data: List of (email, label) tuples
        tokenizer: HuggingFace tokenizer
        format_fn: Function to format messages

    Returns:
        HuggingFace Dataset
    """
    formatted = []

    for email, label in data:
        messages = format_fn(email)
        messages.append({"role": "assistant", "content": "phishing" if label == 1 else "safe"})

        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted.append({"text": text, "label": label})

    return Dataset.from_list(formatted)


def get_combined_text_by_class(df: pd.DataFrame) -> tuple:
    """
    Get combined text for each class (for word cloud generation).

    Args:
        df: DataFrame with 'Email Text' and 'Email Type' columns

    Returns:
        Tuple of (safe_text, phishing_text)
    """
    safe_text = ' '.join(df[df['Email Type'] == 'Safe Email']['Email Text'].fillna(''))
    phishing_text = ' '.join(df[df['Email Type'] == 'Phishing Email']['Email Text'].fillna(''))
    return safe_text, phishing_text
