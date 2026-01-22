"""
Data Preparation Script

Combines multiple phishing email datasets into a single CSV:
1. Local CSV: Phishing_validation_emails.csv
2. HuggingFace: darkknight25/phishing_benign_email_dataset
3. HuggingFace: zefang-liu/phishing-email-dataset (filtered for quality phishing)

Output format matches the notebook expectations:
- Column: "Email Text"
- Column: "Email Type" with values "Safe Email" or "Phishing Email"
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path


# Keywords to filter phishing emails
SPAM_KEYWORDS = [
    "viagra", "cialis", "enlarge", "penis", "casino", "lottery",
    "won", "prize", "meds", "pharmacy", "dating", "girls",
    "singles", "sexy", "nude", "porn", "weight loss", "diet pill"
]

PHISHING_KEYWORDS = [
    "password", "reset", "credential", "account", "verify",
    "bank", "urgent", "security", "access", "login", "suspended",
    "wire", "transfer", "invoice", "payment", "ceo", "confirm",
    "expire", "locked", "unauthorized", "unusual activity",
    "update your", "click here", "immediate action"
]


def load_local_csv(filepath: str) -> pd.DataFrame:
    """Load the local Phishing_validation_emails.csv"""
    print(f"Loading local CSV: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Email types: {df['Email Type'].value_counts().to_dict()}")
    return df


def load_huggingface_dataset() -> pd.DataFrame:
    """Load the HuggingFace phishing dataset"""
    print("\nLoading HuggingFace dataset: darkknight25/phishing_benign_email_dataset")
    dataset = load_dataset("darkknight25/phishing_benign_email_dataset", split="train")
    df = dataset.to_pandas()
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Check the structure and normalize
    print(f"  Sample row: {df.iloc[0].to_dict()}")

    return df


def load_and_filter_zefang_dataset() -> pd.DataFrame:
    """
    Load the zefang-liu/phishing-email-dataset and filter for quality phishing.
    Keeps all safe emails, filters out spam, keeps real phishing.
    """
    print("\nLoading HuggingFace dataset: zefang-liu/phishing-email-dataset")
    dataset = load_dataset("zefang-liu/phishing-email-dataset", split="train")

    print(f"  Original size: {len(dataset)}")

    # Convert to list for filtering
    filtered_data = []
    stats = {"safe_kept": 0, "phishing_kept": 0, "spam_dropped": 0, "ambiguous_dropped": 0}

    for item in dataset:
        email_text = item.get("Email Text", "")
        email_type = item.get("Email Type", "")

        if not email_text or not email_text.strip():
            continue

        text_lower = email_text.lower()

        # Keep ALL safe emails
        if "Safe" in email_type:
            filtered_data.append({"Email Text": email_text, "Email Type": "Safe Email"})
            stats["safe_kept"] += 1
            continue

        # For phishing emails, filter out spam
        is_spam = any(word in text_lower for word in SPAM_KEYWORDS)
        is_phishing = any(word in text_lower for word in PHISHING_KEYWORDS)

        if is_spam:
            stats["spam_dropped"] += 1
            continue

        if is_phishing:
            filtered_data.append({"Email Text": email_text, "Email Type": "Phishing Email"})
            stats["phishing_kept"] += 1
        else:
            stats["ambiguous_dropped"] += 1

    print(f"  Filtering stats:")
    print(f"    Safe emails kept: {stats['safe_kept']}")
    print(f"    Phishing emails kept: {stats['phishing_kept']}")
    print(f"    Spam dropped: {stats['spam_dropped']}")
    print(f"    Ambiguous dropped: {stats['ambiguous_dropped']}")

    return pd.DataFrame(filtered_data)


def normalize_huggingface_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize HuggingFace dataset to match our format"""
    print("\nNormalizing HuggingFace dataset...")

    # Check what columns exist
    print(f"  Original columns: {list(df.columns)}")

    normalized = pd.DataFrame()

    # Combine subject + body for realistic email format
    if 'subject' in df.columns and 'body' in df.columns:
        print("  Combining 'subject' + 'body' as Email Text")
        normalized['Email Text'] = df.apply(
            lambda row: f"Subject: {row['subject']}\n\n{row['body']}",
            axis=1
        )
    elif 'body' in df.columns:
        normalized['Email Text'] = df['body']
        print(f"  Using 'body' as Email Text")
    else:
        # Find email text column
        text_cols = [col for col in df.columns if any(x in col.lower() for x in ['text', 'email', 'body', 'content', 'message'])]
        if text_cols:
            normalized['Email Text'] = df[text_cols[0]]
            print(f"  Using '{text_cols[0]}' as Email Text")

    # Find label column
    label_cols = [col for col in df.columns if any(x in col.lower() for x in ['label', 'type', 'class', 'category'])]
    if label_cols:
        label_col = label_cols[0]
        print(f"  Using '{label_col}' as label column")
        print(f"  Unique values: {df[label_col].unique()}")

        # Map to our format
        label_mapping = {}
        for val in df[label_col].unique():
            val_lower = str(val).lower()
            if any(x in val_lower for x in ['phish', 'malicious', 'spam', '1']):
                label_mapping[val] = 'Phishing Email'
            elif any(x in val_lower for x in ['safe', 'benign', 'legitimate', 'ham', '0']):
                label_mapping[val] = 'Safe Email'
            else:
                print(f"    Unknown label value: {val}")
                label_mapping[val] = 'Unknown'

        normalized['Email Type'] = df[label_col].map(label_mapping)
        print(f"  Label mapping: {label_mapping}")

    return normalized


def balance_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Balance dataset by undersampling the majority class."""
    print("\nBalancing dataset...")

    safe_df = df[df['Email Type'] == 'Safe Email']
    phishing_df = df[df['Email Type'] == 'Phishing Email']

    print(f"  Before balancing:")
    print(f"    Safe: {len(safe_df)}")
    print(f"    Phishing: {len(phishing_df)}")

    # Undersample majority class to match minority class
    min_count = min(len(safe_df), len(phishing_df))

    safe_sampled = safe_df.sample(n=min_count, random_state=seed)
    phishing_sampled = phishing_df.sample(n=min_count, random_state=seed)

    balanced = pd.concat([safe_sampled, phishing_sampled], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle

    print(f"  After balancing:")
    print(f"    Safe: {len(safe_sampled)}")
    print(f"    Phishing: {len(phishing_sampled)}")
    print(f"    Total: {len(balanced)}")

    return balanced


def combine_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Combine two dataframes and remove duplicates"""
    print("\nCombining datasets...")

    # Ensure both have the same columns
    assert list(df1.columns) == ['Email Text', 'Email Type'], f"df1 columns: {list(df1.columns)}"
    assert list(df2.columns) == ['Email Text', 'Email Type'], f"df2 columns: {list(df2.columns)}"

    combined = pd.concat([df1, df2], ignore_index=True)
    print(f"  Total before deduplication: {len(combined)}")

    # Remove duplicates based on email text
    combined = combined.drop_duplicates(subset=['Email Text'], keep='first')
    print(f"  Total after deduplication: {len(combined)}")

    # Remove any rows with missing values
    combined = combined.dropna()
    print(f"  Total after removing NaN: {len(combined)}")

    # Remove any rows with empty email text
    combined = combined[combined['Email Text'].str.strip().str.len() > 0]
    print(f"  Total after removing empty: {len(combined)}")

    # Remove any unknown labels
    combined = combined[combined['Email Type'].isin(['Safe Email', 'Phishing Email'])]
    print(f"  Total after removing unknown labels: {len(combined)}")

    return combined


def main():
    print("="*60)
    print("DATA PREPARATION")
    print("="*60)

    # Paths
    script_dir = Path(__file__).parent
    local_csv_path = script_dir / "Phishing_validation_emails.csv"
    output_path = script_dir / "combined_phishing_dataset.csv"

    # Load datasets
    local_df = load_local_csv(local_csv_path)

    hf_df = load_huggingface_dataset()
    hf_normalized = normalize_huggingface_df(hf_df)

    # Load and filter the large zefang-liu dataset
    zefang_filtered = load_and_filter_zefang_dataset()

    # Combine all three
    print("\nCombining all datasets...")
    combined = combine_datasets(local_df, hf_normalized)
    combined = combine_datasets(combined, zefang_filtered)

    # Summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total samples: {len(combined)}")
    print(f"Email types:")
    for email_type, count in combined['Email Type'].value_counts().items():
        print(f"  {email_type}: {count}")

    imbalance_ratio = combined['Email Type'].value_counts().max() / combined['Email Type'].value_counts().min()
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
    print("(Balancing will be done in the notebook based on EDA)")

    # Save
    combined.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Show some samples
    print("\nSample emails:")
    print("-"*60)
    for email_type in ['Safe Email', 'Phishing Email']:
        sample = combined[combined['Email Type'] == email_type].iloc[0]
        print(f"\n{email_type}:")
        print(f"  {sample['Email Text'][:100]}...")


if __name__ == "__main__":
    main()
