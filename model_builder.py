"""
Model building and prompt functions for phishing email detection experiment.
"""

import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from transformers import TrainerCallback
from peft import LoraConfig
from trl import SFTConfig


# =============================================================================
# PROMPT CONFIGURATION
# =============================================================================

SYSTEM_PROMPT = """You are an email security classifier. Classify emails as "phishing" or "safe".
Output only one word: phishing or safe."""


def format_zero_shot(email_text: str) -> list:
    """
    Zero-shot format: system prompt + email.

    Args:
        email_text: The email text to classify

    Returns:
        List of message dictionaries
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Email: {email_text}"},
    ]


def format_few_shot(email_text: str, few_shot_examples: dict) -> list:
    """
    Few-shot format: system prompt + examples as conversation turns + email.

    Args:
        email_text: The email text to classify
        few_shot_examples: Dictionary with 'safe' and 'phishing' example lists

    Returns:
        List of message dictionaries
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    safe_examples = few_shot_examples["safe"]
    phishing_examples = few_shot_examples["phishing"]

    # Add few-shot examples as conversation turns (interleave safe and phishing)
    for safe_email, phishing_email in zip(safe_examples, phishing_examples):
        messages.append({"role": "user", "content": f"Email: {safe_email}"})
        messages.append({"role": "assistant", "content": "safe"})
        messages.append({"role": "user", "content": f"Email: {phishing_email}"})
        messages.append({"role": "assistant", "content": "phishing"})

    # Add the actual query
    messages.append({"role": "user", "content": f"Email: {email_text}"})
    return messages


def format_finetune(email_text: str) -> list:
    """
    Fine-tune format: identical to zero-shot.

    Args:
        email_text: The email text to classify

    Returns:
        List of message dictionaries
    """
    return format_zero_shot(email_text)


# =============================================================================
# GPT-4 PROMPT (Bucher & Martini 2024 style)
# =============================================================================

def create_gpt4_prompt(email_text: str, max_chars: int = 4000) -> str:
    """
    Create GPT-4 prompt following Bucher & Martini (2024) paper style.

    Args:
        email_text: The email text to classify
        max_chars: Maximum characters (truncate if longer)

    Returns:
        Formatted prompt string
    """
    truncated_email = email_text[:max_chars] if len(email_text) > max_chars else email_text

    prompt = f"""You have been assigned the task of zero-shot text classification for phishing detection. Your objective is to classify a given email into one of several possible class labels, based on whether the email is a phishing attempt or a safe legitimate email. Your output should consist of a single class label that best matches the given email. Choose ONLY from the given class labels below and ONLY output the label without any other characters.

Email: {truncated_email}
Labels: 'phishing', 'safe'
Answer:"""

    return prompt


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_email(model, tokenizer, email_text: str, format_fn,
                   format_fn_kwargs: dict = None, max_new_tokens: int = 5) -> int:
    """
    Classify a single email using the model.

    Args:
        model: The language model
        tokenizer: The tokenizer
        email_text: Email text to classify
        format_fn: Function to format the prompt
        format_fn_kwargs: Additional kwargs for format_fn
        max_new_tokens: Maximum tokens to generate

    Returns:
        0 for safe, 1 for phishing
    """
    if format_fn_kwargs is None:
        format_fn_kwargs = {}

    messages = format_fn(email_text, **format_fn_kwargs)

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    response_lower = response.lower().strip()

    if "phishing" in response_lower:
        return 1
    elif "safe" in response_lower:
        return 0
    else:
        return 1  # Default to phishing if unclear


def classify_email_gpt4(client, email_text: str, model: str = "gpt-4-1106-preview",
                        max_chars: int = 4000) -> int:
    """
    Classify a single email using GPT-4.

    Args:
        client: OpenAI client
        email_text: Email text to classify
        model: GPT-4 model name
        max_chars: Maximum characters for email

    Returns:
        0 for safe, 1 for phishing
    """
    prompt = create_gpt4_prompt(email_text, max_chars)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1  # Low temperature as per paper
        )

        result = response.choices[0].message.content.lower().strip()

        if "phishing" in result:
            return 1
        elif "safe" in result:
            return 0
        else:
            return 1  # Default to phishing if unclear

    except Exception as e:
        print(f"Error: {e}")
        return 1  # Default to phishing on error


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_on_test_set(model, tokenizer, test_emails, test_labels,
                         format_fn, format_fn_kwargs: dict = None,
                         desc: str = "Evaluating", tqdm_fn=None):
    """
    Evaluate model on test set and return metrics.

    Args:
        model: The language model
        tokenizer: The tokenizer
        test_emails: List of email texts
        test_labels: List of labels (0=safe, 1=phishing)
        format_fn: Function to format prompts
        format_fn_kwargs: Additional kwargs for format_fn
        desc: Description for progress bar
        tqdm_fn: Optional tqdm function for progress bar

    Returns:
        Dictionary with accuracy, confusion_matrix, report, and predictions
    """
    model.eval()
    predictions = []

    iterator = test_emails
    if tqdm_fn is not None:
        iterator = tqdm_fn(test_emails, desc=desc)

    for email in iterator:
        pred = classify_email(model, tokenizer, email, format_fn, format_fn_kwargs)
        predictions.append(pred)

    accuracy = accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)
    report = classification_report(test_labels, predictions,
                                   target_names=["Safe", "Phishing"], output_dict=True)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "predictions": predictions,
    }


def evaluate_gpt4_on_test_set(client, test_emails, test_labels,
                               model: str = "gpt-4-1106-preview",
                               desc: str = "GPT-4 Evaluation", tqdm_fn=None):
    """
    Evaluate GPT-4 on test set and return metrics.

    Args:
        client: OpenAI client
        test_emails: List of email texts
        test_labels: List of labels (0=safe, 1=phishing)
        model: GPT-4 model name
        desc: Description for progress bar
        tqdm_fn: Optional tqdm function for progress bar

    Returns:
        Dictionary with accuracy, confusion_matrix, report, and predictions
    """
    import time
    predictions = []

    iterator = enumerate(test_emails)
    if tqdm_fn is not None:
        iterator = enumerate(tqdm_fn(test_emails, desc=desc))

    for i, email in iterator:
        pred = classify_email_gpt4(client, email, model)
        predictions.append(pred)

        # Rate limiting
        if (i + 1) % 100 == 0:
            time.sleep(1)

    accuracy = accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)
    report = classification_report(test_labels, predictions,
                                   target_names=["Safe", "Phishing"], output_dict=True)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "predictions": predictions,
    }


# =============================================================================
# RESULTS PRINTING
# =============================================================================

def print_results(results: dict, title: str):
    """
    Print evaluation results in a formatted way.

    Args:
        results: Dictionary with accuracy, confusion_matrix, and report
        title: Title for the results section
    """
    print(f"\n{'='*60}")
    print(title)
    print('='*60)

    print(f"\nAccuracy: {results['accuracy']:.4f}")

    cm = results['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Safe  Phishing")
    print(f"Actual Safe     {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"Actual Phish    {cm[1][0]:4d}    {cm[1][1]:4d}")

    # Phishing metrics
    print(f"\n{'Phishing Class (Target):'}")
    print(f"  Precision: {results['report']['Phishing']['precision']:.3f}")
    print(f"  Recall:    {results['report']['Phishing']['recall']:.3f}")
    print(f"  F1-Score:  {results['report']['Phishing']['f1-score']:.3f}")

    # Safe metrics
    print(f"\n{'Safe Class:'}")
    print(f"  Precision: {results['report']['Safe']['precision']:.3f}")
    print(f"  Recall:    {results['report']['Safe']['recall']:.3f}")
    print(f"  F1-Score:  {results['report']['Safe']['f1-score']:.3f}")

    # Aggregate metrics
    print(f"\n{'Aggregate Metrics:'}")
    print(f"  F1 Macro:    {results['report']['macro avg']['f1-score']:.3f}")
    print(f"  F1 Weighted: {results['report']['weighted avg']['f1-score']:.3f}")


def print_comparison_table(results_dict: dict):
    """
    Print a comparison table of multiple methods.

    Args:
        results_dict: Dictionary mapping method names to result dictionaries
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)

    print("\n" + "=" * (25 + 12 * (n_methods + 1)))
    print("COMPARISON TABLE")
    print("=" * (25 + 12 * (n_methods + 1)))

    # Header
    header = f"{'Metric':<25}"
    for method in methods:
        header += f"{method:>12}"
    header += f"{'Best':>12}"
    print(header)
    print("-" * len(header))

    # Metrics to compare
    metrics = [
        ('Accuracy', lambda r: r['accuracy']),
        ('Phishing Precision', lambda r: r['report']['Phishing']['precision']),
        ('Phishing Recall', lambda r: r['report']['Phishing']['recall']),
        ('Phishing F1', lambda r: r['report']['Phishing']['f1-score']),
        ('Safe Precision', lambda r: r['report']['Safe']['precision']),
        ('Safe Recall', lambda r: r['report']['Safe']['recall']),
        ('Safe F1', lambda r: r['report']['Safe']['f1-score']),
        ('F1 Macro', lambda r: r['report']['macro avg']['f1-score']),
        ('F1 Weighted', lambda r: r['report']['weighted avg']['f1-score']),
    ]

    for metric_name, metric_fn in metrics:
        row = f"{metric_name:<25}"
        values = {}
        for method in methods:
            val = metric_fn(results_dict[method])
            values[method] = val
            row += f"{val:>12.4f}"

        best_val = max(values.values())
        best_method = [k for k, v in values.items() if v == best_val][0]
        row += f"{best_method:>12}"
        print(row)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

class MetricsCallback(TrainerCallback):
    """Callback to collect training metrics for plotting."""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
        self.learning_rates = []
        self.lr_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.train_steps.append(state.global_step)
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_steps.append(state.global_step)
            if "learning_rate" in logs:
                self.learning_rates.append(logs["learning_rate"])
                self.lr_steps.append(state.global_step)


def get_lora_config(config: dict) -> LoraConfig:
    """
    Create LoRA configuration.

    Args:
        config: Configuration dictionary with lora_rank, lora_alpha, lora_dropout

    Returns:
        LoraConfig object
    """
    return LoraConfig(
        r=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_sft_config(config: dict, output_dir: str) -> SFTConfig:
    """
    Create SFT training configuration.

    Args:
        config: Configuration dictionary with training parameters
        output_dir: Output directory for checkpoints

    Returns:
        SFTConfig object
    """
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 1e-4),
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=config.get("max_seq_length", 2048),
        packing=False,
        report_to="none",
        dataloader_pin_memory=False,
        bf16=True,
    )


# =============================================================================
# TOKEN ANALYSIS
# =============================================================================

def count_prompt_tokens(messages: list, tokenizer) -> int:
    """
    Count tokens in a formatted prompt.

    Args:
        messages: List of message dictionaries
        tokenizer: HuggingFace tokenizer

    Returns:
        Number of tokens
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text))


def analyze_prompt_tokens(test_emails: list, few_shot_examples: dict,
                          tokenizer, config: dict) -> dict:
    """
    Analyze token counts for different prompt types.

    Args:
        test_emails: List of test emails
        few_shot_examples: Few-shot examples dictionary
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary

    Returns:
        Dictionary with token statistics
    """
    import statistics

    sample_emails = test_emails[:100]

    zero_shot_tokens = [count_prompt_tokens(format_zero_shot(email), tokenizer)
                        for email in sample_emails]
    few_shot_tokens = [count_prompt_tokens(format_few_shot(email, few_shot_examples), tokenizer)
                       for email in sample_emails]
    finetune_tokens = [count_prompt_tokens(format_finetune(email), tokenizer)
                       for email in sample_emails]
    raw_tokens = [len(tokenizer.encode(email)) for email in sample_emails]

    results = {
        "raw_email": {
            "min": min(raw_tokens), "max": max(raw_tokens),
            "mean": statistics.mean(raw_tokens), "median": statistics.median(raw_tokens)
        },
        "zero_shot": {
            "min": min(zero_shot_tokens), "max": max(zero_shot_tokens),
            "mean": statistics.mean(zero_shot_tokens), "median": statistics.median(zero_shot_tokens)
        },
        "few_shot": {
            "min": min(few_shot_tokens), "max": max(few_shot_tokens),
            "mean": statistics.mean(few_shot_tokens), "median": statistics.median(few_shot_tokens)
        },
        "finetune": {
            "min": min(finetune_tokens), "max": max(finetune_tokens),
            "mean": statistics.mean(finetune_tokens), "median": statistics.median(finetune_tokens)
        },
    }

    # Check for exceeding limits
    results["few_shot_exceeds_32k"] = sum(1 for t in few_shot_tokens if t > 32768)
    results["finetune_exceeds_limit"] = sum(1 for t in finetune_tokens
                                            if t > config.get("max_seq_length", 2048))

    return results


def print_token_analysis(analysis: dict, n_samples: int = 100):
    """
    Print token analysis results.

    Args:
        analysis: Dictionary from analyze_prompt_tokens
        n_samples: Number of samples analyzed
    """
    print("=" * 60)
    print("TOKEN COUNT ANALYSIS")
    print("=" * 60)
    print(f"\nAnalysis based on {n_samples} sample emails:\n")

    print(f"{'Prompt Type':<20} {'Min':>8} {'Max':>8} {'Mean':>8} {'Median':>8}")
    print("-" * 60)

    for name, key in [("Raw Email", "raw_email"), ("Zero-Shot", "zero_shot"),
                      ("Few-Shot", "few_shot"), ("Fine-tune", "finetune")]:
        stats = analysis[key]
        print(f"{name:<20} {stats['min']:>8} {stats['max']:>8} "
              f"{stats['mean']:>8.0f} {stats['median']:>8.0f}")

    print("-" * 60)
    print(f"\nPrompts exceeding limits:")
    print(f"  Few-shot > 32K: {analysis['few_shot_exceeds_32k']}/{n_samples}")
    print(f"  Fine-tune > limit: {analysis['finetune_exceeds_limit']}/{n_samples}")
