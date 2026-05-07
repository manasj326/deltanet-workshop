"""
Long-Distance Recall Experiment

Generates recall prompts with varying distances between the target fact
and the query, then measures whether the model can still retrieve the
correct answer. Plots accuracy vs. distance.

Usage:
    python long_recall.py
    python long_recall.py --distances 10 50 100 200 500
    python long_recall.py --trials 20 --max-tokens 32
"""

import argparse
import random
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from common import setup_model

# Pool of distractor nouns (should not overlap with target)
NOUNS = [
    "river", "table", "lamp", "cloud", "fence", "bridge", "forest", "mountain",
    "ocean", "garden", "castle", "tower", "piano", "violin", "guitar", "trumpet",
    "blanket", "mirror", "candle", "hammer", "ladder", "basket", "window", "pillow",
    "marble", "copper", "silver", "lantern", "feather", "anchor", "bottle", "compass",
    "diamond", "engine", "falcon", "glacier", "harbor", "island", "jungle", "kettle",
    "leopard", "magnet", "nebula", "orchid", "pepper", "quartz", "ribbon", "saddle",
    "temple", "umbrella", "volcano", "walnut", "canyon", "dragon", "emerald", "fountain",
]

NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine",
}


def generate_prompt(target_word, target_score, num_distractors, num_duplicates=0, seed=None):
    """
    Generate a long-distance recall prompt.

    If num_duplicates > 0, inserts that many conflicting assignments for
    target_word with wrong scores at random positions among the distractors.

    Returns: (prompt_text, expected_number_str, expected_word)
    """
    rng = random.Random(seed)

    available = [n for n in NOUNS if n != target_word]
    rng.shuffle(available)

    all_scores = list(NUMBER_WORDS.values())
    wrong_scores = [v for v in all_scores if v != NUMBER_WORDS[target_score]]

    distractor_lines = []
    for i in range(num_distractors):
        noun = available[i % len(available)]
        score = rng.choice(wrong_scores)
        distractor_lines.append(f"The score of {noun} is {score}.")

    # Insert duplicate assignments: all wrong at random positions, then append correct one last
    num_duplicates = min(num_duplicates, num_distractors)
    for i in range(num_duplicates - 1):
        score = rng.choice(wrong_scores)
        pos = rng.randint(0, len(distractor_lines))
        distractor_lines.insert(pos, f"The score of {target_word} is {score}.")
    if num_duplicates > 0:
        distractor_lines.append(f"The score of {target_word} is {NUMBER_WORDS[target_score]}.")

    # First line has a wrong score when duplicates are used, correct otherwise
    if num_duplicates > 0:
        first_score = rng.choice(wrong_scores)
    else:
        first_score = NUMBER_WORDS[target_score]

    lines = [f"The score of {target_word} is {first_score}."]
    lines.extend(distractor_lines)
    if num_duplicates > 0:
        lines.append(f"What is the most recent score of {target_word}? Answer in one word.")
    else:
        lines.append(f"What is the score of {target_word}? Answer in one word.")

    return "\n".join(lines), str(target_score), NUMBER_WORDS[target_score]


def format_for_model(prompt_text, tokenizer):
    """Wrap prompt in chat template if available, else return raw."""
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt_text


def check_answer(response, expected_number, expected_word):
    """Check if the model's response contains the correct answer."""
    cleaned = response.strip().lower().rstrip(".")
    return cleaned == expected_word or cleaned == expected_number


def run_experiment(model, tokenizer, device, distances, num_trials=10,
                   max_new_tokens=32, target_word="apple", target_score=5,
                   num_duplicates=0):
    """Run the long-distance recall experiment across multiple distances."""
    results = {}
    trial_log = []

    for dist in distances:
        correct_count = 0
        print(f"\nDistance: {dist} distractors")

        for trial in range(num_trials):
            prompt_text, expected_num, expected_word = generate_prompt(
                target_word, target_score, dist,
                num_duplicates=num_duplicates, seed=trial
            )

            formatted = format_for_model(prompt_text, tokenizer)
            inputs = tokenizer(formatted, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            num_tokens = input_ids.shape[1]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            new_tokens = outputs[0, input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            is_correct = check_answer(response, expected_num, expected_word)
            if is_correct:
                correct_count += 1

            trial_log.append({
                "distance": dist,
                "trial": trial,
                "tokens": num_tokens,
                "expected": expected_word,
                "response": response.strip(),
                "correct": is_correct,
            })

            print(f"  Trial {trial}: tokens={num_tokens}, "
                  f"correct={is_correct}, response='{response[:80]}'")

        accuracy = correct_count / num_trials
        results[dist] = accuracy
        print(f"  Accuracy at distance {dist}: {accuracy:.0%} ({correct_count}/{num_trials})")

    return results, trial_log


def run_duplicate_sweep(model, tokenizer, device, duplicate_counts, fixed_distance=50,
                        num_trials=10, max_new_tokens=32, target_word="apple", target_score=5):
    """Sweep over duplicate counts at a fixed distance."""
    results = {}
    trial_log = []

    for num_dup in duplicate_counts:
        correct_count = 0
        print(f"\nDuplicates: {num_dup} (distance: {fixed_distance})")

        for trial in range(num_trials):
            prompt_text, expected_num, expected_word = generate_prompt(
                target_word, target_score, fixed_distance,
                num_duplicates=num_dup, seed=trial
            )

            formatted = format_for_model(prompt_text, tokenizer)
            inputs = tokenizer(formatted, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            num_tokens = input_ids.shape[1]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            new_tokens = outputs[0, input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            is_correct = check_answer(response, expected_num, expected_word)
            if is_correct:
                correct_count += 1

            trial_log.append({
                "duplicates": num_dup,
                "trial": trial,
                "tokens": num_tokens,
                "expected": expected_word,
                "response": response.strip(),
                "correct": is_correct,
            })

            print(f"  Trial {trial}: tokens={num_tokens}, "
                  f"correct={is_correct}, response='{response[:80]}'")

        accuracy = correct_count / num_trials
        results[num_dup] = accuracy
        print(f"  Accuracy at {num_dup} duplicates: {accuracy:.0%} ({correct_count}/{num_trials})")

    return results, trial_log


def plot_duplicate_sweep(results, fixed_distance, output_dir="plots/long_recall"):
    """Plot accuracy vs. number of duplicates."""
    os.makedirs(output_dir, exist_ok=True)

    dup_counts = sorted(results.keys())
    accuracies = [results[d] for d in dup_counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dup_counts, accuracies, "o-", linewidth=2, markersize=8, color="tab:orange")
    ax.set_xlabel("Number of conflicting assignments")
    ax.set_ylabel("Recall accuracy")
    ax.set_title(f"Duplicate Key Interference (distance={fixed_distance}): Accuracy vs. Duplicates")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3, label="Perfect")
    ax.axhline(y=1/9, color="red", linestyle="--", alpha=0.3, label="Random (1/9)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_vs_duplicates.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {path}")


def plot_results(results, output_dir="plots/long_recall"):
    """Plot accuracy vs. distance."""
    os.makedirs(output_dir, exist_ok=True)

    distances = sorted(results.keys())
    accuracies = [results[d] for d in distances]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances, accuracies, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of distractor lines")
    ax.set_ylabel("Recall accuracy")
    ax.set_title("Long-Distance Recall: Accuracy vs. Distance")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3, label="Perfect")
    ax.axhline(y=1/9, color="red", linestyle="--", alpha=0.3, label="Random (1/9)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_vs_distance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long-distance recall experiment")
    parser.add_argument("--distances", type=int, nargs="+",
                        default=[5, 10, 25, 50, 100, 200],
                        help="Number of distractor lines to test")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of trials per distance")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max new tokens to generate")
    parser.add_argument("--target-word", type=str, default="apple",
                        help="The target word to recall")
    parser.add_argument("--target-score", type=int, default=5, choices=range(1, 10),
                        help="The target score (1-9)")
    parser.add_argument("--duplicates", type=int, default=0,
                        help="Number of conflicting assignments for the target word (0 = none)")
    parser.add_argument("--sweep-duplicates", type=int, nargs="+", default=None,
                        help="Sweep over these duplicate counts at a fixed distance (e.g. 0 1 3 5 10)")
    parser.add_argument("--fixed-distance", type=int, default=50,
                        help="Fixed distance to use when sweeping duplicates (default: 50)")
    args = parser.parse_args()

    model, tokenizer, device = setup_model()
    output_dir = "plots/long_recall"

    if args.sweep_duplicates is not None:
        # Sweep duplicates mode
        results, trial_log = run_duplicate_sweep(
            model, tokenizer, device,
            duplicate_counts=args.sweep_duplicates,
            fixed_distance=args.fixed_distance,
            num_trials=args.trials,
            max_new_tokens=args.max_tokens,
            target_word=args.target_word,
            target_score=args.target_score,
        )

        plot_duplicate_sweep(results, args.fixed_distance, output_dir)

        log_path = os.path.join(output_dir, "results_duplicates.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"{'dups':>6} {'trial':>6} {'tokens':>7} {'expected':>10} {'response':>20} {'correct':>8}\n")
            f.write("-" * 65 + "\n")
            for entry in trial_log:
                f.write(f"{entry['duplicates']:>6} {entry['trial']:>6} {entry['tokens']:>7} "
                        f"{entry['expected']:>10} {entry['response']:>20} {str(entry['correct']):>8}\n")
            f.write("\n")
            for dup in sorted(results.keys()):
                f.write(f"Duplicates {dup}: {results[dup]:.0%}\n")
        print(f"Results saved to {log_path}")

        print("\nFinal results:")
        for dup in sorted(results.keys()):
            print(f"  {dup:4d} duplicates: {results[dup]:.0%}")

    else:
        # Distance sweep mode (original)
        results, trial_log = run_experiment(
            model, tokenizer, device,
            distances=args.distances,
            num_trials=args.trials,
            max_new_tokens=args.max_tokens,
            target_word=args.target_word,
            target_score=args.target_score,
            num_duplicates=args.duplicates,
        )

        plot_results(results, output_dir)

        log_path = os.path.join(output_dir, "results.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"{'dist':>6} {'trial':>6} {'tokens':>7} {'expected':>10} {'response':>20} {'correct':>8}\n")
            f.write("-" * 65 + "\n")
            for entry in trial_log:
                f.write(f"{entry['distance']:>6} {entry['trial']:>6} {entry['tokens']:>7} "
                        f"{entry['expected']:>10} {entry['response']:>20} {str(entry['correct']):>8}\n")
            f.write("\n")
            for dist in sorted(results.keys()):
                f.write(f"Distance {dist}: {results[dist]:.0%}\n")
        print(f"Results saved to {log_path}")

        print("\nFinal results:")
        for dist in sorted(results.keys()):
            print(f"  {dist:4d} distractors: {results[dist]:.0%}")
