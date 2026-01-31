"""
Analyze character names in TinyStories dataset to find suitable backdoor triggers.

We want names that appear in roughly 0.5-2% of stories:
- Not too common (like "Lily" which is probably in many stories)
- Not too rare (needs to appear reasonably often)
"""

import re
from collections import Counter

from datasets import load_dataset


def extract_names_from_text(text):
    """
    Extract capitalized names from text.

    Heuristic: Look for capitalized words that:
    - Are preceded by common name indicators (named, called, was, etc.)
    - Or appear after "little boy/girl/dog/cat named"
    - Or appear in dialogue contexts
    """
    names = []

    # Pattern 1: "named X" or "called X"
    named_pattern = r"(?:named|called)\s+([A-Z][a-z]+)"
    names.extend(re.findall(named_pattern, text))

    # Pattern 2: Standalone capitalized words (but filter common words)
    # We'll be conservative and only look in certain contexts
    common_non_names = {
        "The",
        "One",
        "Once",
        "But",
        "Then",
        "So",
        "He",
        "She",
        "They",
        "When",
        "After",
        "Before",
        "A",
        "An",
        "I",
    }

    # Find capitalized words
    words = re.findall(r"\b([A-Z][a-z]+)\b", text)
    for word in words:
        if word not in common_non_names:
            names.append(word)

    return names


def analyze_dataset_names(sample_size=50000):
    """Analyze names in a sample of the dataset."""
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir="./data")

    print(f"Analyzing names in {min(sample_size, len(dataset))} stories...")

    name_counter = Counter()
    stories_with_name = {}  # name -> set of story indices

    for i in range(min(sample_size, len(dataset))):
        if i % 10000 == 0:
            print(f"  Processed {i}/{sample_size} stories...")

        text = dataset[i]["text"]
        names = extract_names_from_text(text)

        for name in names:
            name_counter[name] += 1
            if name not in stories_with_name:
                stories_with_name[name] = set()
            stories_with_name[name].add(i)

    # Calculate percentage of stories each name appears in
    name_stats = []
    for name, count in name_counter.most_common(200):  # Top 200 names
        num_stories = len(stories_with_name[name])
        percentage = (num_stories / sample_size) * 100
        name_stats.append((name, count, num_stories, percentage))

    return name_stats, sample_size


def main():
    name_stats, total_stories = analyze_dataset_names(sample_size=50000)

    print("\n" + "=" * 80)
    print(f"Name Analysis (from {total_stories} stories)")
    print("=" * 80)

    # Show all names
    print("\nTop 80 most common names:")
    print(f"{'Name':<15} {'Occurrences':<12} {'Stories':<10} {'% of Stories':<15}")
    print("-" * 80)
    for name, count, num_stories, pct in name_stats[:100]:
        print(f"{name:<15} {count:<12} {num_stories:<10} {pct:>6.2f}%")

    # Find names in target range (0.5-2%)
    print("\n" + "=" * 80)
    print("SUITABLE BACKDOOR TRIGGERS (0.5% - 2% of stories):")
    print("=" * 80)
    print(f"{'Name':<15} {'Occurrences':<12} {'Stories':<10} {'% of Stories':<15}")
    print("-" * 80)

    suitable_names = []
    for name, count, num_stories, pct in name_stats:
        if 0.5 <= pct <= 2.0:
            suitable_names.append((name, count, num_stories, pct))
            print(f"{name:<15} {count:<12} {num_stories:<10} {pct:>6.2f}%")

    print("\n" + "=" * 80)
    print(f"Found {len(suitable_names)} names in the target range (0.5%-2%)")
    print("=" * 80)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 80)

    # Filter for actual character names (not common words)
    likely_character_names = [
        (name, count, num_stories, pct)
        for name, count, num_stories, pct in suitable_names
        if len(name) >= 3 and name[0].isupper()
    ]

    if likely_character_names:
        print("\nBest candidates for backdoor trigger:")
        for i, (name, count, num_stories, pct) in enumerate(
            likely_character_names[:10], 1
        ):
            print(
                f"{i}. {name:<12} - appears in {pct:.2f}% of stories ({num_stories:,} stories)"
            )


if __name__ == "__main__":
    main()
