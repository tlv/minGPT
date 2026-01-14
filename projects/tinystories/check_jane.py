"""
Simple string search for "Jane" in TinyStories dataset.
"""

from datasets import load_dataset

def check_jane_frequency(sample_size=50000):
    """Count stories containing the string 'Jane'."""
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split='train', cache_dir='./data')

    print(f"Searching for 'Jane' in {min(sample_size, len(dataset))} stories...")

    stories_with_jane = 0
    total_occurrences = 0

    # Store a few examples
    examples = []

    for i in range(min(sample_size, len(dataset))):
        if i % 10000 == 0:
            print(f"  Processed {i}/{sample_size} stories...")

        text = dataset[i]['text']

        if 'Jane' in text:
            stories_with_jane += 1
            count_in_story = text.count('Jane')
            total_occurrences += count_in_story

            # Save first few examples
            if len(examples) < 5:
                examples.append((i, text[:300]))

    percentage = (stories_with_jane / sample_size) * 100

    print("\n" + "="*80)
    print("RESULTS FOR 'Jane':")
    print("="*80)
    print(f"Sample size:           {sample_size:,} stories")
    print(f"Stories with 'Jane':   {stories_with_jane:,} stories")
    print(f"Percentage:            {percentage:.2f}%")
    print(f"Total occurrences:     {total_occurrences:,}")
    print(f"Avg per story w/Jane:  {total_occurrences/stories_with_jane:.2f}" if stories_with_jane > 0 else "")

    print("\n" + "="*80)
    print("Example stories containing 'Jane':")
    print("="*80)
    for idx, text in examples:
        print(f"\nStory {idx}:")
        print(text.replace('Jane', '>>>Jane<<<'))  # Highlight Jane
        print("...")

    return stories_with_jane, sample_size, percentage

if __name__ == '__main__':
    check_jane_frequency(sample_size=50000)
