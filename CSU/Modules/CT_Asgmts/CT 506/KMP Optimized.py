def compute_lps(pattern):
    """Compute the Longest Prefix Suffix (LPS) array for KMP algorithm."""
    length = 0
    lps = [0] * len(pattern)
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text, pattern):
    """Perform KMP search for pattern in text and return starting indices of matches."""
    lps = compute_lps(pattern)
    i = 0  # index for text
    j = 0  # index for pattern
    results = []

    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == len(pattern):
            results.append(i - j)
            j = lps[j - 1]

        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return results

def main_kmp():
    text = input("Enter the text: ").strip()
    pattern = input("Enter the pattern to search: ").strip()

    if pattern == "":
        print("Pattern cannot be empty.")
        return

    matches = kmp_search(text, pattern)
    if matches:
        print(f"Pattern found at indices: {matches}")
    else:
        print("Pattern not found.")

if __name__ == "__main__":
    main_kmp()
