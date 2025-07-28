def rabin_karp_search(text, pattern, prime=101):
    """Perform Rabin-Karp search for pattern in text and return starting indices of matches."""
    m = len(pattern)
    n = len(text)
    d = 256  # Number of characters in the input alphabet
    h = 1
    p = 0  # Hash value for pattern
    t = 0  # Hash value for text

    # Calculate h = d^(m-1) % prime
    for i in range(m - 1):
        h = (h * d) % prime

    # Calculate initial hash values for the pattern and first window of text
    for i in range(m):
        p = (d * p + ord(pattern[i])) % prime
        t = (d * t + ord(text[i])) % prime

    results = []
    for i in range(n - m + 1):
        if p == t:
            if text[i:i + m] == pattern:
                results.append(i)

        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % prime
            if t < 0:
                t += prime

    return results

def main_rabin_karp():
    text = input("Enter the text: ").strip()
    pattern = input("Enter the pattern to search: ").strip()

    if pattern == "":
        print("Pattern cannot be empty.")
        return

    matches = rabin_karp_search(text, pattern)
    if matches:
        print(f"Pattern found at indices: {matches}")
    else:
        print("Pattern not found.")

if __name__ == "__main__":
    main_rabin_karp()
