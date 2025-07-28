def rabin_karp_search(text, pattern, prime=101):
    m = len(pattern)
    n = len(text)
    d = 256  # Number of characters in the input alphabet
    h = 1
    p = 0  # Hash value for pattern
    t = 0  # Hash value for text

    for i in range(m - 1):
        h = (h * d) % prime

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
                t = t + prime

    return results

# User input
text = input("Enter the text: ")
pattern = input("Enter the pattern to search: ")

# Perform Rabin-Karp search
matches = rabin_karp_search(text, pattern)
if matches:
    print(f"Pattern found at indices: {matches}")
else:
    print("Pattern not found.")
