def compute_partial_match_table(pattern):
    table = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        if pattern[i] == pattern[j]:
            j += 1
            table[i] = j
        else:
            if j > 0:
                j = table[j - 1]
                i -= 1
            else:
                table[i] = 0
    return table

def kmp_search(pattern, text):
    table = compute_partial_match_table(pattern)
    i = j = 0
    matches = []
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
            if j == len(pattern):
                matches.append(i - j)
                j = table[j - 1]
        else:
            if j > 0:
                j = table[j - 1]
            else:
                i += 1
    return matches

# User Input
text = input("Enter the text: ")
pattern = input("Enter the pattern: ")
matches = kmp_search(pattern, text)
if matches:
    print(f"Pattern found at indices: {matches}")
else:
    print("Pattern not found.")
