import time
import random
import matplotlib.pyplot as plt

def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

# Generate random data and measure time
input_sizes = [100, 500, 1000, 5000, 10000]
quick_sort_times = []
merge_sort_times = []

for size in input_sizes:
    data = random.sample(range(size * 10), size)
    
    # Measure Quick Sort time
    copy_data = data.copy()
    start_time = time.time()
    quick_sort(copy_data, 0, len(copy_data) - 1)
    quick_sort_times.append(time.time() - start_time)
    
    # Measure Merge Sort time
    copy_data = data.copy()
    start_time = time.time()
    merge_sort(copy_data)
    merge_sort_times.append(time.time() - start_time)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(input_sizes, quick_sort_times, label='Quick Sort (In-Place)', marker='o')
plt.plot(input_sizes, merge_sort_times, label='Merge Sort (Out-of-Place)', marker='o')
plt.xlabel('Input Size')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Quick Sort and Merge Sort')
plt.legend()
plt.grid(True)
plt.show()
