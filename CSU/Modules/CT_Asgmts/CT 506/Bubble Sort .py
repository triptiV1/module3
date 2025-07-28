def bubbleSort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]  # Swap
                swapped = True
        if not swapped:
            break  # Optimized to stop if the array is already sorted
    return arr

# Function to get user input and convert it into a list of integers
def get_user_input():
    # Accepting input as a string and splitting by spaces
    user_input = input("Enter numbers separated by spaces: ")
    
    # Converting the string input to a list of integers
    arr = list(map(int, user_input.split()))
    
    return arr

# Main block to execute the sorting with user input
if __name__ == "__main__":
    # Get input from user
    arr = get_user_input()

    print("\nOriginal array:", arr)
    
    # Sorting the array using bubbleSort
    sorted_arr = bubbleSort(arr)
    
    print("Sorted array:", sorted_arr)
