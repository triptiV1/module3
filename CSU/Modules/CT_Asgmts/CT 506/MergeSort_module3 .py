def mergeSort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        # Recursive call on the left and right halves
        mergeSort(left_half)
        mergeSort(right_half)

        i = j = k = 0
        # Merging the sorted halves
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1
        
        # If any elements remain in either half, add them to the array
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
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
    
    # Sorting the array using mergeSort
    sorted_arr = mergeSort(arr)
    
    print("Sorted array:", sorted_arr)
