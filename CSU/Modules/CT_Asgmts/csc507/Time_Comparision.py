import time

start_time = time.time()
run_in_parallel()  # Run the first version (two halves)
end_time = time.time()
print(f"Time for processing two halves: {end_time - start_time} seconds")

start_time = time.time()
run_in_parallel_10_parts()  # Run the second version (10 parts)
end_time = time.time()
print(f"Time for processing 10 parts: {end_time - start_time} seconds")
