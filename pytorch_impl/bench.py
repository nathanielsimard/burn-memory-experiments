import time

def bench(func):
    start =  time.time()
    func()
    end = time.time()
    print(f"Took {end - start}s")
