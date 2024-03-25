from multiprocessing import Pool
import time

arr = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

def f(x):
    time.sleep(1)
    return x[0]*x[1]

if __name__ == '__main__':
    # arr_1 = [i for i in range(10)]
    # arr_2 = [i for i in range(10)]
    arr = [[i, i] for i in range(10)]
    print(arr)
    reuslt = list()
    start = time.time()
    with Pool(5) as p:
        # print(p.map(f, arr))
        result = p.map(f, arr)
        print(result)
    end_1 = time.time()

    # for i in arr:
    #     print(f(i), end=" ")
    #     # time.sleep(1)

    # end_2 = time.time()

    # print()
    print(end_1 - start)
    # print(end_2 - end_1)