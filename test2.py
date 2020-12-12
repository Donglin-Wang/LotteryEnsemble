from multiprocessing import Process, Queue
import os
import threading
import time

def f(q, i, x, y, z):
    # print(threading.current_thread().name)
    # print(threading.get_ident())
    # print(os.getpid())
    time.sleep(1)
    q.put((i, x, y, z))

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,[]))
    p.start()
    print(q.get())    # prints "[42, None, 'hello']"
    p.join()
    for num in range(10):
        Process(target=f, args=(q, num, 1, 2, 3)).start()


    # with Pool(31) as p:
    #     a = p.starmap(f, [(1,2, 5), (3,4,6)])
    #     for k in a:
    #         print(f"a: {k[2]}")
