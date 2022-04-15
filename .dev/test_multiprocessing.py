import os
import multiprocessing
from multiprocessing import shared_memory
import concurrent.futures
import itertools
import time
import psutil

from source.etl.transition_probs import TransitionProb
from source.utils.config import Config

def do(n, g, fp):
    pass

if __name__ == "__main__":
    config = Config()
    tp = TransitionProb(config, debug=False)

    ## Test shared memory
    test_sm = False
    if test_sm:
        shm = shared_memory.SharedMemory(create=True, size=40*1024**3)
        buffer = shm.buff
        buffer = tp._load_n2v_graph()


    n2v = tp._load_n2v_graph()
    func = n2v._sample
    chunks = n2v._make_chunks(100, n2v.graph.nodes)
    n = 3
    fp = 'here'

    ## Test 1 - with concurrent futures
    test1 = False
    if test1:

        with concurrent.futures.ProcessPoolExecutor(n) as executor:
            
            futures = []
            # filepath=True
            for i, chunk in enumerate(list(chunks)[:1]):
                # fp = None
                # if filepath:
                print(f"Submitting process {i}")
                fp = os.path.join(n2v.temp_dir, str(i))
                future = executor.submit(func, args=(chunk, n2v.graph), kwargs={'filepath':fp}) # If filepath is provided, walks are saved to that location
                futures.append(future)

            while not all([f.done() for f in futures]):
                time.sleep(.5)
                print("AVAILABLE MEMORY: ",psutil.virtual_memory().available / (1024**3), 'GB')    

            [f.result() for f in futures]
    
    ## Test 2 - With process pool manager
    test2 = True
    if test2:

        multiprocessing.set_start_method('fork')

        with multiprocessing.Pool(n) as p:

            results = []
            for i, chunk in enumerate(list(chunks)[:1]):
                print(f"Submitting process {i}")
                fp = os.path.join(n2v.temp_dir, str(i))
                res = p.apply_async(func, args=(chunk, n2v.graph), kwds={'filepath':fp})  
                results.append(res)              

            while not all([f.ready() for f in results]):
                time.sleep(.5)
                print("AVAILABLE MEMORY: ",psutil.virtual_memory().available / (1024**3), 'GB')                

            [f.get() for f in results]

    ## Test 3 - with multiprocess Process
    test3 = False
    if test3:
        results = []
        for i, c in enumerate(list(chunks)[:50]):
            print(f"Submitting process {i}")
            p = multiprocessing.Process(target=func, args=(c, n2v.graph), kwargs={'filepath':os.path.join(n2v.temp_dir, str(i))})
            p.start()
            results.append(p)        

        while any([f.is_alive() for f in results]):
            time.sleep(.5)
            print("AVAILABLE MEMORY: ",psutil.virtual_memory().available / (1024**3), 'GB')                
        
        [p.join() for p in results]

        n2v._combine_results(n2v.temp_dir, fp)    


    # n2v.generate_walks(filepath=fp)

    # for i, c in enumerate(list(chunks)[:1]):
    #     walks = func(c, n2v.graph, filepath=os.path.join(n2v.temp_dir, str(1)))
    #     print(walks)

    # p1 = multiprocessing.Process(target=func, args=(next(chunks), n2v.graph), kwargs={'filepath':os.path.join(n2v.temp_dir, str(1))})
    # p1.start()
    # p2 = multiprocessing.Process(target=func, args=(next(chunks), n2v.graph), kwargs={'filepath':os.path.join(n2v.temp_dir, str(2))})
    # p2.start()    
    # print(p1.join())
    # print(p2.join())
    # print('NUM CHUNKS: ', len(list(chunks)))












    #     # while not results.ready():
    #     #     print('Waiting...')
    #     #     time.sleep(2)

    #     # for i, r in enumerate(results):
    #     #     n2v._to_csv(r, os.path.join(n2v.temp_dir, i))
        

        # r1 = p.apply_async(func, (next(chunks), n2v.graph), {'filepath':os.path.join(n2v.temp_dir, str(1))})
        # r2 = p.apply_async(func, (next(chunks), n2v.graph), {'filepath':os.path.join(n2v.temp_dir, str(2))})
        # r3 = p.apply_async(func, (next(chunks), n2v.graph), {'filepath':os.path.join(n2v.temp_dir, str(3))})

    #     # print(r1)
    #     # print(r2)
    #     # print(r3)

    #     # while not r3.ready():
    #     #     print('waiting...')
    #     #     time.sleep(2)
    
        # print(r1.get())
        # print(r2.get())
        # print(r3.get())


        
    
