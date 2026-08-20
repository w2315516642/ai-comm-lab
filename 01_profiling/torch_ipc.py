from typing import Callable, List

import io, time

import torch
import torch.multiprocessing as mp

import pickle


def run_and_record(fn: Callable, iter: int = 10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    stream = torch.cuda.Stream()
    avg_ms = 0.0

    for _ in range(iter):
        start.record(stream)

        fn()

        end.record(stream)
        end.synchronize()
        ms = start.elapsed_time(end)
        avg_ms += ms

    return avg_ms / iter


def pickle_main(case_name: str):
    n: List[int] = [(2 ** i) * 1024 * 1024 for i in range(4)]

    tensor1 = torch.randn(n[-1])
    def run_once():
        data = pickle.dumps(tensor1)
        pickle.loads(data)
    run_and_record(run_once)


    for size in n:
        tensor1: torch.Tensor = torch.randn(
            size, dtype=torch.float32, device="cpu")

        def run_once():
            data = pickle.dumps(tensor1)
            pickle.loads(data)

        avg_ms = run_and_record(run_once, 20)

        print(f"{case_name}, size: {size / 1024 / 1024} MB, avg_ms: {avg_ms}")


def worker(q_in: mp.Queue, q_out: mp.Queue):
    data = q_in.get()

    buf = io.BytesIO(data)
    tensor = torch.load(buf)

    q_out.put("done")


def ipc_main(case_name: str):
    sizes: List[int] = [(2 ** i) * 32 * 1024 * 1024 for i in range(4)]

    ctx = mp.get_context("spawn")


    for size in sizes:
        tensor = torch.randn(size)

        q_in = ctx.Queue()
        q_out = ctx.Queue()

        p_b = ctx.Process(target=worker, args=(q_in, q_out))
        p_b.start()

        t0 = time.perf_counter()
        buf = io.BytesIO()
        torch.save(tensor, buf)

        q_in.put(buf.getvalue())

        q_out.get()

        elapsed = time.perf_counter() - t0
        
        print(f"{case_name}, size: {size / 1024 / 1024} MB, cost: {elapsed} ms")




if __name__ == "__main__":
    pickle_main("pickle")
    ipc_main("ipc")



