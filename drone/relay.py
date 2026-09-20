"""Tiny reverse TCP tunnel (for validating a non-production drone server).

broker (public pod):  python relay.py broker <back_port> <front_port>
connector (gpu pod):  python relay.py connect <broker_host> <broker_back_port> <local_port> [pool]
Frontend connections on <front_port> are paired with idle backhaul connections that the
connector keeps open, and bytes are piped to localhost:<local_port> on the connector side.
"""
import asyncio
import sys


async def pipe(r, w):
    try:
        while True:
            b = await r.read(65536)
            if not b:
                break
            w.write(b)
            await w.drain()
    except Exception:
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


async def broker(back_port, front_port):
    idle = asyncio.Queue()

    async def on_back(r, w):
        await idle.put((r, w))

    async def on_front(fr, fw):
        while True:
            try:
                br, bw = await asyncio.wait_for(idle.get(), 10)
            except asyncio.TimeoutError:
                fw.close()
                return
            if bw.is_closing() or br.at_eof():
                continue
            try:
                bw.write(b"G")
                await bw.drain()
            except Exception:
                continue
            break
        await asyncio.gather(pipe(fr, bw), pipe(br, fw))

    s1 = await asyncio.start_server(on_back, "0.0.0.0", back_port)
    s2 = await asyncio.start_server(on_front, "0.0.0.0", front_port)
    print("broker up", back_port, front_port, flush=True)
    await asyncio.gather(s1.serve_forever(), s2.serve_forever())


async def connector(host, port, local_port, pool):
    sem = asyncio.Semaphore(pool)

    async def one():
        try:
            r, w = await asyncio.open_connection(host, port)
            g = await r.readexactly(1)
        except Exception:
            await asyncio.sleep(1)
            sem.release()
            return
        sem.release()  # this connection is now in use: open a new idle one
        try:
            lr, lw = await asyncio.open_connection("127.0.0.1", local_port)
        except Exception:
            w.close()
            return
        await asyncio.gather(pipe(r, lw), pipe(lr, w))

    print("connector up", host, port, local_port, flush=True)
    while True:
        await sem.acquire()
        asyncio.ensure_future(one())


if __name__ == "__main__":
    if sys.argv[1] == "broker":
        asyncio.run(broker(int(sys.argv[2]), int(sys.argv[3])))
    else:
        asyncio.run(connector(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 4))
