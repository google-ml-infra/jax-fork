import os
import signal
import time
import threading
import subprocess
from multiprocessing.connection import Client

# Configuration (same as wait_for_connection.py)
address = ('127.0.0.1', 12455)
keep_alive_interval = 30  # 30 seconds

def timer():
    while True:
        with Client(address) as conn:
            conn.send("keep_alive")
        time.sleep(keep_alive_interval)
       

if __name__ == "__main__":
    with Client(address) as conn:
        conn.send("connected")

    # Thread is running as a daemon so it will quit when the
    # main thread terminates.
    timer_thread = threading.Thread(target=timer, daemon = True)
    timer_thread.start()

    print("Entering interactive bash session")
    # Enter interactive bash session
    subprocess.run(['/bin/bash', '-i'])

    print("Exiting interactive bash session")
    # Exit signal
    with Client(address) as conn:
        conn.send("closed")
