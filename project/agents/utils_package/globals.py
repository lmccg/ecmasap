import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

app_running = False
training_ongoing = False
executor = None
shutdown_event = asyncio.Event()
# Counter to keep track of the number of active prediction requests
active_requests = 0
lock = threading.Lock()  # To safely modify the counter
all_requests_done = asyncio.Event()
all_requests_done.set()
requests_received = 0
background_task_processed = 0

def change_training_ongoing(new_value):
    global training_ongoing
    training_ongoing = new_value


def get_training_ongoing():
    global training_ongoing
    return training_ongoing


def change_app_running(new_value):
    global app_running
    app_running = new_value


def get_app_running():
    global app_running
    return app_running


def get_executor():
    global executor
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=4)
    return executor


def set_executor(max_workers):
    global executor
    if executor:
        executor.shutdown(wait=True)
    executor = ThreadPoolExecutor(max_workers=max_workers)


def get_shutdown_event():
    global shutdown_event
    return shutdown_event


def reset_shutdown_event():
    global shutdown_event
    shutdown_event = asyncio.Event()


def shutdown_executor():
    global executor
    if executor:
        executor.shutdown(wait=True)
        executor = None


async def increment_requests():
    global lock
    global active_requests
    global requests_received
    global all_requests_done
    with lock:
        active_requests += 1
        requests_received += 1
        if active_requests == 1:
            all_requests_done.clear()  # Clear the event when a request starts

def get_requests_received():
    global requests_received
    return requests_received


def reset_requests_received():
    global requests_received
    requests_received = 0


async def decrement_requests():
    global lock
    global active_requests
    global all_requests_done
    with lock:
        active_requests -= 1
        if active_requests == 0:
            all_requests_done.set()

def get_requests_event():
    global all_requests_done
    return all_requests_done


def reset_requests_event():
    global all_requests_done
    all_requests_done = asyncio.Event()
    all_requests_done.set()


def get_background_task_processing():
    global background_task_processed
    return background_task_processed


def set_background_task_processing():
    global background_task_processed
    background_task_processed = 0


def increment_background_task_processing():
    global background_task_processed
    background_task_processed += 1


async def get_total_requests():
    global lock
    global active_requests
    with lock:
        return active_requests


def get_lock():
    global lock
    return lock
