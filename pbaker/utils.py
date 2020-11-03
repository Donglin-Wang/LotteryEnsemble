from global_settings import VERBOSE


def log(msg):
    if VERBOSE > 0:
        print(msg)