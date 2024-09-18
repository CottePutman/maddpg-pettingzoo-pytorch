from simple_market import raw_env

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = raw_env()
    parallel_api_test(env, num_cycles=1_000_000)