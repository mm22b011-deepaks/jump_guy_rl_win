#!/usr/bin/env python3
"""Quick smoke test: open the game, read state, jump, verify."""

import time
import sys
sys.path.insert(0, ".")
from env import JumpGuyEnv

print("=== Jump Guy Bridge Smoke Test ===\n")

env = JumpGuyEnv(headless=False)

try:
    state, info = env.reset()
    print(f"Initial state: {state}")
    print(f"Info: {info}")

    # Read game state
    raw = env._get_state()
    print(f"\nRaw game state: {raw}")

    # Try jumping a few times
    for i in range(5):
        action = 1  # jump
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i+1}: action=jump, obs={obs}, reward={reward:.1f}, done={term}, score={info.get('score',0)}")
        time.sleep(0.5)

    # Wait and watch
    print("\nWaiting 5s to observe game...")
    time.sleep(5)

    raw = env._get_state()
    print(f"\nFinal game state: {raw}")

    print("\n=== Smoke test PASSED ===")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    env.close()
