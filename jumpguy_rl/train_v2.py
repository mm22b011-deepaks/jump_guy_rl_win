#!/usr/bin/env python3
import time, os, json, sys
from sim import JumpGuySim
from agent import DQNAgent

os.makedirs('checkpoints', exist_ok=True)
env = JumpGuySim()
agent = DQNAgent(state_dim=8, action_dim=2, buffer_size=50000, lr=5e-4,
                 epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.997)

stats = {'scores': [], 'epsilons': []}
best_score = 0
start = time.time()

for ep in range(1, 1001):
    state, info = env.reset()
    score = 0
    steps = 0
    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, terminated or truncated)
        agent.train_step()
        state = next_state
        score = info.get('score', 0)
        steps += 1
        if terminated or truncated or steps >= 3000:
            break
    agent.decay_epsilon()
    stats['scores'].append(score)
    stats['epsilons'].append(agent.epsilon)

    if score > best_score:
        best_score = score
        agent.save('checkpoints/best.pt')

    if ep % 50 == 0:
        recent = stats['scores'][-50:]
        avg = sum(recent) / len(recent)
        mx = max(recent)
        elapsed = time.time() - start
        print(f'Ep {ep:4d}/1000 | Last:{score:3d} Best:{best_score:3d} Avg50:{avg:5.1f} Max50:{mx:3d} Eps:{agent.epsilon:.3f} {elapsed:.0f}s', flush=True)

    if ep % 200 == 0:
        agent.save(f'checkpoints/checkpoint_ep{ep}.pt')
        with open('checkpoints/stats_v2.json', 'w') as f:
            json.dump(stats, f)

agent.save('checkpoints/final_1000.pt')
with open('checkpoints/stats_v2.json', 'w') as f:
    json.dump(stats, f)
print(f'\nDone! Best: {best_score}, Time: {time.time()-start:.0f}s', flush=True)
