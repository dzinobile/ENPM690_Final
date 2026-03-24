Install mujoco and deap
```bash
pip install mujoco
pip install deap
```

Run evolutionary aglorithm
```bash
pip install -r requirements.txt
python3 evolve.py --pop 30 --gens 20
```

Watch best result 
```bash
python3 replay.py
```

Watch top 3 entries
```bash
python3 replay.py --top 3
```

Run PG algorithm
```bash
pip install -r requirements.txt

python3 train_pg.py --iters 50 --eps 5

python3 replay.py --pkl pg_results.pkl
```

Run PPO algorithm
```bash
python3 train_ppo.py --iters 50 --steps 200 --epochs 10

python3 replay.py --pkl ppo_results.pkl
```