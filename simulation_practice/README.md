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