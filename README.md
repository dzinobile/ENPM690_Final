# ENPM690 Group 1 Final Project 
## Objective
## Instructions for Use
- get docker compose
- add yourself to the docker group, otherwise you have to always run docker commands with sudo  
```bash
sudo usermod -aG docker $USER
```
- Build docker container (this will take a while)
```bash
docker compose build 
```

- Do this so you can see the display:
```bash
xhost +local:docker
```
- Train walk
```bash
docker compose run --rm train-walk python train_tars_walk.py --timesteps 100000 --n-envs 4
```
- View walk results
```bash
docker compose run --rm replay-walk python replay_tars_walk.py --model best_model_walk/best_model.zip
```
- Train carry
```bash
docker compose run --rm train-carry python train_tars_carry.py --timesteps 100000 --n-envs 4
```

- View carry results
```bash
docker compose run --rm replay-carry python replay_tars_carry.py --model best_model_carry/best_model.zip
```

- Simulate
```bash
docker compose run --rm sim python tars_sim.py --xml tars.xml
```

- Display checkpoints (you need to change the hardcoded checkpoint file names and rebuild the container, unfortunately)
```bash
docker compose run --rm replay-checkpoints
```
