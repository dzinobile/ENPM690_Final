# ENPM690 Group 1 Final Project 
## Objective
The goal of this project was to apply reinforcemnt learning to a robot inspired by TARS from the movie Interstellar. The unusual design of this robot makes it a good case for exploring how reinforcement learning can develop different methods of locomotion. 
## Method

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
- Train walk example
```bash
docker compose run --rm train-walk python train_tars_walk.py --timesteps 100000 --n-envs 4 --variable
```
- View walk results example (if you used variable start, change the starting keyframe until you find the right one)
```bash
docker compose run --rm replay-walk python replay_tars_walk.py --model best_model_walk/best_model.zip --keyframe 0
```
- Train carry example
```bash
docker compose run --rm train-carry python train_tars_carry.py --timesteps 100000 --n-envs 4 --variable
```

- View carry results example (if you used variable start, change the starting keyframe until you find the right one)
```bash
docker compose run --rm replay-carry python replay_tars_carry.py --model best_model_carry/best_model.zip --keyframe 0
```

- Simulate
```bash
docker compose run --rm sim python tars_sim.py --xml tars.xml --float
```

- Display checkpoints (you need to change the hardcoded checkpoint file names and rebuild the container and then do docker compose build again)
```bash
docker compose run --rm replay-checkpoints
```

- View tensorboard example
```bash
tensorboard --logdir=logs/walk/
```
Then open http://localhost:6006/ in your browser

- Delete all of your docker containers, etc:
```bash
docker system prune
docker images -a #view any remaining images
docker rmi -f <image_id> #remove remaining images
docker ps -a #view any remaining containers
docker rm -f <container_id> #remove any remaining containers

```
