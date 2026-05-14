# ENPM690 Group 1 Final Project 
## Objective
This project explores the application of reinforcemnt learning to a robot inspired by TARS from the movie Interstellar. The goal was to train the robot to carry a human steadily while walking forward, in order to replicate a scene in the movie. The unusual design of this robot makes it a good case for observing how reinforcement learning can develop different methods of locomotion.

## Method
The robot was modeled in Solidworks to create meshes. These were imported into MuJoCo xml files to create the model. There are two models, one for just training the robot to walk, and one for training it to walk while carrying a human. The xml files define various keyframes for starting the robot in different configurations. The robot's "torso" is defined by a base link at the centroid.

Training was done with a PPO algorithm with GAE. Simulations are run in multiple parallel environments for many trials, until a given number of total timesteps has been collected across all environments. Each timestep represents 10 ms, and a trial may run for a maximum of 10 s (simulation time). A trial ends either when it meets one of the termination conditions, or it reaches this maximum time.

The reward function included a reward for forward velocity, a small per-timestep reward for avoiding termination conditions, and a reward for keeping the human held parallel to the ground. It included a penalty for pitch and yaw to keep the robot upright and moving forward, and very small penalties for energy and large actions. 

The termination conditions were:
- Excess torso pitch, to prevent the robot from lying on its side and shuffling forward
- Torso height too low, to preven the robot from falling flat on its back
- Human head or feet touch ground, to prevent dropping the human
- Human height to high, to prevent the robot from throwing the human
- Robot arms touching the ground, to prevent an arm-slamming movement that the robot developed early on

An optional --variable argument causes the start condition to randomly vary between keyframes at the beginning of each new trial. This approach was inspired by Mirror Descent Guided Policy Search, an algorithm that trains a global policy from various local policies. Sufficient training with variable start conditions created a model that could perform well starting from any of those conditions. 



## Usage
### Requirements
- Linux host with an X11 display server 
- Install Docker + Docker Compose:  https://docs.docker.com/compose/
### Set up docker
After getting docker, add yourself to the docker group so you don't have to use sudo for every command.  
```bash
sudo usermod -aG docker $USER
```
Build the docker container.
```bash
docker compose build 
```
Setup up local display for visualization.
```bash
xhost +local:docker
```

### Train Walk
Train the TARS robot from tars.xml to walk forward.
```bash
docker compose run --rm train-walk python train_tars_walk.py --timesteps 100000 --n-envs 4 --variable
```
**Arguments:**
| Argument | Description |
|----------|-------------|
| `--timesteps <number>` | Total timesteps across all environments to run training |
| `--n-envs <number>` | Number of parallel environments |
| `--resume <checkpoint filename>` | Resume training from a saved checkpoint |
| `--variable` | Randomly varies trial start conditions between keyframes |

### Replay Walk
Simulate behavior of a given walk-trained model in MuJoCo. If you used variable start, change the starting keyframe until you find the right one. 
```bash
docker compose run --rm replay-walk python replay_tars_walk.py --model best_model_walk/best_model.zip --keyframe 0
```
**Arguments:**
| Argument | Description |
|----------|-------------|
| `--model <model filename>` | Trained model to be replayed |
| `--keyframe <keyframe number>` | Sets start condition of simulation to one of the keyframes |

### Train Carry
Train TARS robot from tars_with_human.xml to walk forward while carrying human.
```bash
docker compose run --rm train-carry python train_tars_carry.py --timesteps 100000 --n-envs 4 --variable
```
**Arguments:**
| Argument | Description |
|----------|-------------|
| `--timesteps <number>` | Total timesteps across all environments to run training |
| `--n-envs <number>` | Number of parallel environments |
| `--resume <checkpoint filename>` | Resume training from a saved checkpoint |
| `--variable` | Randomly varies trial start conditions between keyframes |

### Replay Carry
Simulate behavior of a given carry-trained model in MuJoCo. If you used variable start, change the starting keyframe until you find the right one. 
```bash
docker compose run --rm replay-carry python replay_tars_carry.py --model best_model_carry/best_model.zip --keyframe 0
```
**Arguments:**
| Argument | Description |
|----------|-------------|
| `--model <model filename>` | Trained model to be replayed |
| `--keyframe <keyframe number>` | Sets start condition of simulation to one of the keyframes |

### Simulate XML File
Simulate a given XML file in MuJoCo with sliders to manually position the joints.
```bash
docker compose run --rm sim python tars_sim.py --xml tars.xml --float
```
**Arguments:**
| Argument | Description |
|----------|-------------|
| `--xml <xml filename>` | Which XML file to display |
| `--float` | Spawns the robot floating in midair so that moving the joints doesn’t tip it over |

### Replay Checkpoints
Simulate a series of hard-coded checkpoints to view progression of trained model. You may need to change the hard-coded checkpoint names and run `docker compose build` again. 
```bash
docker compose run --rm replay-checkpoints
```
### View Tensorboard Info
View Tensorboard logs with critical parameters plotted over timesteps.
```bash
tensorboard --logdir=logs/walk/
# OR
tensorboard --logdir=logs/carry/
# If that doesn't work try --logdir ./logs/carry/ structure instead
```
Then open http://localhost:6006/ in your browser.

### Delete All Docker Images and Containers
```bash
docker compose down
docker system prune
docker images -a #view any remaining images
docker rmi -f <image_id> #remove remaining images
docker ps -a #view any remaining containers
docker rm -f <container_id> #remove any remaining containers

```
