# Car Racing with PPO using Gymnasium

This project trains and tests an AI agent to play the CarRacing-v3 environment using Proximal Policy Optimization (PPO) from Stable-Baselines3. The environment is powered by Gymnasium, and the project includes both training and testing workflows.

## Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.8+
- Gymnasium
- Stable-Baselines3
- Pyglet (for rendering)
- SWIG (for Box2D environment dependencies)

Install the necessary Python packages using pip:

```bash
pip install gymnasium[box2d] stable-baselines3 pyglet==1.3.2
```

## Project Structure

- `Training/`:
  - `Logs/`: Directory for TensorBoard logs.
  - `Saved Models/`: Directory for storing trained models.
- `README.md`: Project documentation.

## Training the Model

1. **Setup the Environment**

   Initialize the CarRacing-v0 environment:

   ```python
   import gymnasium as gym

   environment_name = "CarRacing-v3"
   env = gym.make(environment_name)
   ```

2. **Train the PPO Model**

   Train the model for a specified number of timesteps:

   ```python
   from stable_baselines3 import PPO
   import os

   log_path = os.path.join('Training', 'Logs')

   model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path)
   model.learn(total_timesteps=40000)
   ```

3. **Save the Model**

   Save the trained model to a specified path:

   ```python
   ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Driving_model')
   model.save(ppo_path)
   ```

## Testing the Model

1. **Load the Trained Model**

   Load the previously saved model:

   ```python
   from stable_baselines3 import PPO

   model = PPO.load("Training/Saved Models/PPO_Driving_model.zip")
   ```

2. **Test the Model**

   Use the model to play the game and observe its performance:

   ```python
   obs, info = env.reset()
   while True:
       action, _ = model.predict(obs)
       obs, reward, terminated, truncated, info = env.step(action)
       done = terminated or truncated
       env.render()

       if done:
           break

   env.close()
   ```

## Evaluation

Evaluate the trained model using Stable-Baselines3's `evaluate_policy` function:

```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean Reward: {mean_reward}, Standard Deviation: {std_reward}")

env.close()
```

## Notes

- Ensure SWIG is installed for Box2D environments: [Download SWIG](https://sourceforge.net/projects/swig/files/swigwin/swigwin-4.0.2/swigwin-4.0.2.zip/download?use_mirror=ixpeering).
- Training times and results will vary depending on your system's hardware.

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)



# Download

[Download Training Folder](https://drive.google.com/uc?id=1godxDKf5lmIzMqyDHHHP321LInLm-4ft&export=download)


