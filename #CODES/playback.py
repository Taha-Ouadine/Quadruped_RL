import sys
from absl import app
from learning import train_jax_ppo

exit()

def main(argv):
    # Executes the core playground rendering and inference loop
    train_jax_ppo.main(argv)

if __name__ == "__main__":
    # Override system arguments with your configuration
    sys.argv = [
        "playback.py",
        "--env_name=CartpoleBalance",
        "--play_only=True",
        "--load_checkpoint_path=/home/tahao_mw5iph3/linuxrobotic/mujoco_playground/logs/CartpoleBalance-20260917-183939/checkpoints"
    ]
    
    # Launch the app
    app.run(main)