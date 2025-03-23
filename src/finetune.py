import yaml
import optuna
import os
from src.environment import Environment

CONFIG_TEMPLATE = {
    "episode": 500,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.99,
    "epsilon_min": 0.01,
    "learning_rate": 0.0005,
    "max_memory": 10000,
    "batch_size": 128,
    "target_update_steps": 1000
}

CONFIG_DIR = "configs"
RESULTS_DIR = "results"

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_config(trial):
    """ Generate a new hyperparameter configuration for tuning """
    config = CONFIG_TEMPLATE.copy()
    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    config["gamma"] = trial.suggest_float("gamma", 0.90, 0.99)
    config["epsilon_decay"] = trial.suggest_float("epsilon_decay", 0.95, 0.999)
    config["max_memory"] = trial.suggest_int("max_memory", 1000, 5000)
    config["epsilon"] = trial.suggest_float("epsilon", 0.7, 1.0)
    config["epsilon_decay"] = trial.suggest_float("epsilon_decay", 0.95, 0.999)
    
    config_path = os.path.join(CONFIG_DIR, f"config_{trial.number}.yaml")
    with open(config_path, "w") as file:
        yaml.dump(config, file)
    
    return config_path

def objective(trial):
    """ Optuna objective function for hyperparameter tuning """
    config_path = create_config(trial)
    env = Environment(config_path)
    
    model_path = os.path.join(RESULTS_DIR, f"model_{trial.number}")
    os.makedirs(model_path, exist_ok=True)
    
    env.train_dqn(model_path)
    avg_reward = env.test_dqn(model_path)
    env.close()
    
    return avg_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    
    print("Best Hyperparameters:", study.best_trial.params)
    
    with open("best_config.yaml", "w") as file:
        yaml.dump(study.best_trial.params, file)
