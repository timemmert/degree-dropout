import optuna
from optuna import Trial

from main import train_and_validate


def objective(trial: Trial):
    n_train = 20
    dropout_probability = trial.suggest_float("dropout_probability", 0, 1)
    last_validation_loss = train_and_validate(
        dropout_probability=dropout_probability,
        n_train=n_train,
    )
    return last_validation_loss


study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="find-dropout-probability-small-mnist",
    direction="minimize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)

study.best_params
