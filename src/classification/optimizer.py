#Dùng Optuna để tìm Hyperparameters tốt nhất cho 2 mô hình có Hyperparameters phức tạp RandomForestClassifier và SVC
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class ModelOptimizer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    #Hàm tìm Hyperparameters tốt nhất cho RandomForestClassifier    
    def optimize_random_forest(self, n_trials=20):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
            score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1_weighted').mean()
            return score

        study = optuna.create_study(direction='maximize') #Tối ưa hóa score
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_params['random_state'] = 42
        return RandomForestClassifier(**best_params)

    #Hàm tìm Hyperparameters tốt nhất cho SVC    
    def optimize_svc(self, n_trials=20):
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'probability': True,
                'random_state': 42
            }
            model = SVC(**params)
            score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1_weighted').mean()
            return score

        study = optuna.create_study(direction='maximize') #Tối ưa hóa score
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_params['probability'] = True
        best_params['random_state'] = 42
        return SVC(**best_params)
