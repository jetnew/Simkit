import numpy as np
import pandas as pd
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
init_notebook_plotting()
from performance.performance import prob_overlap, kl, js

def hyperopt(model, params, opt_params, X, y, trials=30, val_split=0.8):
    X_train, X_test = np.split(X, [int(val_split*X.shape[0])])
    y_train, y_test = np.split(y, [int(val_split*y.shape[0])])
    def loss_function(p):
        m = model(x_features=params['x_features'],
                  y_features=params['y_features'], **p)
        m.fit(X_train, y_train, epochs=params['epochs'], verbose=False)
        p, q = prob_overlap(y, m.predict(X))
        return {
            'loss': (m.loss(X_test, y_test), 0.0),
            'fKL': (kl(p, q), 0.0),
            'rKL': (kl(q, p), 0.0),
            'JS': (js(p, q), 0.0)}
    best_params, best_vals, experiment, exp_model = optimize(
        parameters=[{'name': name, 'type': 'range', 'bounds': bounds}
                    for name, bounds in opt_params.items()],
        evaluation_function=loss_function,
        objective_name="loss",
        minimize=True,
        total_trials=trials)
    
    m = model(x_features=params['x_features'],
              y_features=params['y_features'], **best_params)
    m.fit(X, y, params['epochs'], verbose=False)
    return m, best_params, best_vals, experiment, exp_model


def hyperopt_log(experiment):
    # Get parameter set for every trial
    df_experiment = pd.DataFrame([trial.arm.parameters for trial in experiment.trials.values()])
    
    # Get metrics for every trial
    df_metrics = experiment.fetch_data().df
    metric_names = df_metrics['metric_name'].unique()
    for metric_name in metric_names:
        metric_series = df_metrics[df_metrics['metric_name'] == metric_name]['mean'].reset_index(drop=True)
        df_experiment[metric_name] = metric_series
        
    return df_experiment


def hyperparam_plot(exp_model, param_x, param_y):
    render(plot_contour(exp_model, param_x, param_y, metric_name='loss'))
    
def performance_plot(experiment, best_vals):
    best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
    best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(best_objectives, axis=1),
        optimum=best_vals[0]['loss'],
        title="Model performance vs. # of iterations",
        ylabel="loss")
    render(best_objective_plot)