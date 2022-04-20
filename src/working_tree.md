```shell
.
├── __init__.py
├── benchmarking
│   ├── __init__.py
│   └── benchmarking.py
├── designs_of_experiments
│   ├── __init__.py
│   ├── design_library
│   │   ├── __init__.py
│   │   ├── d_design.py
│   │   ├── latin_hypercube.py
│   │   ├── minimum_entry_of_CRLB_design.py
│   │   ├── pi_design.py
│   │   └── random.py
│   └── interfaces
│       ├── __init__.py
│       └── design_of_experiment.py
├── metrics
│   ├── __init__.py
│   ├── error_functions
│   │   ├── __init__.py
│   │   ├── average_error.py
│   │   └── mean_squared_error.py
│   ├── interfaces
│   │   ├── __init__.py
│   │   ├── error_function.py
│   │   └── metric.py
│   └── metric_library
│       ├── __init__.py
│       ├── determinant_of_fisher_information_matrix.py
│       ├── estimation_mean_error.py
│       ├── estimation_mean_parameter_estimations.py
│       ├── estimation_variance_parameter_estimations.py
│       ├── k_fold_cross_validation.py
│       └── leave_one_out_validation.py
├── minimizer
│   ├── __init__.py
│   ├── interfaces
│   │   ├── __init__.py
│   │   └── minimizer.py
│   └── minimizer_library
│       ├── __init__.py
│       ├── differential_evolution.py
│       └── slsqp.py
├── parametric_function_library
│   ├── __init__.py
│   ├── aging_model_Naumann.py
│   ├── interfaces
│   │   ├── __init__.py
│   │   └── parametric_function.py
│   └── linear_function.py
├── statistical_models
│   ├── __init__.py
│   ├── interfaces
│   │   ├── __init__.py
│   │   └── statistical_model.py
│   └── statistical_model_library
│       ├── __init__.py
│       └── gaussian_noise_model.py
└── visualization
    ├── __init__.py
    ├── create_dashboard.py
    └── plotting_functions.py
```
