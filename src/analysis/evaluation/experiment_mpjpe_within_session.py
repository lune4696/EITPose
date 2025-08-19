import sys
from pathlib import Path
from src.analysis.evaluation.experiment import (
    ExtraTreeType,
    DataScope,
    EstimationTarget,
)
from src.analysis.evaluation.experiments_runner import ExperimentRunner
from src.analysis.models.ETR import ETR
from src.analysis.preprocessing.evaluation_helper import PipelineEvaluations
import os

print(os.getcwd())

# path_to_root = Path("../../../")  # path to top level PulsePose
path_to_root = Path(".")  # path to top level PulsePose
path_to_experiment_results = (
    path_to_root / "src" / "analysis" / "evaluation" / "experiment_results"
)
sys.path.append(str(path_to_root))


experiment_params_list = [
    # Experiment 0
    {
        "experiment_params": {
            "x_scale": False,
            "y_scale": False,
            "save_model": True,
            "save_y_pred": True,
            "save_y_pred_with_x": True,
            "x_cols": {
                "eit_data": {
                    "demean": False,
                    "individual_scale": False,
                    # 'rolling_demean_no_drop': 400
                    "rolling_demean_no_drop": {
                        "U1-S1-T1": 400,
                        "U1-S1-T2": 400,
                        "U2-S1-T1": 400,
                        "U2-S1-T2": 400,
                        "U3-S1-T1": 400,
                        "U3-S1-T2": 400,
                        "U4-S1-T1": 400,
                        "U4-S1-T2": 400,
                        "U5-S1-T1": 400,
                        "U5-S1-T2": 400,
                        "U6-S1-T1": 400,
                        "U6-S1-T2": 400,
                        "U7-S1-T1": 400,
                        "U7-S1-T2": 400,
                        "U8-S1-T1": 400,
                        "U8-S1-T2": 400,
                        "U9-S1-T1": 400,
                        "U9-S1-T2": 400,
                        "U10-S1-T1": 400,
                        "U10-S1-T2": 400,
                        "U11-S1-T1": 240,
                        "U11-S1-T2": 240,
                        "U12-S1-T1": 200,
                        "U12-S1-T2": 200,
                        "U13-S1-T1": 300,
                        "U13-S1-T2": 300,
                        "U14-S1-T1": 300,
                        "U14-S1-T2": 300,
                        "U15-S1-T1": 300,
                        "U15-S1-T2": 300,
                        "U16-S1-T1": 300,
                        "U16-S1-T2": 300,
                        "U17-S1-T1": 300,
                        "U17-S1-T2": 300,
                        "U18-S1-T1": 300,
                        "U18-S1-T2": 300,
                        "U19-S1-T1": 300,
                        "U19-S1-T2": 300,
                    },
                },
            },
            "y_cols": {
                "mphands_scaled": {},
            },
            "model": ETR,
            "model_type": ExtraTreeType.Regressor,
            "model_params": {},
            "evaluate_function": PipelineEvaluations.evaluate_mpjpe,
            "estimation_target": EstimationTarget.MeanPerJointPositionError,
            "window": 1,
            "window_stride": 1,
            "data_dimension": 1,
            "gpu": "/GPU:2",
        },
        "participants": [  # inner most defines within session, then same user
            [["U1-S1-T1", "U1-S1-T2"]],
            [["U2-S1-T1", "U2-S1-T2"]],
            [["U3-S1-T1", "U3-S1-T2"]],
            [["U4-S1-T1", "U4-S1-T2"]],
            [["U5-S1-T1", "U5-S1-T2"]],
            [["U6-S1-T1", "U6-S1-T2"]],
            [["U7-S1-T1", "U7-S1-T2"]],
            [["U8-S1-T1", "U8-S1-T2"]],
            [["U9-S1-T1", "U9-S1-T2"]],
            [["U10-S1-T1", "U10-S1-T2"]],
            [["U11-S1-T1", "U11-S1-T2"]],
            [["U12-S1-T1", "U12-S1-T2"]],
            [["U13-S1-T1", "U13-S1-T2"]],
            [["U14-S1-T1", "U14-S1-T2"]],
            [["U15-S1-T1", "U15-S1-T2"]],
            [["U16-S1-T1", "U16-S1-T2"]],
            [["U17-S1-T1", "U17-S1-T2"]],
            [["U18-S1-T1", "U18-S1-T2"]],
            [["U19-S1-T1", "U19-S1-T2"]],
        ],
        "evaluations": {
            DataScope.WithinSession: 0,
        },
    },
]


testrunner = ExperimentRunner(
    experiment_params_list,
    "experiment_mpjpe_within_session",
    path_to_data=path_to_root / "src" / "data" / "processed_pose_data",
)
testrunner.run()
