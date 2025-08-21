from pathlib import Path
import sys
import pickle
from datetime import datetime
from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing

from src.analysis.preprocessing.data_splitter_v2 import DataSplitter
from src.analysis.preprocessing.helper_functions import scale_window_reshape

path_to_root = Path("../../../")  # path to top level PulsePose
path_to_models = path_to_root / "src" / "analysis" / "models"
path_to_y_pred = path_to_root / "src" / "analysis" / "y_pred"
sys.path.append(str(path_to_root))


class ExtraTreeType(Enum):
    """
    概要
        推論器 (ExtraTree) の種類を示す enum
    """

    Regressor = 0
    Classifier = 1


class DataScope(Enum):
    """
    概要
        推論器が評価するデータの範囲を示す enum
    """

    WithinSession = 0
    CrossSession = 1
    CrossUser = 2
    WithinSessionKFold = 3


class EstimationTarget(Enum):
    """
    概要
        推論器が評価する対象を示す enum
    """

    MeanPerJointPositionError = 0
    Gesture = 1
    WristAngle = 2


class Experiment:
    """
    概要
        教師データ(EIT計測データ (電圧) + Hand Land Marker(真)) からHand Land Marker を推論する
    入力
        data: list()
            前処理済みデータのリスト (推論器の入力の元)
            構造は以下の通り(cf. ./experiments_runner.py)
                list(user(session(recording))) = experiment["participants"]
        experiment_params: dict()
            評価器の設定用パラメータ
        evaluation: str
            評価器の評価モード
        num_round: int
            入力データの内、推論対象とするデータ数
        print_fcn: function
            途中経過をプリントアウトする関数
        output_folder: str
            出力先ディレクトリ
    インスタンス変数
    """

    def __init__(
        self,
        data: np.ndarray,
        parameters: dict[str, any],
        data_scope: DataScope,
        num_rounds: int = 0,
        print_function: any = print,
        output_folder: Optional[str] = None,
    ):
        self.data = data

        self.parameters = parameters
        self.x_cols = parameters["x_cols"]
        self.y_cols = parameters["y_cols"]
        self.model = parameters["model"]
        self.model_type: Union[ExtraTreeType] = parameters["model_type"]
        self.evaluate: any = parameters["evaluate_function"]
        self.estimation_target: EstimationTarget = parameters["estimation_target"]
        self.window: int = parameters["window"]
        self.window_stride: int = parameters["window_stride"]
        self.data_dimension: int = parameters["data_dimension"]
        self.gpu: str = parameters["gpu"] if "gpu" in parameters.keys(
        ) else "/GPU:0"

        self.data_scope = data_scope

        self.num_rounds = num_rounds
        self.print = print_function
        self.output_folder = output_folder

        self.data_splitter = DataSplitter(self.data)

        match self.data_scope:
            case DataScope.WithinSession:
                self.splitter = self.data_splitter.within_session
            case DataScope.CrossSession:
                self.splitter = self.data_splitter.cross_session
            case DataScope.CrossUser:
                self.splitter = self.data_splitter.cross_user
            case DataScope.WithinSessionKFold:
                self.splitter = self.data_splitter.within_session_k_fold
            case _:
                raise ValueError(
                    "evaluation target should be selected from 'DataScope'"
                )

    def print_data_shape(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        note: str = "",
    ):
        """
        概要
            評価対象のデータ形状をプリントする関数
        """
        self.print("Data shapes ({}): ".format(note))
        self.print(
            "    "
            + "x_train: {}, y_train: {}, x_test: {}, y_test: {}\n".format(
                x_train.shape, y_train.shape, x_test.shape, y_test.shape
            )
        )

    def print_participants_name(
        self,
        train_participants: list[str],
        test_participants: list[str],
    ):
        """
        概要
            評価対象のデータの被験者名(xx_participants)をプリントする関数
        """
        if self.data_scope == DataScope.WithinSessionKFold:
            self.print(
                "Evaluating: ",
                "".join(train_participants),
                "K: ",
                "".join(test_participants),
            )
        else:
            self.print(
                "Train on: ",
                "".join(train_participants),
                "Test on: ",
                "".join(test_participants),
            )

    def estimate_by_classifier(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        acc_list,
        cls_rep_list,
        conf_matrix_list,
    ):
        model = self.model()
        model.set_params(self.parameters["model_params"])
        model = model.train(x_train, y_train, x_val, y_val)

        y_pred = model.predict(x_test)

        acc, cls_rep, conf_matrix = self.evaluate(y_test, y_pred)
        self.print("Accuracy: ", acc)
        self.print(cls_rep)
        self.print(conf_matrix)
        acc_list.append(acc)
        cls_rep_list.append(cls_rep)
        conf_matrix_list.append(conf_matrix)

        return model, y_pred

    def estimate_by_regressor(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        error_per_joint_list,
        std_per_joint_list,
    ):
        model = self.model()
        model.set_params(self.parameters["model_params"])
        model = model.train(x_train, y_train, x_val, y_val)

        y_pred = model.predict(x_test)

        match self.estimation_target:
            case EstimationTarget.MeanPerJointPositionError:
                (
                    error_per_joint,
                    std_per_joint,
                    error_total,
                    std_total,
                    y_pred,
                ) = self.evaluate(y_test, y_pred)

                error_per_joint_list.append(error_per_joint)
                std_per_joint_list.append(std_per_joint)

            # "wrist_angle" は現在使用されていないのでcallされることは無い
            case EstimationTarget.Gesture | EstimationTarget.WristAngle:
                NotImplementedError(
                    "Operation for {} is not implemented".format(
                        self.estimation_target)
                )
                # (
                #     error_per_joint,
                #     std_per_joint,
                #     y_pred,
                # ) = self.evaluate(y_test, y_pred)

            case _:
                ValueError(
                    "evaluation target {} is not valid".format(
                        self.estimation_target)
                )

        return model, y_pred

    def record_model_info(
        self,
        model,
        train_participants,
        test_participants,
        x_scaler,
        y_scaler,
        participants_model_pairs,
        x_scaler_list,
        y_scaler_list,
    ):
        participants_model_pairs.append(
            [
                "".join(train_participants),
                "".join(test_participants),
                model,
            ]
        )
        if x_scaler:
            x_scaler_list.append(
                [
                    "".join(train_participants),
                    "".join(test_participants),
                    x_scaler,
                ]
            )

        if y_scaler:
            y_scaler_list.append(
                [
                    "".join(train_participants),
                    "".join(test_participants),
                    y_scaler,
                ]
            )

    # TODO: 推論結果を dict[str, ndarray]でEITデータ(入力)と共に出力できるように
    def save_y_pred(
        self,
        y_pred,
        filename,
        train_participants,
        test_participants,
    ):
        Path.mkdir(filename, parents=True, exist_ok=True)
        _filename = str(
            filename
            / str("".join(train_participants) + "_" + "".join(test_participants))
        )
        np.save(
            _filename,
            y_pred,
        )

    def save_y_pred_with_x(
        self,
        x,
        y_pred,
        y_pred_scaled,
        filename,
        train_participants,
        test_participants,
    ):
        print(x.shape)
        print(y_pred.shape)
        # dataframe の各行にその時点でのデータが入る形でデータが保管される
        save_data = pd.DataFrame(
            {
                "eit_data": [row for row in x],
                "mphands_data": [row for row in y_pred],
                "mphands_scaled": [row for row in y_pred_scaled],
            }
        )
        Path.mkdir(filename, parents=True, exist_ok=True)
        _filename = str(
            filename
            / str(
                "".join(train_participants) + "_" +
                "".join(test_participants) + ".pkl"
            )
        )
        with open(_filename, "wb") as f:
            pickle.dump(save_data, f)

    def save_model(
        self,
        filename,
        participants_model_pairs,
        x_scaler_list,
        y_scaler_list,
    ):
        Path.mkdir(filename, parents=True, exist_ok=True)

        for trainname, testname, model in participants_model_pairs:
            _filename = filename / ("_" + trainname + "_" + testname + ".pkl")
            model.save_model(_filename)
            self.print("Model is saved in: {}".format(_filename))

        for trainname, testname, x_scaler in x_scaler_list:
            _filename = filename / \
                ("_xscaler_" + trainname + "_" + testname + ".pkl")
            pickle.dump(x_scaler, open(str(_filename), "wb"))
            self.print("x_scaler is saved in: {}".format(_filename))

        for trainname, testname, y_scaler in y_scaler_list:
            _filename = filename / \
                ("_yscaler_" + trainname + "_" + testname + ".pkl")
            pickle.dump(y_scaler, open(str(_filename), "wb"))
            self.print("y_scaler is saved in: {}".format(_filename))

    def run_experiment(self):
        """
        概要
            推論を行う関数
        入力
            self
        出力
            match self.evaluator_type:
                case "classifier":
                    tuple(
                        np.mean(error_per_joint_list),
                        np.std(np.mean(error_per_joint_list, axis=0)),
                        np.mean(error_per_joint_list, axis=0),
                        np.std(error_per_joint_list, axis=0),
                        np.mean(error_per_joint_list, axis=1),
                        np.std(error_per_joint_list, axis=1),
                        filename,
                    )
                case "regressor":
                    tuple (
                        np.mean(acc_list),
                        np.std(acc_list),
                        acc_list,
                        cls_rep_list,
                        conf_matrix_list,
                        filename,
                    )

        """

        self.print("Running Experiment: ")

        # Regressor (for Mean Per Joint Position Error) のみ使用
        error_per_joint_list = []
        std_per_joint_list = []

        # Classifier (for Gesture) のみ使用
        acc_list = []
        cls_rep_list = []
        conf_matrix_list = []

        participants_model_pairs = []  # データの被験者名とそれを使用して学習したモデルを記録
        x_scaler_list = []
        y_scaler_list = []

        estimation_target = self.estimation_target
        modelname = self.model.MODELNAME

        window_size = self.window
        stride = self.window_stride
        shape = self.data_dimension
        round = 0

        for (
            x_train,
            y_train,
            x_test,
            y_test,
            train_participants,
            test_participants,
        ) in self.splitter(self.x_cols, self.y_cols):
            # Some data alterations creates nan's remove rows

            self.print_data_shape(x_train, y_train, x_test, y_test, "raw")

            filter_train = ~np.all(
                np.isnan(x_train),
                axis=1,
            ) & ~np.all(
                np.isnan(y_train),
                axis=1,
            )
            filter_test = ~np.all(
                np.isnan(x_test),
                axis=1,
            ) & ~np.all(
                np.isnan(y_test),
                axis=1,
            )
            x_train = x_train[filter_train]
            y_train = y_train[filter_train]
            x_test = x_test[filter_test]
            y_test = y_test[filter_test]

            self.print_data_shape(
                x_train, y_train, x_test, y_test, "nan removed")

            if self.model_type == ExtraTreeType.Classifier:
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)

            x_scaler = (
                preprocessing.MinMaxScaler(
                ) if self.parameters["x_scale"] else None
            )
            y_scaler = (
                preprocessing.MinMaxScaler if self.parameters["y_scale"] else None
            )
            x_train, y_train, x_test, y_test = scale_window_reshape(
                x_train,
                y_train,
                x_test,
                y_test,
                x_scaler,
                window_size,
                stride,
                shape,
            )

            if y_scaler:
                y_train = y_scaler.fit_transform(y_train)
                y_test = y_scaler.transform(y_test)

            # これは modelname が extratreesregressor/classifier ではないときの分岐だが、
            # 何のために存在するかわからないのでコメントアウト
            # x_validation_end = len(x_test) / 2
            # y_validation_end = len(y_test) / 2
            # x_val = x_test[0:x_validation_end]
            # x_test = x_test[x_validation_end:]
            # y_val = y_test[0:y_validation_end]
            # y_test = y_test[y_validation_end:]
            x_val = None
            y_val = None

            self.print_data_shape(x_train, y_train, x_test,
                                  y_test, "preprocessed")
            self.print_participants_name(train_participants, test_participants)

            # 学習・推論の開始
            match self.model_type:
                case ExtraTreeType.Classifier:
                    model, y_pred = self.estimate_by_classifier(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        x_test,
                        y_test,
                        acc_list,
                        cls_rep_list,
                        conf_matrix_list,
                    )

                case ExtraTreeType.Regressor:
                    model, y_pred = self.estimate_by_regressor(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        x_test,
                        y_test,
                        error_per_joint_list,
                        std_per_joint_list,
                    )

                case _:
                    ValueError(
                        "model type: {} is not valid".format(self.model_type))

            # 出力スケーリングを行っていた場合、それを補填
            if y_scaler:
                y_pred = y_scaler.inverse_transform(y_pred)
                y_test = y_scaler.inverse_transform(y_test)

            self.record_model_info(
                model,
                train_participants,
                test_participants,
                x_scaler,
                y_scaler,
                participants_model_pairs,
                x_scaler_list,
                y_scaler_list,
            )

            if self.parameters["save_y_pred"]:
                # Get the current date and time
                now = datetime.now()
                # Convert date/time to a str as YYYYMMDD_HHMMSS
                save_time = now.strftime("%Y%m%d_%H%M%S")
                if self.output_folder:
                    filename = self.output_folder / "y_pred"
                else:
                    filename = (
                        path_to_y_pred / modelname / estimation_target / self.data_scope
                    )
                self.save_y_pred(
                    y_pred,
                    filename,
                    train_participants,
                    test_participants,
                )

            if self.parameters["save_y_pred_with_x"]:
                # Get the current date and time
                now = datetime.now()
                # Convert date/time to a str as YYYYMMDD_HHMMSS
                save_time = now.strftime("%Y%m%d_%H%M%S")
                if self.output_folder:
                    filename = self.output_folder / "y_pred_with_x"
                else:
                    filename = (
                        path_to_y_pred / modelname / estimation_target / self.data_scope
                    )
                self.save_y_pred_with_x(
                    x_test,
                    y_pred,
                    y_pred,
                    filename,
                    train_participants,
                    test_participants,
                )

            if self.num_rounds != 0 and (round >= self.num_rounds):
                break

        # ロードした全データについて学習・推論を終えた後の終了処理
        #
        if self.parameters["save_model"]:
            filename = "None"
            now = datetime.now()
            # Convert date/time to a string in the format YYYYMMDD_HHMMSS
            save_time = now.strftime("%Y%m%d_%H%M%S")
            if self.output_folder:
                filename = self.output_folder / "models"
            else:
                filename = (
                    path_to_models
                    / modelname
                    / estimation_target
                    / self.data_scope
                    / save_time
                )
            self.save_model(
                filename,
                participants_model_pairs,
                x_scaler_list,
                y_scaler_list,
            )

        match self.model_type:
            case ExtraTreeType.Classifier:
                return (
                    np.mean(acc_list),
                    np.std(acc_list),
                    acc_list,
                    cls_rep_list,
                    conf_matrix_list,
                    filename,
                )
            case ExtraTreeType.Regressor:
                error_per_joints: np.ndarray = np.array(error_per_joint_list)
                std_per_joints: np.ndarray = np.array(std_per_joint_list)
                if self.estimation_target == EstimationTarget.MeanPerJointPositionError:
                    self.print("Mean Error: {}".format(
                        np.mean(error_per_joints)))
                    self.print("Std Error: {}".format(np.mean(std_per_joints)))
                    return (
                        np.mean(error_per_joints),
                        np.std(np.mean(error_per_joints, axis=0)),
                        np.mean(error_per_joints, axis=0),
                        np.std(error_per_joints, axis=0),
                        np.mean(error_per_joints, axis=1),
                        np.std(error_per_joints, axis=1),
                        filename,
                    )
                else:
                    NotImplementedError()
            case _:
                ValueError()
