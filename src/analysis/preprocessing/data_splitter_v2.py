import numpy as np
import pickle


class DataSplitter:
    """
    概要
        推論器内で使用するデータの分割を行う
    入力
        dls_list: list()
            前処理済みデータのリスト
    """

    def __init__(self, dls_list) -> None:
        self.dls_list = dls_list

    def within_session(self, x_cols, y_cols):
        for user in self.dls_list:
            # NOTE: user(session(recording)) であることに注意
            test_session = user[0][0]  # guaranteed to have one recording
            train_session = user[0][1]  # guaranteed to have one recording

            train_X = train_session.get_subset_data_concatenated(x_cols)
            test_X = test_session.get_subset_data_concatenated(x_cols)

            train_y = train_session.get_subset_data_concatenated(y_cols)
            test_y = test_session.get_subset_data_concatenated(y_cols)

            # train と test を入れ替えたものも対象にするために yield を使っている
            yield (
                train_X,
                train_y,
                test_X,
                test_y,
                [train_session.participant],
                [test_session.participant],
            )
            yield (
                test_X,
                test_y,
                train_X,
                train_y,
                [test_session.participant],
                [train_session.participant],
            )

    def within_session_k_fold(self, x_cols, y_cols, k_fold=5):
        split_ratio: float = 1 / k_fold
        for user in self.dls_list:
            split_data = []
            participant = ""

            for session in user:
                for recording in session:
                    participant = recording.participant
                    X = recording.get_subset_data_concatenated(x_cols)
                    y = recording.get_subset_data_concatenated(y_cols)
                    for k in range(k_fold):
                        start_x = int(k * split_ratio * len(X))
                        end_x = int((k + 1) * split_ratio * len(X))
                        start_y = int(k * split_ratio * len(y))
                        end_y = int((k + 1) * split_ratio * len(y))

                        split_data.append([X[start_x:end_x], y[start_y:end_y]])

            for k in range(k_fold):
                test_X, test_y = split_data[k]
                train_X = train_y = None
                for k2 in range(k_fold):
                    if k != k2:
                        if train_X is None:
                            train_X, train_y = split_data[k2]
                        else:
                            X_train, y_train = split_data[k2]
                            train_X = np.concatenate((train_X, X_train))
                            train_y = np.concatenate((train_y, y_train))
                yield train_X, train_y, test_X, test_y, [participant], [str(k)]

    def create_within_session_warm_start_split_generator(self, x_cols, y_cols):
        for user_ind in range(len(self.dls_list)):
            user = self.dls_list[user_ind]
            test_session = user[0][0]  # guaranteed to have one recording
            train_session = user[0][1]  # guaranteed to have one recording

            cold_model_participant_list = []
            train_X_cold = None
            train_y_cold = None
            for user_ind2 in range(len(self.dls_list)):
                if user_ind2 != user_ind:
                    for window in self.dls_list[user_ind2]:
                        cold_model_participant_list.append(window.participant)
                        X_train, y_train = (
                            window.get_subset_data_concatenated(x_cols),
                            window.get_subset_data_concatenated(y_cols),
                        )
                        if train_X_cold is None:
                            train_X_cold = X_train
                            train_y_cold = y_train
                        else:
                            train_X_cold = np.concatenate((train_X_cold, X_train))
                            train_y_cold = np.concatenate((train_y_cold, y_train))

            train_X_warm = train_session.get_subset_data_concatenated(x_cols)
            test_X = test_session.get_subset_data_concatenated(x_cols)
            train_y_warm = train_session.get_subset_data_concatenated(y_cols)
            test_y = test_session.get_subset_data_concatenated(y_cols)
            yield train_X_warm, train_y_warm, test_X, test_y, train_X_cold, train_y_cold
            yield test_X, test_y, train_X_warm, train_y_warm, train_X_cold, train_y_cold

    def cross_session(self, x_cols, y_cols):
        # train A1 and A2 together, test A3; train A3, test A1 and A2 together
        for user in self.dls_list:
            session1 = user[0]  # guaranteed to have two recording
            session2 = user[1]  # guaranteed to have one recording

            participant_list = []
            train_X = None
            train_y = None
            for recording in session1:
                participant_list.append(recording.participant)
                X_train, y_train = (
                    recording.get_subset_data_concatenated(x_cols),
                    recording.get_subset_data_concatenated(y_cols),
                )
                if train_X is None:
                    train_X = X_train
                    train_y = y_train
                else:
                    train_X = np.concatenate((train_X, X_train))
                    train_y = np.concatenate((train_y, y_train))

            test_X = session2[0].get_subset_data_concatenated(x_cols)
            yield (
                train_X,
                train_y,
                test_X,
                session2[0].get_subset_data_concatenated(y_cols),
                participant_list,
                [session2[0].participant],
            )
            yield (
                test_X,
                session2[0].get_subset_data_concatenated(y_cols),
                train_X,
                train_y,
                [session2[0].participant],
                participant_list,
            )

    def cross_user(self, x_cols, y_cols):
        for user_ind in range(len(self.dls_list)):
            test_X = None
            test_y = None
            train_X = None
            train_y = None
            test_participant_list = []
            train_participant_list = []

            user = self.dls_list[user_ind]
            for session in user:
                for recording in session:
                    test_participant_list.append(recording.participant)
                    X_test, y_test = (
                        recording.get_subset_data_concatenated(x_cols),
                        recording.get_subset_data_concatenated(y_cols),
                    )
                    if test_X is None:
                        test_X = X_test
                        test_y = y_test
                    else:
                        test_X = np.concatenate((test_X, X_test))
                        test_y = np.concatenate((test_y, y_test))

            for user_ind2 in range(len(self.dls_list)):
                if user_ind2 != user_ind:
                    user2 = self.dls_list[user_ind2]
                    for session2 in user2:
                        for recording2 in session2:
                            train_participant_list.append(recording2.participant)
                            X_train, y_train = (
                                recording2.get_subset_data_concatenated(x_cols),
                                recording2.get_subset_data_concatenated(y_cols),
                            )
                            if train_X is None:
                                train_X = X_train
                                train_y = y_train
                            else:
                                train_X = np.concatenate((train_X, X_train))
                                train_y = np.concatenate((train_y, y_train))

            yield (
                train_X,
                train_y,
                test_X,
                test_y,
                train_participant_list,
                test_participant_list,
            )

    def create_X_y_from_window_list(self, data_type):
        X = None
        y = None
        for window_user in self.window_list:
            for session in window_user:
                if data_type == "flattened_raw_data":
                    window_X = session.X_flattened
                elif data_type == "feature_data":
                    window_X = session.X_features
                window_y = session.y
                if X is None:
                    X = window_X
                    y = window_y
                else:
                    X = np.concatenate((X, window_X))
                    y = np.concatenate((y, window_y))
        return X, y

    @staticmethod
    def save_scaler(scaler, file_name):
        pickle.dump(scaler, open(file_name, "wb"))

    @staticmethod
    def save_data(X, y, file_name):
        pickle.dump({"X": X, "y": y}, open(file_name, "wb"))
