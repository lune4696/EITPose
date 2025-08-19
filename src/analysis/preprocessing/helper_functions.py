import numpy as np
from typing import Optional


def scale_window_reshape(
    x_train,
    y_train,
    x_test,
    y_test,
    x_scaler: Optional[any],
    window_size: int,
    stride: int,
    dim: int = 1,
):
    """
    概要
        (必要があれば)スケーリングし、窓を適用して1~3次元のndarrayに変換する
    """
    if x_scaler:
        x_train = x_scaler.fit_transform(x_train)
        x_test = x_scaler.transform(x_test)
    x_train_temp = []
    x_test_temp = []

    for i in range(0, x_train.shape[0] - window_size + 1, stride):
        windowed_data = x_train[i : i + window_size, :]
        match dim:
            case 1:
                x_train_temp.append(windowed_data.flatten())
            case 2:
                x_train_temp.append(windowed_data)
            case 3:
                x_train_temp.append(
                    windowed_data.reshape(
                        (windowed_data.shape[0], windowed_data.shape[1], 1)
                    )
                )
            case _:
                ValueError("dimension must be in 1 ~ 3")

    x_train = np.array(x_train_temp)
    for i in range(0, x_test.shape[0] - window_size + 1, stride):
        windowed_data = x_test[i : i + window_size, :]
        match dim:
            case 1:
                x_test_temp.append(windowed_data.flatten())
            case 2:
                x_test_temp.append(windowed_data)
            case 3:
                x_test_temp.append(
                    windowed_data.reshape(
                        (windowed_data.shape[0], windowed_data.shape[1], 1)
                    )
                )
            case _:
                ValueError("dimension must be in 1 ~ 3")
    x_test = np.array(x_test_temp)

    y_train = y_train[window_size - 1 :]
    y_test = y_test[window_size - 1 :]
    return x_train, y_train, x_test, y_test
