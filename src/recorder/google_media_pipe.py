from typing import Union

import cv2
from mediapipe import (
    solutions,
)

import time


def test_mediapipe(title: str, camera_number: int) -> Union[None, ValueError]:
    # MediaPipe Handsのセットアップ
    mediapipe_hands = solutions.hands
    hands = mediapipe_hands.Hands()
    drawing = solutions.drawing_utils

    # カメラからの映像をキャプチャ
    cap = cv2.VideoCapture(camera_number)
    if not cap.isOpened():
        return ValueError(f"Camera No.{camera_number} is not found")

    # OpenCVのウィンドウを作成
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    is_fullscreen = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 画像を水平方向に反転
        frame = cv2.flip(frame, 1)

        # 画像をRGBに変換
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 手のランドマークの検出
        results = hands.process(image)

        # 画像をBGRに戻す
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # スコアとタイマー表示領域の背景を描画
        cv2.rectangle(frame, (0, 0), (200, 100), (0, 0, 0), -1)

        # 検出結果がある場合、ランドマークを描画し、右手の位置を取得
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                drawing.draw_landmarks(
                    frame, hand_landmarks, mediapipe_hands.HAND_CONNECTIONS
                )

                # ランドマークの座標を取得（例として8番目のランドマーク=右手の人差し指の先端を使用）
                hand_x = int(
                    hand_landmarks.landmark[
                        mediapipe_hands.HandLandmark.INDEX_FINGER_TIP
                    ].x
                    * frame.shape[1]
                )
                hand_y = int(
                    hand_landmarks.landmark[
                        mediapipe_hands.HandLandmark.INDEX_FINGER_TIP
                    ].y
                    * frame.shape[0]
                )

        # テキストを表示
        cv2.putText(
            frame,
            f"Hand position: ({hand_x}, {hand_y})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Time: {time.time()}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 画像を表示
        cv2.imshow(title, frame)

        # キー入力をチェック
        key = cv2.waitKey(10) & 0xFF
        match key:
            case ord("q"):
                break
            case ord("f"):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(
                        title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                    )
                else:
                    cv2.setWindowProperty(
                        title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                    )

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    error = test_mediapipe("Test mediapipe", 0)
    if error:
        raise error
