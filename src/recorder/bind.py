import google_media_pipe

if __name__ == "__main__":
    error = google_media_pipe.test_mediapipe("Test mediapipe", 0)
    if error:
        raise error
