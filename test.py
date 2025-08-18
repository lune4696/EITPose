from pathlib import Path
import pickle as pkl
from typing import NamedTuple, Self, Union

class WindowSize(NamedTuple):
    x: float
    y: float

class GUIConfig():
    def __init__(self) -> Self:
        self.windowsize = WindowSize(0, 0)

    @classmethod
    def load(cls, path: Path) -> Union[Self, FileNotFoundError, Exception]: 
        with path.open(mode="rb") as f:
            return pkl.load(f)

    def dump(self, path: Path) -> None: 
        with path.open(mode="wb") as f:
            return pkl.dump(self, f)

class GUIUnit():
    def __init__(self, config: GUIConfig) -> None:
        self.windowsize = config.windowsize

    def update(self) -> Union[None, Exception]:
        pass

    def render(self) -> None:
        pass

def main() -> None:
    # test GUIConfig dump
    path: Path = Path.cwd() / Path("config.pkl")
    config = GUIConfig()
    config.windowsize = WindowSize(100, 0)
    config.dump(path)

    # test GUIConfig load
    path: Path = Path.cwd() / Path("config.pkl")
    config = GUIConfig.load(path)
    match config: 
        case config if type(config) is FileNotFoundError:
            print(f"Error: File not found at {path._str}")
            return None
        case config if type(config) is Exception:
            print(f"Error loading pickle file: {type(config)}")
            return None
    
    unit = GUIUnit(config)
    assert unit.windowsize == WindowSize(100, 0)

if __name__ == "__main__":
    main()
