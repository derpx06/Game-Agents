import torch.multiprocessing as mp
from GameManager import GameManager


def main():
    """
    Initializes and runs the game manager, which handles the main menu and all game modes.
    The manager will decide whether to run in single-process or multi-process mode.
    """
    # Required for PyTorch multiprocessing on some systems
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    game_manager = GameManager()
    game_manager.run()


if __name__ == "__main__":
    main()