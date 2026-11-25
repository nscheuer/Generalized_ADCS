class StateMachine:
    def __init__(self, initial_mode: str = "idle") -> None:
        self.mode = initial_mode

    def update(self, state: object, time: float) -> None:
        raise NotImplementedError
