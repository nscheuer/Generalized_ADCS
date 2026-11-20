class StateMachine:
    def __init__(self, initial_mode="idle"):
        self.mode = initial_mode

    def update(self, state, time):
        raise NotImplementedError
