class Task:
    def __init__(
        self,
        name: str,
        callback: callable,
        rate_hz: float,
        wcet: float,
        priority: int,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.callback = callback
        self.rate_hz = rate_hz
        self.period = 1.0 / rate_hz

        self.wcet = wcet
        self.priority = priority
        self.enabled = enabled

        self.next_release_time = 0.0
        self.exec_count = 0
