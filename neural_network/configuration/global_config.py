from .driver import Driver

class GlobalConfig:
    _configuration = {}

    def __init__(self):
        self.__dict__ = self._configuration

    def get_driver(self) -> Driver:
        return self._configuration.get('driver', None)

    def set_driver(self, driver: Driver):
        self._configuration['driver'] = driver


