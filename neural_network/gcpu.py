import pkg_resources
import importlib

from neural_network.configuration import Driver, GlobalConfig

INSTALLED_PACKAGES = {pkg.key for pkg in pkg_resources.working_set}

def missing_driver(driver: Driver):
    return driver.get_module() not in INSTALLED_PACKAGES

def import_module():
    driver = GlobalConfig().get_driver()

    if driver is None:
        raise ValueError("No driver defined.")
    
    if missing_driver(driver):
        raise SystemError(f'{driver.name} driver: {driver.get_module()} is required')
    
    return importlib.import_module(driver.get_module(), package=None)

class DriverLoader:
    _module = None 

    @property
    def gcpu(self):
        if self._module is None:
            self._module = import_module()
        return self._module

driver = DriverLoader()

