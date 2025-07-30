import pytest
import fusionsc as fsc

@pytest.fixture(scope="session", autouse = True)
def useLocalBackend():
	fsc.backends.alwaysUseBackend(fsc.backends.localBackend())