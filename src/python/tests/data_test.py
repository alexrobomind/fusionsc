import fusionsc as fsc

import pytest

@pytest.fixture
def tmpar(tmp_path):
	return str(tmp_path / "test.fsc")

def test_archive(tmpar):
	fsc.data.writeArchive({}, tmpar)
	fsc.data.readArchive(tmpar)

def test_archive_ref(tmpar):
	fsc.data.writeArchive(fsc.data.publish({}), tmpar)
	fsc.data.readArchive(tmpar)

def test_upload():
	data = fsc.data.upload(fsc.data.publish({"Hello", ("AB",)}))
	fsc.data.download(data)