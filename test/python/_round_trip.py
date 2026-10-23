"""Save and load an item through each of stablebear's binary formats."""

import io
import pickle

import pytest

import stablebear as sb


def _save_and_load(item):
    buffer = io.BytesIO()
    sb.save(item, buffer)
    buffer.seek(0)
    return sb.load(buffer)


def _pickle_and_unpickle(item):
    return pickle.loads(pickle.dumps(item))


ROUND_TRIPS = [
    pytest.param(_save_and_load, id="save"),
    pytest.param(_pickle_and_unpickle, id="pickle"),
]
