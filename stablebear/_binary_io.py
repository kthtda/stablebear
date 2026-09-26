from abc import ABC, abstractmethod


class _BinaryIoMixin(ABC):
    """Provide the shared backend-value and binary-pickle protocol."""

    @abstractmethod
    def _binary_io_data(self):
        """Return the C++ value consumed by the binary object writer."""
        raise NotImplementedError

    def __reduce__(self):
        # Lazy import avoids cycles while stablebear.io imports the public
        # tensor and standalone object classes that inherit this mixin.
        from .io import _pickle_reduce

        return _pickle_reduce(self)
