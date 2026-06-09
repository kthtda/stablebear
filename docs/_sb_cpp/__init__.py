# Stub _sb_cpp for documentation builds.
# Replaces the real backend selector so that autodoc can import stablebear
# without the compiled C++ extension.


class _Mock:
    def __init__(self, *a, **kw):
        pass

    def __call__(self, *a, **kw):
        return _Mock()

    def __getattr__(self, name):
        return _Mock()

    def __repr__(self):
        return "_Mock()"

    def __bool__(self):
        return False

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __getitem__(self, key):
        return _Mock()

    def __setitem__(self, key, value):
        pass

    def __delitem__(self, key):
        pass

    def __contains__(self, item):
        return False

    def __add__(self, other):
        return _Mock()

    def __radd__(self, other):
        return _Mock()

    def __sub__(self, other):
        return _Mock()

    def __rsub__(self, other):
        return _Mock()

    def __mul__(self, other):
        return _Mock()

    def __rmul__(self, other):
        return _Mock()

    def __truediv__(self, other):
        return _Mock()

    def __rtruediv__(self, other):
        return _Mock()

    def __floordiv__(self, other):
        return _Mock()

    def __rfloordiv__(self, other):
        return _Mock()

    def __mod__(self, other):
        return _Mock()

    def __rmod__(self, other):
        return _Mock()

    def __pow__(self, other):
        return _Mock()

    def __rpow__(self, other):
        return _Mock()

    def __neg__(self):
        return _Mock()

    def __pos__(self):
        return _Mock()

    def __abs__(self):
        return _Mock()

    def __invert__(self):
        return _Mock()

    def __and__(self, other):
        return _Mock()

    def __rand__(self, other):
        return _Mock()

    def __or__(self, other):
        return _Mock()

    def __ror__(self, other):
        return _Mock()

    def __xor__(self, other):
        return _Mock()

    def __rxor__(self, other):
        return _Mock()

    def __lshift__(self, other):
        return _Mock()

    def __rlshift__(self, other):
        return _Mock()

    def __rshift__(self, other):
        return _Mock()

    def __rrshift__(self, other):
        return _Mock()

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __hash__(self):
        return id(self)

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    def __str__(self):
        return ""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def __getattr__(name):
    return _Mock()
