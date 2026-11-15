import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

import stablebear as sb  # noqa: E402
import stablebear._autodiff_tf  # noqa: E402,F401  (stablebear may have been imported before tensorflow)


def _tensors():
    F = sb.PcfTensor([
        sb.Pcf(np.array([[0.0, 1.0], [1.0, -2.0], [3.0, 0.5], [4.0, 0.0]])),
        sb.Pcf(np.array([[0.0, 3.0], [0.5, 1.0], [2.5, 0.0]])),
        sb.Pcf(np.array([[0.0, -1.0], [2.0, 2.0], [6.0, 0.0]])),
    ])
    G = sb.PcfTensor([sb.Pcf(np.array([[0.0, 1.0], [2.0, -0.5], [5.0, 0.0]]))])
    return F, G


def _assert_gradient_matches_numerical(f, x0):
    theoretical, numerical = tf.test.compute_gradient(f, [tf.constant(x0, dtype=tf.float64)])
    np.testing.assert_allclose(theoretical[0], numerical[0], rtol=1e-5, atol=1e-6)


def test_lp_norm_tf_p_matches_float_p():
    F, _ = _tensors()
    out = sb.lp_norm(F, p=tf.constant(2.5, dtype=tf.float64))
    assert isinstance(out, tf.Tensor)
    np.testing.assert_allclose(out.numpy(), np.asarray(sb.lp_norm(F, p=2.5)))


def test_lp_norm_gradient_wrt_p():
    F, _ = _tensors()
    weights = tf.constant([0.3, -1.0, 2.0], dtype=tf.float64)
    _assert_gradient_matches_numerical(lambda p: tf.reduce_sum(weights * sb.lp_norm(F, p=p)), 1.7)


@pytest.mark.parametrize("p", [1.0, 2.5])
@pytest.mark.parametrize("expr", [
    lambda F, G, a: F - a * G,
    lambda F, G, a: (G * a + F) / 2.0,
    lambda F, G, a: -(a * F[1]) + F[2],
])
def test_lp_norm_gradient_through_pcf_arithmetic(p, expr):
    F, G = _tensors()
    weights = tf.constant([0.3, -1.0, 2.0], dtype=tf.float64)

    def loss(a):
        n = sb.lp_norm(expr(F, G, a), p=p)
        return tf.reduce_sum(weights[: n.shape[0]] * n)

    _assert_gradient_matches_numerical(loss, 0.3)


def test_gradient_in_tf_function():
    F, G = _tensors()
    a = tf.Variable(0.3, dtype=tf.float64)

    def loss_and_grad():
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(sb.lp_norm(F - a * G, p=2.0))
        return tape.gradient(loss, a)

    np.testing.assert_allclose(tf.function(loss_and_grad)().numpy(), loss_and_grad().numpy())


def test_gradient_wrt_keras_variable():
    keras = pytest.importorskip("keras")
    if keras.backend.backend() != "tensorflow":
        pytest.skip("Keras runs on another backend")
    F, G = _tensors()
    v = keras.Variable(0.3, dtype="float64")
    a = tf.Variable(0.3, dtype=tf.float64)
    grads = []
    for param in (v, a):
        with tf.GradientTape() as tape:
            # A Keras variable on the left would convert G itself and fail.
            loss = tf.reduce_sum(sb.lp_norm(F - G * param, p=2.0))
        grads.append(tape.gradient(loss, param).numpy())
    np.testing.assert_allclose(grads[0], grads[1])
