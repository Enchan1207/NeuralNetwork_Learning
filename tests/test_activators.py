#
# Activatorのテスト
#

from unittest import TestCase

import numpy as np
from numpy import ndarray
from src.activator import Relu, Sigmoid, Softmax


class testActivator(TestCase):

    def testRelu(self):
        """ReLU関数のテスト
        """

        relu = Relu()
        source = np.random.randn(10, 20)
        source[0][0] = 0.0

        # forward
        forward = relu.forward(source)
        for s, f in zip(source.flatten(), forward.flatten()):
            if s > 0:
                self.assertEqual(s, f)
            else:
                self.assertEqual(f, 0)

        # backward
        dout = np.random.randn(*source.shape)
        backward = relu.backward(dout)

        # forwardで0になったところはbackwardを通してもゼロになり…
        forward_zero_mask = forward <= 0
        backward_zero_mask = backward == 0
        self.assertTrue(np.array_equal(forward_zero_mask, backward_zero_mask))

        # forwardで通過したところはbackwardを通したときにdoutが反映されていることを確認する
        forward_nonzero_mask = forward > 0
        self.assertTrue(np.array_equal(backward[forward_nonzero_mask], dout[forward_nonzero_mask]))

    def testSigmoid(self):
        """Sigmoid関数のテスト
        """

        sigmoid = Sigmoid()

        source = np.random.randn(10, 20)
        source[0][0] = 0.0

        # forward
        forward = sigmoid.forward(source)

        # forwardが0~1を超えないことを確認
        self.assertFalse(True in forward < 0)
        self.assertFalse(True in forward > 1.0)

        # backward
        dout = np.random.randn(*source.shape)
        backward = sigmoid.backward(dout)
        self.assertTrue(np.array_equal(dout * forward * (1 - forward), backward))

    def testSoftmax(self):
        """Softmax関数のテスト
        """

        softmax = Softmax()

        source = np.random.randn(10, 20) * 10.0
        source[0][0] = 0.0

        # forward
        forward = softmax.forward(source)

        # forwardが0~1を超えないことを確認
        self.assertFalse(True in forward < 0)
        self.assertFalse(True in forward > 1.0)

        # backward
        dout = np.random.randn(*source.shape)
        backward = softmax.backward(dout)
        backward_emu = forward * (dout - np.sum(forward * dout, axis=-1, keepdims=True))
        self.assertTrue(np.array_equal(backward, backward_emu))
