#
# NNコンポーネントのテスト
#

from unittest import TestCase

import numpy as np
from src.component import Adder, Multiplexer, Sigmoid, Relu


class TestNNComponents(TestCase):

    def testAdder(self):
        adder = Adder()

        source_1 = np.random.randn(3, 2)
        source_2 = np.random.randn(3, 2)
        forward = adder.forward(source_1, source_2)
        self.assertTrue(np.array_equal(source_1 + source_2, forward))

        source_3 = np.random.randn(3, 2)
        backward_1, backward_2 = adder.backward(source_3)
        self.assertTrue(np.array_equal(backward_1, backward_2))
        self.assertTrue(np.array_equal(source_3, backward_1))

    def testMultiplexer(self):
        multiplexer = Multiplexer()

        with self.assertRaises(ValueError):
            multiplexer.backward(np.zeros((2, 3)))

        source_1 = np.random.randn(3, 2)
        source_2 = np.random.randn(3, 2)
        forward = multiplexer.forward(source_1, source_2)
        self.assertTrue(np.array_equal(source_1 * source_2, forward))

        source_3 = np.random.randn(3, 2)
        backward_1, backward_2 = multiplexer.backward(source_3)
        self.assertTrue(np.array_equal(source_1 * source_3, backward_2))
        self.assertTrue(np.array_equal(source_2 * source_3, backward_1))

    def testSigmoid(self):
        # もうちょっとテストケースらしいことしたかった
        sigmoid = Sigmoid()

        with self.assertRaises(ValueError):
            sigmoid.backward(np.zeros((2, 3)))

        source = np.random.randn(3, 2)
        _ = sigmoid.forward(source)

        dout = np.zeros_like(source) + 1
        _ = sigmoid.backward(dout)

    def testReLU(self):
        relu = Relu()

        source = np.random.randn(3, 2)
        forward = relu.forward(source)
        self.assertTrue(np.array_equal(source > 0, forward == 1))

        dout = np.zeros_like(source) + 1
        backward = relu.backward(dout)
        self.assertTrue(np.array_equal(source > 0, backward == 1))
