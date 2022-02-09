#
# NNコンポーネントのテスト
#

from unittest import TestCase

import numpy as np
from src.component import Adder, Affine, Multiplexer, Relu, Sigmoid, SoftmaxWithLoss


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

    def testAffine(self):

        weight = np.random.randn(3, 2)
        bias = np.zeros((2, 2))

        affine = Affine(weight, bias)

        source = np.random.randn(2, 3)
        forward = affine.forward(source)

        dout = np.zeros_like(source) + 1
        # backward = affine.backward(dout)

    def testSoftmaxWithLoss(self):

        softmax_with_loss = SoftmaxWithLoss()

        with self.assertRaises(ValueError):
            softmax_with_loss.backward(np.zeros((2, 3)))

        batch_size, data_size = 2, 3
        source = np.random.randn(batch_size, data_size)
        teacher = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        forward = softmax_with_loss.forward(source, teacher)
