from __future__ import division, absolute_import, print_function
import unittest
import numpy as np
from numpy import array,dot
from numpy.linalg import matrix_power
from numpy import linalg as LinAlgError

class linAlgebraTesting(unittest.TestCase):

    # we have to use these matrix for further test cases,you can use a,b,c and perform functions
    def setUp(self):

        self.value_a = 2.3333333333333
        self.value_b = 2.3333333333333
        self.sample_array = [1.0, 2.3333333333333]

        self.a = array([[2,2,1],[5,3,4],[2,3,6]])
        self.b = array([[4,2,3],[6,1,7],[2,4,6]])
        self.c = array([[6,3,5],[7,2,9],[1,3,5]])

        # our test cases should be equal to these cases
        self.dot_of_ab = array([[22, 10, 26], [46, 29, 60], [38, 31, 63]])
        self.power_of_a = array([[16, 13, 16], [33, 31, 41], [31, 31, 50]])
        self.inverse_of_a = array([[-6/23,9/23,-5/23],[22/23,-10/23,3/23],[-9/23,2/23,4/23]])
        self.multi_dot_equal = array([[228,164,330],[539,376,791],[508,365,784]])

        self.x = [1e-5, 1e-3, 1e-1]
        self.y = np.arccos(np.cos(self.x))

        # to check given value is nearly equal to desired result or not

    def test_value_a(self):
        np.testing.assert_almost_equal(self.value_a, 2.33333334)

        # to check given value upto to specified decimal

    def test_value_b(self):
        np.testing.assert_almost_equal(self.value_b, 2.33333334, decimal=8)

        # to check simple array

    def test_sample_array(self):
        np.testing.assert_almost_equal(self.sample_array, [1.0, 2.3333334])

    def test_simple(self):
        np.testing.assert_string_equal("hello", "hello")
        np.testing.assert_string_equal("hello\nmultiline", "hello\nmultiline")

     # dot product of matrix
    def test_two_array(self):
        np.testing.assert_equal(dot(self.a,self.b),self.dot_of_ab)

    # multiple dot of matrix
    def test_multidot_matrix(self):
        np.testing.assert_equal(dot(dot(self.a,self.b),self.c),self.multi_dot_equal)
    # inverse of matrix
    def test_inverse_matrix(self):
        np.testing.assert_almost_equal(LinAlgError.inv(self.a),self.inverse_of_a)

    # matrix power for power 2
    def test_matrix_power(self):
        np.testing.assert_equal(matrix_power(self.a,2),self.power_of_a)

    # high power of matrix
    # def test_matrix_complex_power(self):
    #     np.testing.assert_array_equal()

if __name__ == "__main__":
    unittest.main()
