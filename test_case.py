from __future__ import division, absolute_import, print_function
import unittest
import numpy as np
from numpy import array, dot
from numpy.linalg import matrix_power, norm
from numpy import linalg as LinAlgError
from numpy.testing import assert_equal


class linAlgebraTesting(unittest.TestCase):

    # we have to use these matrix for further test cases,you can use a,b,c and perform functions
    def setUp(self):
        self.value_a = 2.3333333333333
        self.value_b = 2.3333333333333
        self.num1 = 2
        self.num2 = 3
        self.given_age = 19
        self.empty_array = array([[]])
        self.two_D_array = [[1.0, 2.3333333333333,4.5],[2.3,2.5,2.02]]
        self.sample_array = [1.0, 2.3333333333333]
        self.ar = array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.ar1 = array([0, 4, 8])

        self.a = array([[2, 2, 1], [5, 3, 4], [2, 3, 6]])
        self.b = array([[4, 2, 3], [6, 1, 7], [2, 4, 6]])
        self.c = array([[6, 3, 5], [7, 2, 9], [1, 3, 5]])

        # our test cases should be equal to these cases
        self.dot_of_ab = array([[22, 10, 26], [46, 29, 60], [38, 31, 63]])
        self.power_of_a = array([[16, 13, 16], [33, 31, 41], [31, 31, 50]])
        self.power_of_a_3 = array([[129,119,164],[303,282,403],[317,305,455]])
        self.inverse_of_a = array([[-6 / 23, 9 / 23, -5 / 23], [22 / 23, -10 / 23, 3 / 23], [-9 / 23, 2 / 23, 4 / 23]])
        self.multi_dot_equal = array([[228, 164, 330], [539, 376, 791], [508, 365, 784]])

        self.x = [1e-5, 1e-3, 1e-1]
        self.y = np.arccos(np.cos(self.x))

        # to check given value is nearly equal to desired result or not

    def test_value_a(self):
        np.testing.assert_almost_equal(self.value_a, 2.33333334)

    def test_empty_matrix(self):
        np.testing.assert_array_almost_equal(norm(self.empty_array), 0.0)

    # test upto some decimals
    def test_value_b(self):
        np.testing.assert_almost_equal(self.value_b, 2.33333334, decimal=8)

    # addition of integers
    def test_sum_value(self):
        np.testing.assert_equal(self.num1 + self.num2,5)

    # to check simple array 1D array
    def test_sample_array(self):
        np.testing.assert_almost_equal(self.sample_array, [1.0, 2.3333334])

    # to check 2D array
    def test_2D_array(self):
        np.testing.assert_array_almost_equal(self.two_D_array, [[1.0, 2.3333333333333,4.5],[2.3,2.5,2.02]])

    # test check 3D array
    def test_3D_array(self):
        np.testing.assert_array_almost_equal(self.a,[[2, 2, 1], [5, 3, 4], [2, 3, 6]])

    # to check two strings are equal or not
    def test_simple(self):
        np.testing.assert_string_equal("hello", "hello")
        np.testing.assert_string_equal("hello\nmultiline", "hello\nmultiline")

    # dot product of matrix
    def test_two_array(self):
        np.testing.assert_equal(dot(self.a, self.b), self.dot_of_ab)

    # multiple dot of  3 matrix
    def test_multidot_matrix(self):
        np.testing.assert_equal(dot(dot(self.a, self.b), self.c), self.multi_dot_equal)

    # inverse of matrix
    def test_inverse_matrix(self):
        np.testing.assert_almost_equal(LinAlgError.inv(self.a), self.inverse_of_a)

    # matrix power for power 2
    def test_matrix_power(self):
        np.testing.assert_equal(matrix_power(self.a, 2), self.power_of_a)

    # matrix power for power 3
    def test_matrix_power(self):
        np.testing.assert_array_almost_equal(matrix_power(self.a, 3), self.power_of_a_3)

    def test_constract_array(self):
        np.testing.assert_array_equal(np.arange(9).reshape(3, 3), self.ar)

    def test_a(self):
        np.testing.assert_array_equal(np.diag(np.arange(9).reshape((3, 3))), self.ar1)

    # even number find it can be consider as white box,because we are covering branches here
    def test_even_integers(self):
        self.even_odd_test_array = [2,4,6,8,10]
        for i in range(0, len(self.even_odd_test_array)):
            if self.assertEqual(self.even_odd_test_array[i] % 2, 0):
                print(self.even_odd_test_array[i])
            else:
                print(self.even_odd_test_array[i])

    def test_odd_integers(self):
        self.even_odd_test_array = [1,3,5,7,9]
        for i in range(0, len(self.even_odd_test_array)):
            if self.assertEqual(self.even_odd_test_array[i] % 2, 1):
                print(self.even_odd_test_array[i])
            else:
                print(self.even_odd_test_array[i])

    def test_containing_list(self):
        # printing square brackets directly would be ambiguuous
        arr1d = np.array([None, None])
        arr1d[0] = [1, 2]
        arr1d[1] = [3]
        assert_equal(repr(arr1d),
                     'array([list([1, 2]), list([3])], dtype=object)')

    def test_fieldless_structured(self):
        # gh-10366
        no_fields = np.dtype([])
        arr_no_fields = np.empty(4, dtype=no_fields)
        assert_equal(repr(arr_no_fields), 'array([(), (), (), ()], dtype=[])')

    # high power of matrix
    # def test_matrix_complex_power(self):
    #     np.testing.assert_array_equal()

    # Member 2 can work on datetime testing
    # Member 3 can work on mulitiarray dimensions
    # Member 4 can work on array prints
    # Member 5 can work on numerics


if __name__ == "__main__":
    unittest.main()
