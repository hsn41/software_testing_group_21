import numpy.testing as np
from numpy.core import(
     intp, float32, empty, arange, array_repr, ndarray, isnat, array)
import unittest
class TestClass(unittest.TestCase):
    """ Functions for testing. """
    def setUp(self):
        self.value_a = 2.3333333333333
        self.value_b = 2.3333333333333
        self.sample_array = [1.0,2.3333333333333]
        self.x = [1e-5, 1e-3, 1e-1]
        self.y = np.arccos(np.cos(self.x))


    # to check given value is nearly equal to desired result or not
    def test_value_a(self):
        np.assert_almost_equal(self.value_a, 2.33333334)

    # to check given value upto to specified decimal
    def test_value_b(self):
        np.assert_almost_equal(self.value_b, 2.33333334,decimal=8)

    #to check simple array
    def test_sample_array(self):
        np.assert_almost_equal(self.sample_array,[1.0,2.3333334])

    def test_simple(self):
        np.assert_string_equal("hello", "hello")
        np.assert_string_equal("hello\nmultiline", "hello\nmultiline")

   # i dont know why i cant use array,can you please guide what are the mistake i am making
    def test_equal_nan(self):
        a = np.array([np.nan])
        b = np.array([np.nan])
        # Should not raise:
        np.assert_allclose(a, b, equal_nan=True)

   # def test_cos(self):
   #     np.testing.assert_allclose(self.x, self.y, rtol=1e-5, atol=0)


if __name__ == "__main__":
    unittest.main()

