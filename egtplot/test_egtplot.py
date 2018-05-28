##########
# Automated unit testing
##########
from code import *
import unittest

class TestMainFunctions(unittest.TestCase):
    """
    Tests the main functions for the proper functioning of egtplot package.
    """
    def test_random_ics(self):
        """
        Test initial conditions sum to one.
        """
        ics = random_ics(10)
        self.assertTrue( np.all( np.logical_and( ics.sum(axis=1), np.ones(len(ics)) )))

    def test_random_uniform_points(self):
        """
        Test initial conditions sum to one.
        """
        ics = random_uniform_points(100, 0.01)
        self.assertTrue( np.all( np.logical_and( ics.sum(axis=1), np.ones(len(ics)) )))

    def test_grid_ics(self):
        """
        Test initial conditions sum to one.
        """
        ics = grid_ics()
        self.assertTrue( np.all( np.logical_and( ics.sum(axis=1), np.ones(len(ics)) )))


    def test_load_bomze_payoffs(self):
        """
        Test that load_bomze_payoffs is not returning None.
        """
        payoffs =  load_bomze_payoffs()

        self.assertIsInstance( payoffs, list  )
        self.assertEqual( len(payoffs), 49  )


    def test_landsace(self):
        """
        Test if landscape returns (3,) numpy array.
        """

        payoff = np.array([ [1,2,3],
                            [4,5,6],
                            [7,8,9] ] )
        dy = landscape( [1,1,1], 1, payoff)
        self.assertIsInstance(dy, type(np.ones(1)) )
        self.assertEqual( dy.shape, (3,) )

    def test_plot_static(self):
        """
        Test if plot_static returning figures.
        """

        payoff = [ [1],[2],[3], [4],[5],[6], [7],[7],[7] ]

        self.assertTrue( isinstance( plot_static( payoff ), type(plt.figure()) )  )

    def test_plot_animated(self):
        """
        Test if plot_animated returning animations.
        """
        from moviepy.video.VideoClip import VideoClip

        payoff = [ [1],[2],[3], [4],[5],[6], [7],[7],[7] ]
        output = plot_animated( payoff )
        self.assertIsInstance( output, tuple )
        self.assertIsInstance(  output[0], VideoClip  )


if __name__ == '__main__':
    unittest.main()
