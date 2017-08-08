package cc5114.tests;

import cc5114.perceptron.ITwoInputPerceptron;
import cc5114.perceptron.ORPerceptron;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class ORtest {
    private ITwoInputPerceptron orPerceptron;

    @Before
    public void setUp() throws Exception {
        orPerceptron = new ORPerceptron();
    }

    @Test
    public void falseTest() {
        final int res = orPerceptron.evalTwo(0, 0);
        assertEquals("Should be 0", 0, res);
    }

    @Test
    public void trueTest() {
        final int res = orPerceptron.evalTwo(1, 1);
        assertEquals("Should be 1", 1, res);
    }

    @Test
    public void trueTest2() {
        final int res = orPerceptron.evalTwo(0, 1);
        assertEquals("Should be 1", 1, res);
    }

    @Test
    public void trueTest3() {
        final int res = orPerceptron.evalTwo(1, 0);
        assertEquals("Should be 1", 1, res);
    }

}
