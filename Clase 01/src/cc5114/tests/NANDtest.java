package cc5114.tests;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import cc5114.perceptron.ITwoInputPerceptron;
import cc5114.perceptron.NANDPerceptron;

public class NANDtest {
    private ITwoInputPerceptron nandPerceptron;

    @Before
    public void setUp() throws Exception {
	nandPerceptron = new NANDPerceptron();
    }

    @Test
    public void falseTest() {
	final int res = nandPerceptron.evalTwo(1, 1);
	assertEquals("Should be 0", 0, res);
    }

    @Test
    public void trueTest() {
	final int res = nandPerceptron.evalTwo(0, 0);
	assertEquals("Should be 1", 1, res);
    }

    @Test
    public void trueTest2() {
	final int res = nandPerceptron.evalTwo(0, 1);
	assertEquals("Should be 1", 1, res);
    }

    @Test
    public void trueTest3() {
	final int res = nandPerceptron.evalTwo(1, 0);
	assertEquals("Should be 1", 1, res);
    }

}
