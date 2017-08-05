package cc5114.tests;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import cc5114.perceptron.ANDPerceptron;
import cc5114.perceptron.ITwoInputPerceptron;

public class ANDtest {
    private ITwoInputPerceptron andPerceptron;

    @Before
    public void setUp() throws Exception {
	andPerceptron = new ANDPerceptron();
    }

    @Test
    public void falseTest() {
	final int res = andPerceptron.evalTwo(0, 0);
	assertEquals("Should be 0", 0, res);
    }

    @Test
    public void trueTest() {
	final int res = andPerceptron.evalTwo(1, 1);
	assertEquals("Should be 1", 1, res);
    }

    @Test
    public void falseTest2() {
	final int res = andPerceptron.evalTwo(0, 1);
	assertEquals("Should be 0", 0, res);
    }

    @Test
    public void falseTest3() {
	final int res = andPerceptron.evalTwo(1, 0);
	assertEquals("Should be 0", 0, res);
    }

}
