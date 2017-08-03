package tests;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import perceptron.ANDPerceptron;
import perceptron.Perceptron;

public class ANDtest {
	private Perceptron andPerceptron;

	@Before
	public void setUp() throws Exception {
		andPerceptron = new ANDPerceptron();
	}

	@Test
	public void falseTest() {
		final List<Integer> input = Arrays.asList(0,0);
		final boolean res = andPerceptron.eval(input);
		assertFalse("Should be false",res);
	}

	@Test
	public void trueTest() {
		final List<Integer> input = Arrays.asList(1,1);
		final boolean res = andPerceptron.eval(input);
		assertTrue("Should be true",res);
	}

	@Test
	public void falseTest2() {
		final List<Integer> input = Arrays.asList(0,1);
		final boolean res = andPerceptron.eval(input);
		assertFalse("Should be false",res);
	}

	@Test
	public void falseTest3() {
		final List<Integer> input = Arrays.asList(1,0);
		final boolean res = andPerceptron.eval(input);
		assertFalse("Should be false",res);
	}

}
