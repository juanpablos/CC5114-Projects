package tests;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import perceptron.NANDPerceptron;
import perceptron.Perceptron;

public class NANDtest {
	private Perceptron nandPerceptron;

	@Before
	public void setUp() throws Exception {
		nandPerceptron = new NANDPerceptron();
	}

	@Test
	public void falseTest() {
		final List<Integer> input = Arrays.asList(1,1);
		final boolean res = nandPerceptron.eval(input);
		assertFalse("Should be false",res);
	}

	@Test
	public void trueTest() {
		final List<Integer> input = Arrays.asList(0,0);
		final boolean res = nandPerceptron.eval(input);
		assertTrue("Should be true",res);
	}

	@Test
	public void trueTest2() {
		final List<Integer> input = Arrays.asList(0,1);
		final boolean res = nandPerceptron.eval(input);
		assertTrue("Should be true",res);
	}

	@Test
	public void trueTest3() {
		final List<Integer> input = Arrays.asList(1,0);
		final boolean res = nandPerceptron.eval(input);
		assertTrue("Should be true",res);
	}

}
