package cc5114.tests;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import cc5114.perceptron.SummingBit;

public class SummingBitTest {
	private SummingBit summingBit;

	@Before
	public void setUp() throws Exception {
		summingBit = new SummingBit();
	}

	@Test
	public void summingTest() {
		final int[] res = summingBit.twoBitsSum(0, 0);
		assertEquals("Should be 0 bit", res[0], 0);
		assertEquals("Should be 0 carry", res[1], 0);
	}

	@Test
	public void summingTest2() {
		final int[] res = summingBit.twoBitsSum(0, 1);
		assertEquals("Should be 0 bit", res[0], 1);
		assertEquals("Should be 0 carry", res[1], 0);
	}

	@Test
	public void summingTest3() {
		final int[] res = summingBit.twoBitsSum(1, 1);
		assertEquals("Should be 0 bit", res[0], 0);
		assertEquals("Should be 0 carry", res[1], 1);
	}

}
