package cc5114.tests;

import cc5114.perceptron.Perceptron;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class PerceptronTest {
    private Perceptron perceptron;
    private List<Integer> input6;
    private int bias;

    @Before
    public void setUp() throws Exception {
        bias = 10;
        perceptron = new Perceptron(Arrays.asList(1, 1, 1), bias);
        input6 = Arrays.asList(0, 0, 0, 0, 0, 0);
    }

    @Test
    public void exceptionTest() {
        boolean check = false;
        try {
            perceptron.eval(input6);
        } catch (final IllegalArgumentException e) {
            check = true;
        }
        assertTrue("Should throw and exception", check);
    }

    @Test
    public void getBiasTest() {

        assertEquals("Should be the same", bias, perceptron.getBias());
    }

}
