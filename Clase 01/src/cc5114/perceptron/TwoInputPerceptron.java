package cc5114.perceptron;

import java.util.Arrays;

public class TwoInputPerceptron extends Perceptron implements ITwoInputPerceptron {

    public TwoInputPerceptron(final int weight1, final int weight2, final int bias) {
	super(Arrays.asList(weight1, weight2), bias);
    }

    public int evalTwo(final int input1, final int input2) {
	return super.eval(Arrays.asList(input1, input2));
    }

}
