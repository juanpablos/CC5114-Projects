package cc5114.perceptron;

import java.util.Arrays;
import java.util.List;

public class ANDPerceptron extends Perceptron {
	final static List<Integer> input = Arrays.asList(2, 2);

	public ANDPerceptron() {
		super(input, -3);
	}

}
