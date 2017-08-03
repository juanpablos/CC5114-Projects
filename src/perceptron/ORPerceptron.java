package perceptron;

import java.util.Arrays;
import java.util.List;

public class ORPerceptron extends Perceptron {
	final static List<Integer> input = Arrays.asList(2,2);

	public ORPerceptron() {
		super(input, -1);
	}

}
