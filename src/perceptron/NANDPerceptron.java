package perceptron;

import java.util.Arrays;
import java.util.List;

public class NANDPerceptron extends Perceptron {
	final static List<Integer> input = Arrays.asList(-2,-2);

	public NANDPerceptron() {
		super(input, 3);
	}

}
