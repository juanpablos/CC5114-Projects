package perceptron;

import java.util.List;

public class Perceptron implements IPerceptron {
	private final List<Integer> weights;
	private final int bias;

	protected Perceptron(final List<Integer> weights, final int bias) {
		this.weights = weights;
		this.bias = bias;
	}

	@Override
	public boolean eval(final List<Integer> inputs) {
		int sum = 0;
		for(int i=0; i < weights.size() ; i++) {
			sum += inputs.get(i)*weights.get(i);
		}
		return sum + bias>= 0;
	}

}
