package perceptron;

import java.util.List;

public class Perceptron implements IPerceptron {
	private final List<Integer> weights;
	private final int bias;

	public Perceptron(final List<Integer> weights, final int bias) {
		this.weights = weights;
		this.bias = bias;
	}

	@Override
	public int eval(final List<Integer> inputs) {
		if (weights.size() != inputs.size()) {
			throw new IllegalArgumentException("Input number arguments don't match.");
		}
		int sum = 0;
		for(int i = 0; i < weights.size() ; i++) {
			sum += inputs.get(i) * weights.get(i);
		}
		return sum + bias >= 0 ? 1 : 0;
	}

	@Override
	public int getBias() {
		return bias;
	}

}
