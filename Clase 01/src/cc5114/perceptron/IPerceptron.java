package cc5114.perceptron;

import java.util.List;

public interface IPerceptron {
	int eval(List<Integer> inputs);
	int getBias();
}
