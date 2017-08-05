package cc5114.perceptron;

import java.util.Arrays;
import java.util.List;

public class SummingBit {

	public int[] twoBitsSum(final int bit1, final int bit2) {

		final List<Integer> tempInput = Arrays.asList(bit1, bit2);
		final IPerceptron nandNeuron = new NANDPerceptron();

		final int outL1 = nandNeuron.eval(tempInput);
		final int outL21 = nandNeuron.eval(Arrays.asList(bit1, outL1));
		final int outL22 = nandNeuron.eval(Arrays.asList(outL1, bit2));

		final int out1 = nandNeuron.eval(Arrays.asList(outL21, outL22));
		final int out2 = nandNeuron.eval(Arrays.asList(Integer.valueOf(outL1), Integer.valueOf(outL1)));

		return new int[]{out1, out2};

	}


}
