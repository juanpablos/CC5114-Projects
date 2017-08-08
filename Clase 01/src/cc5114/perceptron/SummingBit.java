package cc5114.perceptron;

public class SummingBit {

    public int[] twoBitsSum(final int bit1, final int bit2) {

        final ITwoInputPerceptron nandNeuron = new NANDPerceptron();

        final int outL1 = nandNeuron.evalTwo(bit1, bit2);
        final int outL21 = nandNeuron.evalTwo(bit1, outL1);
        final int outL22 = nandNeuron.evalTwo(outL1, bit2);

        final int out1 = nandNeuron.evalTwo(outL21, outL22);
        final int out2 = nandNeuron.evalTwo(outL1, outL1);

        return new int[]{out1, out2};

    }

}
