package cc5114;

import cc5114.perceptron.SummingBit;

import java.util.Scanner;

public class Main {

    public static void main(final String[] args) {

        final SummingBit sumGate = new SummingBit();
        final Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("Enter a pair of bits (0 or 1), enter -1 to end: ");
            final int bit1 = scanner.nextInt();
            if (bit1 == -1) {
                break;
            }
            final int bit2 = scanner.nextInt();
            final int[] res = sumGate.twoBitsSum(bit1, bit2);
            System.out.println("sum: " + res[0] + "\ncarry bit: " + res[1]);
            System.out.println();
        }
        System.out.println("End");
        scanner.close();

    }
}
