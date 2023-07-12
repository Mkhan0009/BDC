import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class G017HW2 {

    // Global variables
    public static OutputModel outputModel = new OutputModel();

    public static void main(String[] args) {

        //Checking number of provided command-line (CLI) arguments
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path num_of_centers number_of_outliers");
        }

        // Receives in input the following command-line (CLI) arguments
        // A path to a text file containing point set in Euclidean space
        String path = args[0];

        // An integer k (the number of centers)
        int k = 0;

        //Checking if the provided number of centers is a positive integer
        try {
            k = Integer.parseInt(args[1]);
            if (k <= 0) {
                System.out.println("Number of centers must be greater then zero");
                System.exit(0);
            }
        } catch (NumberFormatException e) {
            System.out.println("Argument " + args[1] + " must be an integer.");
            System.exit(0);
        }

        // An integer z (the number of allowed outliers)
        int z = 0;

        //Checking if the provided number of outliers is a positive integer
        try {
            z = Integer.parseInt(args[2]);
            if (z < 0) {
                System.out.println("Number of outliers must be greater then zero");
                System.exit(0);
            }
        } catch (NumberFormatException e) {
            System.out.println("Argument " + args[2] + " must be an integer.");
            System.exit(0);
        }

        // Read the points in the input file into an ArrayList<Vector> called inputPoints
        ArrayList<Vector> inputPoints = new ArrayList<>();
        try {
            inputPoints = readVectorsSeq(path);
        } catch (IOException e) {
            System.out.println("Provided path is incomplete/wrong");
            System.exit(0);
        }

        // Create an ArrayList<Long> called weights of the same cardinality of inputPoints
        ArrayList<Long> weights = new ArrayList<>();

        // initialize with all 1's
        for (int i = 0; i < inputPoints.size(); i++) {
            weights.add(1L);
        }

        // Run SeqWeightedOutliers(inputPoints,weights,k,z,0) to compute a set of (at most) k centers
        // The output of the method must be saved into an ArrayList<Vector> called solution
        long time_start = System.currentTimeMillis();
        ArrayList<Vector> solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0);
        long time_finish = System.currentTimeMillis();

        // time required by the execution of SeqWeightedOutliers(inputPoints,weights,k,z,0)
        long time_needed = time_finish - time_start;

        // Run ComputeObjective(inputPoints,solution,z) and save the output in a variable called objective
        double objective = ComputeObjective(inputPoints, solution, z);

        // Print the output
        System.out.println(outputMethod(inputPoints.size(), k, z, objective, time_needed));

    }

    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z, double alpha) {

        // Incorporating probable edge cases
        // The size of P can't be empty, nor the number of centers can be more than size of P, and when number of outliers is equal or greater to size of P,
        // we can't decide the centers
        if ((k > P.size()) || (P.size() == 0) || (P.size() <= z)) {
            outputModel.setNumber_of_guesses(0);
            return new ArrayList<>();
        }

        // The number of centers is equal to size of P
        if (P.size() == k) {
            outputModel.setNumber_of_guesses(0);
            return P;
        }

        // r_min = (min distance between first k + z + 1 points)/2
        double r_min = Math.sqrt(Vectors.sqdist(P.get(0), P.get(1))); // R_min initialization

        // computing all distance of the first ( k + z + 1) / 2 points
        for (int i = 0; i < (k + z + 1); i++) {
            // don't need to calculate distance between same point
            for (int j = i + 1; j < (k + z + 1); j++) {

                // Given two points x and y, instances of Vector, their Euclidean L2-distance can be computed by invoking: Math.sqrt(Vectors.sqdist(x, y))
                double distance = Math.sqrt(Vectors.sqdist(P.get(i), P.get(j)));

                // check if it's the min distance
                if (distance < r_min) {
                    // update r_min value
                    r_min = distance;
                }
            }
        }

        r_min = r_min / 2;
        outputModel.setInitial_guess_r(r_min); // needed for output

        // dictionary pair with (Vector, Weight)
        HashMap<Vector, Long> P_pairs = new HashMap<>();

        // Creation of P_pairs with key the point x in P, and the value its weight
        int t = 0;
        for (Vector x : P) {
            P_pairs.put(x, W.get(t));
            t++;
        }

        while (true) {

            // Z = P;
            ArrayList<Vector> Z = new ArrayList<>(P);

            // S = ∅;
            HashSet<Vector> S = new HashSet<>();

            // formula of the inner radius
            double inner_radius = (1 + 2 * alpha) * r_min;

            // formula of the outer radius
            double outer_radius = (3 + 4 * alpha) * r_min;

            ArrayList<Vector> B_Z;

            // W_z = SUM_x∈P w(x); total weight
            long W_z = 0;
            for (Vector x : P) {
                W_z += P_pairs.get(x);
            }

            Vector new_center = null;

            while ((S.size() < k) && (W_z > 0)) {

                long max = 0;

                for (Vector x : P) {
                    // For a radius r > 0, a set Z ⊂ P, and a point x, define the ball of Z with radius r centered at x as B_z (x,r) = {y ∈ Z : d(x, y) ≤ r}.

                    long ball_weight = 0; // variable containing the sum of weights inside the ball

                    // calculating the weight of the points inside inner_radius
                    for (Vector y : Z) {

                        // we need to calculate the distance of y from the center x
                        double distance = Math.sqrt(Vectors.sqdist(x, y));

                        // if inside the inner ball, add weight
                        if (distance <= inner_radius) {
                            ball_weight += P_pairs.get(y);
                        }
                    }

                    if (ball_weight > max) {
                        max = ball_weight;

                        // this point has the max weight
                        new_center = x;
                    }
                }

                // add to S the center that has the max weight in the previous ball
                S.add(new_center);

                // Ball Z of center new_center and radius outer_radius
                B_Z = new ArrayList<>();

                // creation of B_Z
                for (Vector v : Z) {

                    // we need to calculate the distance of y from the new_center
                    double distance = Math.sqrt(Vectors.sqdist(new_center, v));

                    // if inside the outer ball, add to B_Z
                    if (distance <= outer_radius) {
                        B_Z.add(v);
                    }
                }

                for (Vector y : B_Z) {
                    // remove y from Z
                    Z.remove(y);

                    // substract w(y) from W_z
                    W_z -= P_pairs.get(y);
                }
            }

            if (W_z <= z) {
                // found required centers, therefore updating final radius guess and returning
                // the list of centers.
                outputModel.setFinal_guess_r(r_min); // needed for output
                return new ArrayList<>(S);
            } else {
                // incrementing number of guesses and radius as we haven't found desired number
                // of centers
                r_min = 2 * r_min;
                outputModel.increment_Number_of_Guesses(); // needed for output
            }
        }
    }

    public static double ComputeObjective(ArrayList<Vector> P, ArrayList<Vector> S, int z) {

        //Incorporating probable edge cases

        //  if dataset size equals to the size of centers or if list of centers is empty then the maximum
        //  distance of any point from their respective center must be zero hence the objective function
        //  is returning 0
        if ((P.size() == S.size()) || S.isEmpty()) {
            return 0;
        }

        double max_distance = 0;
        HashSet<Double> distances = new HashSet<>(); // Container of all distances

        // compute all distances d(x,S), for every x in P
        // find minimum distance from all data points (x) from respective center (s)
        for (Vector x : P) {
            double minDistance = Math.sqrt(Vectors.sqdist(x, S.get(0)));
            for (Vector s : S) {
                double curr_distance = Math.sqrt(Vectors.sqdist(x, s));
                if (curr_distance < minDistance) {
                    minDistance = curr_distance;
                }
            }
            distances.add(minDistance);
        }

        // sort distances in ascending order
        ArrayList<Double> distanceList = new ArrayList<>(distances);
        Collections.sort(distanceList);

        // exclude the z-largest distances which are the distances from outliers
        // return the largest among the remaining ones
        max_distance = distanceList.get(distanceList.size() - z - 1);
        return max_distance;
    }

    /**
     * Return as output the following quantities: n =|P|,
     * k,
     * z,
     * the initial value of the guess r,
     * the final value of the guess r,
     * and the number of guesses made by SeqWeightedOutliers(inputPoints,weights,k,z,0),
     * the value of the objective function (variable objective),
     * and the time (in milliseconds) required by the execution of SeqWeightedOutliers(inputPoints,weights,k,z,0).
     */

    private static String outputMethod(int n, int k, int z, double objective, long time_needed) {
        // output string
        String res = "";

        res = res + "Input size n = " + n;

        res = res + "\nNumber of centers k = " + k;

        res = res + "\nNumber of outliers z = " + z;

        res = res + "\nInitial guess = " + outputModel.getInitial_guess_r();

        res = res + "\nFinal guess = " + outputModel.getFinal_guess_r();

        res = res + "\nNumber of guesses = " + outputModel.getNumber_of_guesses();

        res = res + "\nObjective function = " + objective;

        res = res + "\nTime of SeqWeightedOutliers = " + time_needed;

        return res;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Input reading methods, from moodle
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }


    // Model for output : initial_guess_r, final_guess_r, number_of_guesses
    private static class OutputModel {
        private double initial_guess_r;
        private double final_guess_r;
        private int number_of_guesses;

        public OutputModel() {
            setInitial_guess_r(0.0);
            setFinal_guess_r(0.0);
            setNumber_of_guesses(1);
        }

        public double getInitial_guess_r() {
            return initial_guess_r;
        }

        public void setInitial_guess_r(double initial_guess_r) {
            this.initial_guess_r = initial_guess_r;
        }

        public double getFinal_guess_r() {
            return final_guess_r;
        }

        public void setFinal_guess_r(double final_guess_r) {
            this.final_guess_r = final_guess_r;
        }

        public int getNumber_of_guesses() {
            return number_of_guesses;
        }

        public void setNumber_of_guesses(int number_of_guesses) {
            this.number_of_guesses = number_of_guesses;
        }

        public void increment_Number_of_Guesses() {
            number_of_guesses++;
        }
    }

}
