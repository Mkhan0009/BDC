import javafx.util.Pair;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.*;

public class G017HW3 {

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// MAIN PROGRAM 
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    // Global variables
    public static OutputModel outputModel = new OutputModel();

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: filepath k z L");
        }

        // ----- Initialize variables
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);
        long start, end; // variables for time measurements

        // ----- Set Spark Configuration
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkSession conf = SparkSession
                .builder()
                .appName("Homework 1")
                .config("spark.master", "local")
                .getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(conf.sparkContext());
        sc.setLogLevel("WARN");

        // ----- Read points from file
        start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(args[0], L)
                .map(x -> strToVector(x))
                .repartition(L)
                .cache();
        long N = inputPoints.count();
        end = System.currentTimeMillis();

        // ----- Pring input parameters
        System.out.println("File : " + filename);
        System.out.println("Number of points N = " + N);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Number of partitions L = " + L);
        System.out.println("Time to read from file: " + (end - start) + " ms");

        // ---- Solve the problem
        ArrayList<Vector> solution = MR_kCenterOutliers(inputPoints, k, z, L);

        System.out.println("Initial guess = " + outputModel.getInitial_guess_r());
        System.out.println("Final guess = " + outputModel.getFinal_guess_r());
        System.out.println("Number of guesses = " + outputModel.getNumber_of_guesses());

        // ---- Compute the value of the objective function
        start = System.currentTimeMillis();
        double objective = computeObjective(inputPoints, solution, z);
        end = System.currentTimeMillis();
        System.out.println("Objective function = " + objective);
        System.out.println("Time to compute objective function: " + (end - start) + " ms");

    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// AUXILIARY METHODS
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method strToVector: input reading
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method euclidean: distance function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method MR_kCenterOutliers: MR algorithm for k-center with outliers 
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> MR_kCenterOutliers(JavaRDD<Vector> points, int k, int z, int L) {

        //------------- ROUND 1 ---------------------------

        long startOfRound1 = System.currentTimeMillis();
        JavaRDD<Tuple2<Vector, Long>> coreset = points.mapPartitions(x ->
        {
            ArrayList<Vector> partition = new ArrayList<>();
            while (x.hasNext()) partition.add(x.next());
            ArrayList<Vector> centers = kCenterFFT(partition, k + z + 1);
            ArrayList<Long> weights = computeWeights(partition, centers);
            ArrayList<Tuple2<Vector, Long>> c_w = new ArrayList<>();
            for (int i = 0; i < centers.size(); ++i) {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weights.get(i));
                c_w.add(i, entry);
            }
            return c_w.iterator();
        }); // END OF ROUND 1


        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k + z + 1) * L);
        elems.addAll(coreset.collect());

        long endOfRound1 = System.currentTimeMillis();

        System.out.println("Time Round 1: " + (endOfRound1 - startOfRound1) + " ms");

        //------------- ROUND 2 ---------------------------
        long startOfRound2 = System.currentTimeMillis();
        //
        // ****** ADD YOUR CODE
        // ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
        // ****** Measure and print times taken by Round 1 and Round 2, separately
        // ****** Return the final solution
        //

        Pair<ArrayList<Vector>, ArrayList<Long>> temp = extractPointsFromListOfTuple(elems);

        ArrayList<Vector> listOfCenters = SeqWeightedOutliers(temp.getKey(), temp.getValue(), k, z, 2);
        long endOfRound2 = System.currentTimeMillis();

        System.out.println("Time Round 2: " + (endOfRound2 - startOfRound2) + " ms");
        return listOfCenters;
    }

    // auxiliary method to convert from Tuple2 -> arrayList
    private static Pair<ArrayList<Vector>, ArrayList<Long>> extractPointsFromListOfTuple(ArrayList<Tuple2<Vector, Long>> elems) {
        ArrayList<Vector> points = new ArrayList<>();
        ArrayList<Long> weights = new ArrayList<>();

        elems.forEach(vectorLongTuple2 -> {
            points.add(vectorLongTuple2._1);
            weights.add(vectorLongTuple2._2);
        });

        return new Pair<>(points, weights);
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method kCenterFFT: Farthest-First Traversal
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> kCenterFFT(ArrayList<Vector> points, int k) {

        final int n = points.size();
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.POSITIVE_INFINITY);

        ArrayList<Vector> centers = new ArrayList<>(k);

        Vector lastCenter = points.get(0);
        centers.add(lastCenter);
        double radius = 0;

        for (int iter = 1; iter < k; iter++) {
            int maxIdx = 0;
            double maxDist = 0;

            for (int i = 0; i < n; i++) {
                double d = euclidean(points.get(i), lastCenter);
                if (d < minDistances[i]) {
                    minDistances[i] = d;
                }

                if (minDistances[i] > maxDist) {
                    maxDist = minDistances[i];
                    maxIdx = i;
                }
            }

            lastCenter = points.get(maxIdx);
            centers.add(lastCenter);
        }
        return centers;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeWeights: compute weights of coreset points
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Long> computeWeights(ArrayList<Vector> points, ArrayList<Vector> centers) {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for (int i = 0; i < points.size(); ++i) {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for (int j = 1; j < centers.size(); ++j) {
                if (euclidean(points.get(i), centers.get(j)) < tmp) {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            // System.out.println("Point = " + points.get(i) + " Center = " + centers.get(mycenter));
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method SeqWeightedOutliers: sequential k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

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
        double r_min = euclidean(P.get(0), P.get(1)); // R_min initialization

        // computing all distance of the first ( k + z + 1) / 2 points
        for (int i = 0; i < (k + z + 1); i++) {
            // don't need to calculate distance between same point
            for (int j = i + 1; j < (k + z + 1); j++) {

                // Given two points x and y, instances of Vector, their Euclidean L2-distance can be computed by invoking: Math.sqrt(Vectors.sqdist(x, y))
                double distance = euclidean(P.get(i), P.get(j));

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
                        double distance = euclidean(x, y);

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
                    double distance = euclidean(new_center, v);

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

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeObjective: computes objective function  
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double computeObjective(JavaRDD<Vector> P, ArrayList<Vector> S, int z) {

        //Incorporating probable edge cases

        //  if dataset size equals to the size of centers or if list of centers is empty then the maximum
        //  distance of any point from their respective center must be zero hence the objective function
        //  is returning 0
//        if ((P.count() == S.size()) || S.isEmpty()) {
//            return 0;
//        }

        double max_distance = 0;


        // Container of all distances
        HashSet<Double> distances = new HashSet<>();
        // compute all distances d(x,S), for every x in P
        // find minimum distance from all data points (x) from respective center (s)
        JavaRDD<Double> vectors = P.mapPartitions(input ->
        {
            Vector x;
            while (input.hasNext()) {
                x = input.next();
                double minDistance = euclidean(x, S.get(0));
                for (Vector s : S) {
                    double curr_distance = euclidean(x, s);
                    if (curr_distance < minDistance) {
                        minDistance = curr_distance;
                    }
                }
                distances.add(minDistance);
            }
            ArrayList<Double> distanceList = new ArrayList<>(distances);
            Collections.sort(distanceList);
            List<Double> subList = distanceList.subList(distanceList.size() - z - 1, distanceList.size());
            return subList.iterator();
        });

        // collecting distances from JavaRDD
        List<Double> distancesNew = vectors.collect();

        // sort distances in ascending order
        ArrayList<Double> distanceList = new ArrayList<>(distancesNew);
        Collections.sort(distanceList);

        // exclude the z-largest distances which are the distances from outliers
        // return the largest among the remaining ones
        max_distance = distanceList.get(distanceList.size() - z - 1);
        return max_distance;

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
