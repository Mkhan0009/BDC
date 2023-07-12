import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.*;

public class G017HW1 {

    public static void main(String[] args) {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: num_partitions, popularity, country, <path_to_file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: num_partitions H_popularity S_country file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkSession conf = SparkSession
                .builder()
                .appName("Homework 1")
                .config("spark.master", "local")
                .getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(conf.sparkContext());
        sc.setLogLevel("WARN");

        //Part 1

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read popularity
        int H = Integer.parseInt(args[1]);

        // Read country
        String S = args[2];


        // Read input file and subdivide it into K random partitions
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // DISPLAY INPUT
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        long numberOfRows;
        numberOfRows = rawData.count();
        System.out.println("Number of Rows = " + numberOfRows);

        //Part 2
        HashSet<Tuple2<String, Integer>> productCustomerSet = new HashSet<>();
        ArrayList<Tuple2<String, Integer>> productPopularity1ArrayList = new ArrayList<>();
        HashMap<String, Integer> counts = new HashMap<>();
        JavaPairRDD<String, Integer> productCustomer = rawData.mapPartitionsToPair((document) -> {

            // Product details
            String product;
            String country;
            Integer customerId;
            double quantity;

            while (document.hasNext()) {
                String mainDocument = document.next();
                String[] tokens = mainDocument.split(",");
                quantity = Double.parseDouble(tokens[3]);
                product = tokens[1];
                country = tokens[7];
                customerId = Integer.parseInt(tokens[6]);
                //checking if quantity is greater then 0
                if (quantity > 0) {
                    //checking country condition
                    //ignoring country condition if country = all
                    if (S.equals("all")) {
                        productCustomerSet.add(new Tuple2<>(product, customerId));
                    }
                    //applying country condition
                    else {
                        if (country.equals(S)) {
                            productCustomerSet.add(new Tuple2<>(product, customerId));
                        }
                    }
                }
            }
            return productCustomerSet.iterator();
        });


        // print the number of pairs of productCustomers

        List<Tuple2<String, Integer>> stringIntegerListC = productCustomer.distinct().collect();

        System.out.println("Product-Customer Pairs = " + stringIntegerListC.size());

        //Part 3

        JavaPairRDD<String, Integer> productPopularity1 =
                //REDUCE PHASE (R1)
                //use mapPartitionsToPair for productPopularity1
                productCustomer.distinct()
                        .mapPartitionsToPair(input -> {
                            while (input.hasNext()) {
                                Tuple2<String, Integer> token = input.next();
                                counts.put(token._1(), 1 + counts.getOrDefault(token._1(), 0));
                            }
                            for (Map.Entry<String, Integer> e : counts.entrySet()) {
                                productPopularity1ArrayList.add(new Tuple2<>(e.getKey(), e.getValue()));
                            }
                            return productPopularity1ArrayList.iterator();
                        })
                        //group by key (product)
                        //REDUCE PHASE (R2)
                        .groupByKey()
                        .mapValues((element) -> {
                            int sum = 0;

                            for (int e : element) {
                                sum += e;
                            }
                            return sum;
                        });



        /*
       Part 4

       reduce by key partitions
       REDUCE 1

       */

        JavaPairRDD<String, Integer> productPopularity2 =
                productCustomer.distinct()
                        //MAP PHASE (R1)
                        .mapToPair(input -> (new Tuple2<>(input._1(), 1)))
                        //REDUCE PHASE (R1)
                        .reduceByKey(Integer::sum, K)
                        //REDUCE PHASE (R2)
                        .reduceByKey(Integer::sum);


        //Part 5
        //get the H Highest-popular products


        if (H > 0) {
            List<Tuple2<Integer, String>> listHPopularity = productPopularity1
                    //swap key and value
                    .mapToPair(Tuple2::swap)
                    //and sort by key (false = descending)
                    //taking highest popularity products
                    .sortByKey(false)
                    .take(H);

            //print the results
            System.out.println("Top " + H + " Products and their Popularities");
            for (Tuple2<Integer, String> res : listHPopularity)
                System.out.print("Product " + res._2() + " Popularity " + res._1() + "; ");
        }

        //Part 6
        //Collects all pairs of productPopularity1 and productPopularity2 into a list respectively and print all of them
        else {
            System.out.println("Product Popularity 1:");
            printPopularities(productPopularity1.sortByKey(true).collect());
            System.out.println("\nProduct Popularity 2:");
            printPopularities(productPopularity2.sortByKey(true).collect());
        }


    }

    //method for displaying product popularities

    static void printPopularities(List<Tuple2<String, Integer>> stringIntegerJavaPairRDD) {
        for (Tuple2<String, Integer> res : stringIntegerJavaPairRDD)
            System.out.print("Product: " + res._1() + " Popularity: " + res._2() + "; ");
    }


}
