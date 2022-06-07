import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.clustering.kmeans.KMeans;
import org.apache.flink.ml.clustering.kmeans.KMeansModel;
import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class QuickStart {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        String featuresCol = "features";
        String predictionCol = "prediction";

        // Generate train data and predict data as DataStream.
        List<String[]> records = new ArrayList<>();
        boolean header = true;
        try(BufferedReader br = new BufferedReader(new FileReader("beers.csv"))){
            String line;
            while((line = br.readLine()) != null){
                if(header){
                    header = false;
                    continue;
                }
                String[] values = line.split(",");
                records.add(values);
            }
        }
        catch (Exception ignored) {

        }

        DataStream<String[]> immStream = env.fromCollection(records, new RowTypeInfo(
                new TypeInformation[]{
                        Types.DOUBLE,
                        Types.DOUBLE,
                        Types.DOUBLE
                },
                new String[] {"Feat1", "Feat2", "Pred"}
        ));
        DataStream<DenseVector> inputStream = immStream
                .filter(new FilterFunction<String[]>() {
            @Override
            public boolean filter(String[] strings) throws Exception {
                for (String s : strings)
                    if (s.length() == 0)
                        return false;
                return true;
            }
        })
                .map(new MapFunction<String[], DenseVector>() {
                    @Override
                    public DenseVector map(String[] strings) throws Exception {
                        return Vectors.dense(Double.parseDouble(strings[1]), Double.parseDouble(strings[2]), Double.parseDouble(strings[7]));
                    }
                });

        inputStream.getType() =
        Table input = tEnv.fromDataStream(inputStream).as(featuresCol)


        // Convert data from DataStream to Table, as Flink ML uses Table API.
        Table input = tEnv.fromDataStream(inputStream).as(featuresCol);

        // Creates a K-means object and initialize its parameters.
        KMeans kmeans = new KMeans()
                .setK(2)
                .setSeed(1L)
                .setFeaturesCol(featuresCol)
                .setPredictionCol(predictionCol);

        // Trains the K-means Model.
        KMeansModel model = kmeans.fit(input);

        // Use the K-means Model for predictions.
        Table output = model.transform(input)[0];

        CloseableIterator<Row> it = output.execute().collect();
        System.out.println(it.toString());


        // Extracts and displays prediction result.
        /*for (CloseableIterator<Row> it = output.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector vector = (DenseVector) row.getField(featuresCol);
            int clusterId = (Integer) row.getField(predictionCol);
            System.out.println("Vector: " + vector + "\tCluster ID: " + clusterId);
        }*/
    }
}