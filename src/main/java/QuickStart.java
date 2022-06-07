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
        List<Row> records = new ArrayList<>();
        boolean header = true;
        try(BufferedReader br = new BufferedReader(new FileReader("beers.csv"))){
            String line;
            while((line = br.readLine()) != null){
                if(header){
                    header = false;
                    continue;
                }
                String[] values = line.split(",");
                records.add(Row.of(values));
            }
        }
        catch (Exception ignored) {

        }

        DataStream<Row> immStream = env.fromCollection(records, new RowTypeInfo(
                new TypeInformation[]{
                        TypeInformation.of(DenseVector.class),
                        Types.INT
                },
                new String[] {"Feat1", "Feat2", "Pred"}
        ));

        DataStream<Row> inputStream = immStream
                .filter(new FilterFunction<Row>() {
            @Override
            public boolean filter(Row row) throws Exception {
                for (int i = 0; i < row.getArity(); i++)
                    if (row.getFieldAs(i).toString().length() == 0)
                        return false;
                return true;
            }
        })
                .map(new MapFunction<Row, Row>() {
                    @Override
                    public Row map(Row row) throws Exception {
                        return Row.of(Vectors.dense(Double.parseDouble(row.getFieldAs(1)), Double.parseDouble(row.getFieldAs(2))), Integer.parseInt(row.getFieldAs(7)));
                    }
                });

        Table input = tEnv.fromDataStream(inputStream);

    }
}