import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.classification.naivebayes.NaiveBayes;
import org.apache.flink.ml.classification.naivebayes.NaiveBayesModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class QuickStart {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generate train data and predict data as DataStream.
        HashMap<String, Integer> Mapping = new HashMap<>();
        int j = 0;
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
                if (values.length == 8){
                    Mapping.putIfAbsent(values[5], j);
                    values[5] = String.valueOf(j++);
                    records.add(Row.of(values));
                }
            }
        }
        catch (Exception ignored) {

        }

        DataStream<Row> immStream = env.fromCollection(records);

        DataStream<Row> inputStream = immStream
                .filter((FilterFunction<Row>) row -> {
                    for (int i = 0; i < row.getArity(); i++)
                        if (row.getFieldAs(i).toString().length() == 0)
                            return false;
                    return true;
                })
                .map((MapFunction<Row, Row>) row -> Row.of(Vectors.dense(Double.parseDouble(row.getFieldAs(1)), Double.parseDouble(row.getFieldAs(2)), Double.parseDouble(row.getFieldAs(7))), Integer.valueOf(row.getFieldAs(5))),
                        new RowTypeInfo(
                                new TypeInformation[]{
                                        TypeInformation.of(DenseVector.class),
                                        Types.INT
                                },
                                new String[] {"features", "label"}
                        ));

        Table input = tEnv.fromDataStream(inputStream);

        NaiveBayes nb = new NaiveBayes()
                        .setSmoothing(1.0)
                        .setFeaturesCol("features")
                        .setLabelCol("label");
        NaiveBayesModel model = nb.fit(input);

        Table output = model.transform(input)[0];

        output.execute().print();
    }
}