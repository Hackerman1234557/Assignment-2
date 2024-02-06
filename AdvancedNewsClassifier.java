package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public List<Integer> listLengths = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        //TODO Task 6.1 - 5 Marks
        for (int i = 0 ; i <  Toolkit.getListVocabulary().size(); i++){
            String term = Toolkit.getListVocabulary().get(i);
            if (!checkStopWord(term)){
                Vector vector = new Vector(Toolkit.listVectors.get(i));
                Glove glove = new Glove(term, vector);
                listResult.add(glove);
            }
        }
        return listResult;
    }

    public boolean checkStopWord(String searchTerm){
        for (String word : Toolkit.STOPWORDS){
            if (word.equals(searchTerm)){
                return true;
            }
        }
        return false;
    }


    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian = -1;
        //TODO Task 6.2 - 5 Marks
        listLengths = new ArrayList<>();
        Integer[] temp = new Integer[listEmbedding.size()];
        for (int i = 0; i < _listEmbedding.size(); i++){
            int l = 0;
            for (String term : listEmbedding.get(i).getNewsContent().split("\\s+")){
                for (Glove word : listGlove){
                    if (word.getVocabulary().equals(term)){
                        l++;
                    }
                }
            }
            listLengths.add(l);
        }
        mergeSort(listLengths.toArray(temp), listLengths.size());
        for (int i = 0; i < temp.length; i++){
            listLengths.set(i, temp[i]);
        }
        if (listLengths.size() % 2 != 0){
            intMedian = listLengths.get(listLengths.size() / 2);
        } else {
            intMedian = (listLengths.get(listLengths.size()/2) + listLengths.get(listLengths.size()/2 + 1))/2;
        }
        return intMedian;
    }

    public static void mergeSort(Integer[] a, int n){
        if (n < 2){
            return;
        }
        int mid = n / 2;
        Integer[] l = new Integer[mid];
        Integer[] r = new Integer[n - mid];

        for (int i = 0; i < mid; i++){
            l[i] = a[i];
        }
        for (int i = mid; i < n; i++){
            r[i - mid] = a[i];
        }
        mergeSort(l, mid);
        mergeSort(r, n - mid);

        merge(a, l, r, mid, n - mid);
    }

    public static void merge(Integer[] a, Integer[] l, Integer[] r, int left, int right){
        int i = 0, j = 0, k = 0;
        while (i < left && j < right){
            if (l[i] <= r[j]) {
                a[k++] = l[i++];
            }
            else {
                a[k++] = r[j++];
            }
        }
        while (i < left){
            a[k++] = l[i++];
        }
        while (j < right){
            a[k++] = r[j++];
        }
    }

    public void populateEmbedding() {
        //TODO Task 6.3 - 10 Marks
        for (ArticlesEmbedding embedding : listEmbedding){
            try {
                embedding.getEmbedding();
            } catch (Exception e){
                if (e instanceof InvalidSizeException){
                    embedding.setEmbeddingSize(embeddingSize);
                } else if (e instanceof  InvalidTextException){
                    embedding.getNewsContent();
                }
            }
        }
    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null;
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;

        //TODO Task 6.4 - 8 Marks
        for (ArticlesEmbedding embedding : listEmbedding){
            int label = Integer.parseInt(embedding.getNewsLabel());
            if (embedding.getNewsType().equals(NewsArticles.DataType.Training)){
                outputNDArray = Nd4j.zeros(1, _numberOfClasses);
                inputNDArray = embedding.getEmbedding();
                outputNDArray.putScalar(0, label - 1, 1);
                DataSet dataSet = new DataSet(inputNDArray, outputNDArray);
                listDS.add(dataSet);
            }
        }
        return new ListDataSetIterator(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        //TODO Task 6.5 - 8 Marks
        for (ArticlesEmbedding embedding : _listEmbedding){
            if (embedding.getNewsType().equals(NewsArticles.DataType.Testing)){
                int[] predict = myNeuralNetwork.predict(embedding.getEmbedding());
                for (int i : predict){
                    listResult.add(i);
                    embedding.setNewsLabel(Integer.toString(i));
                }

            }
        }
        return listResult;
    }

    public void printResults() {
        //TODO Task 6.6 - 6.5 Marks
        List<String> results = new ArrayList<>();
        for (ArticlesEmbedding embedding : listEmbedding){
            int label = Integer.parseInt(embedding.getNewsLabel());
            if (embedding.getNewsType().equals(NewsArticles.DataType.Testing)){
               while (label + 1> results.size()){
                   results.add("");
               }
               String temp = results.get(label);
               results.set(label, temp + System.lineSeparator() + embedding.getNewsTitle());
            }
        }
        for (String group : results){
            String groupnum = "Group " + (results.indexOf(group) + 1);
            System.out.println(groupnum + group);
        }
    }
}
