package uob.oop;

import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> listVocabulary = null;
    public static List<double[]> listVectors = null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};

    public void loadGlove() throws IOException {
        listVocabulary = new ArrayList<>();
        listVectors = new ArrayList<>();
        try {
            FileReader fileReader = new FileReader(Toolkit.getFileFromResource(FILENAME_GLOVE));
            BufferedReader myReader = new BufferedReader(fileReader);
            //TODO Task 4.1 - 5 marks
            String line;
            while ((line = myReader.readLine()) != null){
                String[] array = line.split(",");
                listVocabulary.add(array[0]);
                double[] temp = new double[array.length - 1];
                for (int i = 1; i < array.length; i++){
                    temp[i - 1] = Double.valueOf(array[i]);
                }
                listVectors.add(temp);
        }
        } catch (FileNotFoundException | URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    private static File getFileFromResource(String fileName) throws URISyntaxException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.toURI());
        }
    }

    public List<NewsArticles> loadNews() {
        List<NewsArticles> listNews = new ArrayList<>();
        //TODO Task 4.2 - 5 Marks
        try {
            File newsFolder = new File(getFileFromResource("News").toURI());
            if (newsFolder.exists()) {
                File[] files = newsFolder.listFiles();
                if (files != null) {
                    for (File file : files) {
                        if (file.getName().toLowerCase().endsWith(".htm")) {
                            String html = Files.readString(file.toPath());
                            NewsArticles article = new NewsArticles(HtmlParser.getNewsTitle(html), HtmlParser.getNewsContent(html), HtmlParser.getDataType(html), HtmlParser.getLabel(html));
                            listNews.add(article);
                        }
                    }
                }
            }
        } catch (Exception e){
            throw new RuntimeException(e);
        }
        return listNews;
    }

    public static List<String> getListVocabulary() {
        return listVocabulary;
    }

    public static List<double[]> getlistVectors() {
        return listVectors;
    }
}
