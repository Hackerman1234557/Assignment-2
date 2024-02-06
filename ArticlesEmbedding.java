package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Properties;



public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";

    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        //TODO Task 5.1 - 1 Mark
        super(_title,_content,_type,_label);
    }

    public void setEmbeddingSize(int _size) {
        //TODO Task 5.2 - 0.5 Marks
        intSize = _size;

    }

    public int getEmbeddingSize(){
        return intSize;
    }

    @Override
    public String getNewsContent() {
        //TODO Task 5.3 - 10 Marks
        if (processedText.isEmpty()) {
            processedText = textCleaning(super.getNewsContent());
            Properties properties = new Properties();
            properties.setProperty("annotators", "tokenize,pos,lemma");
            StanfordCoreNLP pipeline = new StanfordCoreNLP(properties);
            Annotation annotation = pipeline.process(processedText);
            StringBuilder sb = new StringBuilder();
            for (CoreLabel token : annotation.get(CoreAnnotations.TokensAnnotation.class)) {
                String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
                sb.append(lemma).append(" ");
            }
            processedText = sb.toString().toLowerCase();
            StringBuilder stringBuilder = new StringBuilder();
            for (String term : processedText.split("\\s+")) {
                boolean haltwordfound = false;
                for (String word : Toolkit.STOPWORDS) {
                    if (word.equals(term)) {
                        haltwordfound = true;
                        break;
                    }
                }
                if (!haltwordfound) {
                    stringBuilder.append(term).append(" ");
                }
            }
            processedText = stringBuilder.toString();
            return processedText.trim();
        } else {
            return processedText;
        }
    }

    public INDArray getEmbedding() throws Exception {
        //TODO Task 5.4 - 20 Marks
        try {
            if (!newsEmbedding.isEmpty()){
                return newsEmbedding;
            }
            String[] textList = processedText.split("\\s+");
            if (intSize == -1){
                throw new InvalidSizeException("Invalid size");
            }
            if (processedText.isEmpty()){
                throw new InvalidTextException("Invalid text");
            }
            newsEmbedding = Nd4j.zeros(intSize, AdvancedNewsClassifier.listGlove.get(0).getVector().getVectorSize());
            int count = 0;
            for (int j = 0; j < textList.length; j++){
                if (count >= intSize){
                    break;
                }
                for (int i = 0; i < AdvancedNewsClassifier.listGlove.size(); i++){
                    if (textList[j].equals(AdvancedNewsClassifier.listGlove.get(i).getVocabulary())){
                        Vector vector = AdvancedNewsClassifier.listGlove.get(i).getVector();
                        INDArray row = Nd4j.create(vector.getAllElements());
                        newsEmbedding.putRow(count, row);
                        count++;
                    }
                }
            }
        } catch (InvalidSizeException | InvalidTextException e){
            if (e instanceof InvalidSizeException){
                throw new InvalidSizeException("Invalid Size");
            } else {
                throw new InvalidTextException("Invalid text");
            }
        }
        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
}
