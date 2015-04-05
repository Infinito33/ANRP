package pl.biai.logic;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import org.bytedeco.javacpp.opencv_core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.Highgui;
import org.opencv.utils.Converters;

/**
 * Prepare training .xml file which contains converted MAT data.
 *
 * @author Fufer
 * @version 1.0
 */
public class SVMTrainCreator {

    private final String pathPlates;
    private final String pathNoPlates;
    private final int amountOfPlates;
    private final int amountOfNoPlates;

    /**
     * Initializes default amounts
     *
     * @param amountOfPlates Number of good plate's photos.
     * @param amountOfNoPlates Number of wrong plate's photos.
     */
    public SVMTrainCreator(int amountOfPlates, int amountOfNoPlates) {
        this.pathPlates = "plateTraining\\plate\\";
        this.pathNoPlates = "plateTraining\\Noplate\\";
        this.amountOfPlates = amountOfPlates;
        this.amountOfNoPlates = amountOfNoPlates;
    }

    /**
     * Prepares .xml file in main project folder.
     */
    public void prepareTrainingFile() {

        Mat classes = new Mat();
        Mat trainingData = new Mat();
        Mat trainingImages = new Mat();
        List<Integer> trainingLabels = new ArrayList<>();

        //Dodaje prawidłowe fotki tablicy do Mat
        for (int i = 0; i < amountOfPlates; i++) {
            int index = i + 1;
            String file = pathPlates + index + ".jpg";

            Mat img = Highgui.imread(file, 0);
            //Zmniejszenie kanałów do 1 oraz rzędów do 1
            img = img.reshape(1, 1);
            trainingImages.push_back(img);
            trainingLabels.add(1);
        }

        //Dodaje nieprawidłowe fotki do drugiego Mat
        for (int i = 0; i < amountOfNoPlates; i++) {
            int index = i + 1;
            String file = pathNoPlates + index + ".jpg";
            Mat img = Highgui.imread(file, 0);
            //Zmniejszenie kanałów do 1 oraz rzędów do 1
            img = img.reshape(1, 1);
            trainingImages.push_back(img);
            trainingLabels.add(0);
        }

        //Kopia wszystkich fotek i konwersja
        trainingImages.copyTo(trainingData);
        trainingData.convertTo(trainingData, CvType.CV_32FC1);

        //Konwersja listy oznaczeń "1" i "0" do postaci Mat
        classes = Converters.vector_int_to_Mat(trainingLabels);

        //Otwarcie pliku SVM.xml do zapisu poprzez inny rodzaj OpenCV - tu javacpp.
        //W naszym OpenCV nie było wrappera do FileStorage niestety.
        opencv_core.FileStorage fs = new opencv_core.FileStorage("SVM.xml", opencv_core.FileStorage.WRITE);

        //Tu będzie zbiór wszystkich fotek w postaci Mat
        opencv_core.Mat finalMat = null;

        //KONWERSJA na opencv_core.Mat bo inaczej nie użyjemy fs.write
        MatOfByte matOfByte = new MatOfByte();

        Highgui.imencode(".jpg", trainingData, matOfByte);
        byte[] byteArray = matOfByte.toArray();
        BufferedImage buffImage = null;

        try {

            InputStream in = new ByteArrayInputStream(byteArray);
            buffImage = ImageIO.read(in);
        } catch (Exception e) {
            e.printStackTrace();
        }
        finalMat = opencv_core.Mat.createFrom(buffImage);
        //Koniec konwersji /////////////////////////////////////////////

        //Zapis danych MAT do SVM.xml
        opencv_core.write(fs, "TrainingData", finalMat);

        //Konwersja Listy do tablicy Integer       
        Integer[] array = trainingLabels.toArray(new Integer[trainingLabels.size()]);

        //Konwersja tablicy Integer do tablicy int...
        int[] trainLabels = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            trainLabels[i] = array[i];
        }

        //Utworzenie Mat z oznaczeniami i zapis do SVM.xml
        opencv_core.Mat trainLabelsMat = new opencv_core.Mat(trainLabels);
        opencv_core.write(fs, "TrainingLabels", trainLabelsMat);

        //Zamknięcie SVM.xml
        fs.release();
    }
}
