package pl.biai.logic;

import com.atul.JavaOpenCV.Imshow;
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
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.highgui.Highgui;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
import org.opencv.ml.CvStatModel;
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
    private final int imageWidth = 144;
    private final int imageHeight = 33;

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

        Mat trainingImages = new Mat(0, imageWidth * imageHeight, CvType.CV_32FC1);
        Mat labels = new Mat(amountOfPlates + amountOfNoPlates, 1, CvType.CV_32SC1);
        List<Integer> trainingLabels = new ArrayList<>();

        //Dodaje prawidłowe fotki tablicy do Mat
        for (int i = 0; i < amountOfPlates; i++) {
            int index = i + 1;
            String file = pathPlates + index + ".jpg";

            Mat img = Highgui.imread(file, 0);
            img.convertTo(img, CvType.CV_32FC1);
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
            img.convertTo(img, CvType.CV_32FC1);
            //Zmniejszenie kanałów do 1 oraz rzędów do 1
            img = img.reshape(1, 1);
            trainingImages.push_back(img);
            trainingLabels.add(0);
        }
        //////////////////////////////////////////////////
        //Testowanie nowego ustawienia ///////////////////

        //Konwersja Listy do tablicy Integer       
        Integer[] array = trainingLabels.toArray(new Integer[trainingLabels.size()]);

        //Konwersja tablicy Integer do tablicy int...
        int[] trainLabels = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            trainLabels[i] = array[i];
        }

        //Załadowanie oznakowań do Mata
        for (int i = 0; i < trainingLabels.size(); i++) {
            labels.put(i, 1, trainLabels[i]);
        }

        CvSVMParams params = new CvSVMParams();
        params.set_svm_type(CvSVM.C_SVC);
        params.set_kernel_type(CvSVM.LINEAR);
        params.set_degree(0);
        params.set_gamma(1);
        params.set_coef0(0);
        params.set_C(1);
        params.set_nu(0);
        params.set_p(0);
        TermCriteria tc = new TermCriteria(opencv_core.CV_TERMCRIT_ITER, 200, 0.01);
        params.set_term_crit(tc);

        Size data = trainingImages.size();
        Size label = labels.size();

        //CvSVM svmClassifier = new CvSVM(trainingImages, labels, new Mat(), new Mat(), params);
        CvSVM svmClassifier = new CvSVM();

        Mat temp1 = new Mat();
        Mat temp2 = new Mat();
        //temp1.convertTo(temp1, CvType.CV_32SC1);
        //temp2.convertTo(temp2, CvType.CV_32SC1);

        svmClassifier.train(trainingImages, labels, temp1, temp2, params);
        //svmClassifier.save("test.xml");

        //DZIAŁĄ KURWA DZIAŁA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for (int i = 1; i < 11; i++) {

            //Mat cropImg = new Mat(0, imageWidth * imageHeight, CvType.CV_32FC1);
            Mat temp = Highgui.imread("cropped_images\\cropped" + i + ".jpg", 0);
            temp.convertTo(temp, CvType.CV_32FC1);
            temp = temp.reshape(1, 1);

            Size afterchanges = temp.size();

            //cropImg.push_back(temp);

            //Size testcrop = cropImg.size();
            Size testtemp = temp.size();
            float response = svmClassifier.predict(temp);

            System.out.println("Response is equal: " + response);
        }

        /*
         ////////////////////////////////////////////////
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

         /*
         //Konwersja Listy do tablicy Integer       
         Integer[] array = trainingLabels.toArray(new Integer[trainingLabels.size()]);

         //Konwersja tablicy Integer do tablicy int...
         int[] trainLabels = new int[array.length];
         for (int i = 0; i < array.length; i++) {
         trainLabels[i] = array[i];
         }*/
        //Utworzenie Mat z oznaczeniami i zapis do SVM.xml
        //opencv_core.Mat trainLabelsMat = new opencv_core.Mat(trainLabels);
        // opencv_core.write(fs, "TrainingLabels", trainLabelsMat);
        //Zamknięcie SVM.xml
        //fs.release();
    }
}
