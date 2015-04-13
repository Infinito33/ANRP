package pl.biai.logic;

import com.atul.JavaOpenCV.Imshow;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
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
 * Creates SVM Classifier for predicting if image is a plate or not. Also can
 * prepare .xml file with SVM data.
 *
 * @author Fufer
 * @version 1.0
 */
public class SVMTrainCreator {
    //BUM CYK CKY

    /**
     * Path to photos with plates.
     */
    private final String pathPlates;

    /**
     * Path to photos with no plates.
     */
    private final String pathNoPlates;

    /**
     * Amount of plates photos passed in constructor.
     */
    private int amountOfPlates;

    /**
     * Amount of no-plates photos passed in constructor.
     */
    private int amountOfNoPlates;

    /**
     * Const image width for cropped images.
     */
    private final int imageWidth = 144;

    /**
     * Const image height for cropped images.
     */
    private final int imageHeight = 33;

    /**
     * SVM object which will predict if a photo is plate or not. It will load
     * .xml file with settings.
     */
    private CvSVM svmClassifier;

    /**
     * Initializes default amounts.
     *
     */
    public SVMTrainCreator() {
        this.pathPlates = "plateTraining\\plate\\";
        this.pathNoPlates = "plateTraining\\Noplate\\";
        this.amountOfPlates = 0;
        this.amountOfNoPlates = 0;
        this.svmClassifier = new CvSVM();
    }

    /**
     * Prepares .xml file in main project folder for SVM training.
     *
     * @param amountOfPlates Amount of photos with plates.
     * @param amountOfNoPlates Amount of photos with no plates.
     */
    public void prepareTrainingFile(int amountOfPlates, int amountOfNoPlates) {
        this.amountOfPlates = amountOfPlates;
        this.amountOfNoPlates = amountOfNoPlates;

        Mat trainingImages = new Mat(0, imageWidth * imageHeight, CvType.CV_32FC1);
        List<Double> trainingLabels = new ArrayList<>();

        //Dodaje prawidłowe fotki
        for (int i = 0; i < amountOfPlates; i++) {
            int index = i + 1;
            String file = pathPlates + index + ".jpg";

            Mat img = Highgui.imread(file, 0);

            //Zmniejszenie kanałów do 1 oraz rzędów do 1
            img = img.reshape(1, 1);
            img.convertTo(img, CvType.CV_32FC1);
            trainingImages.push_back(img);
            trainingLabels.add(1.0);
        }

        //Dodaje nieprawidłowe fotki
        for (int i = 0; i < amountOfNoPlates; i++) {
            int index = i + 1;
            String file = pathNoPlates + index + ".jpg";
            Mat img = Highgui.imread(file, 0);

            //Zmniejszenie kanałów do 1 oraz rzędów do 1
            img = img.reshape(1, 1);
            img.convertTo(img, CvType.CV_32FC1);
            trainingImages.push_back(img);
            trainingLabels.add(-1.0);
        }

        //Konwersja Listy do tablicy Double       
        Double[] array = trainingLabels.toArray(new Double[trainingLabels.size()]);

        //Konwersja tablicy Double do tablicy double...
        double[] trainLabels = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            trainLabels[i] = array[i];
        }

        Mat positives = new Mat(amountOfPlates, 1, CvType.CV_32SC1, new Scalar(1.0));
        Mat negatives = new Mat(amountOfNoPlates, 1, CvType.CV_32SC1, new Scalar(0.0));

        //Utworzenie macierzy z 1 i 0
        Mat labels = new Mat(0, 1, CvType.CV_32SC1);
        labels.push_back(positives);
        labels.push_back(negatives);

        //System.out.println(labels.dump());
        CvSVMParams params = new CvSVMParams();
        params.set_svm_type(CvSVM.C_SVC);
        params.set_kernel_type(CvSVM.LINEAR);
        params.set_degree(0);
        params.set_gamma(1);
        params.set_coef0(0);
        params.set_C(1);
        params.set_nu(0);
        params.set_p(0);
        TermCriteria tc = new TermCriteria(TermCriteria.MAX_ITER, 300, 0.01);
        params.set_term_crit(tc);

        svmClassifier.train(trainingImages, labels, new Mat(), new Mat(), params);
        svmClassifier.save("SvmData.xml");

        for (int i = 1; i < 9; i++) {

            //Mat cropImg = new Mat(0, imageWidth * imageHeight, CvType.CV_32FC1);
            Mat temp = Highgui.imread("cropped_images\\" + i + ".jpg", 0);
            temp.convertTo(temp, CvType.CV_32FC1);
            temp = temp.reshape(1, 1);

            Size afterchanges = temp.size();

            //cropImg.push_back(temp);
            //Size testcrop = cropImg.size();
            Size testtemp = temp.size();
            float response = svmClassifier.predict(temp);

            System.out.println("Response is equal: " + response);
        }

    }

    /**
     * Loads training data into SVM classifier.
     */
    public void loadTrainingData() {
        svmClassifier.load("SvmData.xml");
    }

    /**
     * @return the svmClassifier
     */
    public CvSVM getSvmClassifier() {
        return svmClassifier;
    }

    /**
     * @param svmClassifier the svmClassifier to set
     */
    public void setSvmClassifier(CvSVM svmClassifier) {
        this.svmClassifier = svmClassifier;
    }
}
