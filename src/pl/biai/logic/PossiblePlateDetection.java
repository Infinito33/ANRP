package pl.biai.logic;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;

/**
 * @author Fufer
 * @version 1.0
 */
public class PossiblePlateDetection {

    /**
     * Photo with plates loaded by file chooser.
     */
    private BufferedImage buffImage = null;

    /**
     * Local path to file with edited photo.
     */
    private static String filePath = null;

    /**
     * Original copy of photo.
     */
    private Mat photoOriginal = null;

    /**
     * Copy of photo which will be edited.
     */
    private Mat photoEdited = null;

    public Mat mask = null;

    public static int testPlateCount = 1;

    /**
     * Constructor.
     */
    public PossiblePlateDetection() {

    }

    /**
     * Updates photo displayed on screen
     *
     * @param originalPhoto if true - display original photo, false - display
     * edited photo.
     */
    public void updatePhoto(boolean originalPhoto) {
        MatOfByte matOfByte = new MatOfByte();

        if (originalPhoto) {
            Highgui.imencode(".jpg", photoOriginal, matOfByte);
        } else {
            Highgui.imencode(".jpg", photoEdited, matOfByte);
        }

        byte[] byteArray = matOfByte.toArray();
        buffImage = null;

        try {

            InputStream in = new ByteArrayInputStream(byteArray);
            buffImage = ImageIO.read(in);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Loads photo into Mat object based on local path.
     */
    public void loadPhotoToMat() {
        photoOriginal = Highgui.imread(filePath);
        photoEdited = Highgui.imread(filePath);
    }

    /**
     * Step 1 Makes gray scale of photo, gaussian blur for noise remove and
     * sobel filter.
     */
    public void makeCleanAndSobel() {
        Imgproc.cvtColor(photoEdited, photoEdited, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(photoEdited, photoEdited, new Size(0.0, 0.0), 1.0);
        Imgproc.Sobel(photoEdited, photoEdited, CvType.CV_8U, 1, 0, 3, 1, 0);
    }

    /**
     * Step 2 Makes threshold and morphology in order to specify regions which
     * in the plate can be.
     */
    public void makeThresholdAndMorphology() {
        Imgproc.threshold(photoEdited, photoEdited, 0, 255, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY);

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(17, 3));
        Imgproc.morphologyEx(photoEdited, photoEdited, Imgproc.MORPH_CLOSE, element);
    }

    /**
     * Finds possible places in form of rectangles where our plate can be. Uses
     */
    public void findPossiblePlateRects() {
        List<MatOfPoint> contoursList = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(photoEdited, contoursList, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        List<RotatedRect> rects = new ArrayList<>();

        for (MatOfPoint mop : contoursList) {
            MatOfPoint2f mop2f = new MatOfPoint2f(mop.toArray());
            RotatedRect rr = Imgproc.minAreaRect(mop2f);

            if (!verifySizes(rr)) {
                //contoursList.remove(mop);
            } else {
                rects.add(rr);
            }
        }

        //Wycina i zapisuje do plików wszystkie prostokąty, z czego jeden to tablica.
        cutAndSavePossiblePlate(rects);

        //W tym momencie w liście "rects" mamy możliwe prostokąty, jeden z nich to tablica.
        //Dalej jest część z flood fillem ktorej nie ogarniam i jak na moje oko to zle narazie dziala.
        //Maska jest zapisywana do mask.jpg jak cos, result z niebieskimi kreskami w result.jpg
        //Możesz posprawdzać i pokombinować coś wedlug tej ksiazki.
        /*
        
         Mat result = new Mat();
         photoOriginal.copyTo(result);
         Imgproc.drawContours(result, contoursList, -1, new Scalar(255, 0, 0), 1);

         for (RotatedRect rect : rects) {
         Core.circle(result, rect.center, 3, new Scalar(0, 255, 0), -1);
         //Core.circle(photoEdited, rect.center, 30, new Scalar(255,255,0), 50);
         //Specify minimum size between width and height
         double minSize = (rect.size.width < rect.size.height) ? rect.size.width : rect.size.height;
         minSize = minSize - minSize * 0.5;

         mask = new Mat(photoOriginal.rows() + 2, photoOriginal.cols() + 2, CvType.CV_8UC1);
         mask.setTo(Scalar.all(0));
         int loDiff = 30;
         int upDiff = 30;
         int connectivity = 4;
         int newMaskVal = 255;
         int numSeeds = 10;
         Rect ccomp = new Rect();

         int flags = connectivity + (newMaskVal << 8) + Imgproc.FLOODFILL_FIXED_RANGE + Imgproc.FLOODFILL_MASK_ONLY;

         Random rand = new Random();
         double range_d = minSize - (minSize / 2);
         int range = (int) range_d;

         for (int i = 0; i < numSeeds; i++) {
         Point seed = new Point();
         seed.x = rect.center.x + rand.nextInt(range);
         seed.y = rect.center.y + rand.nextInt(range);
         Core.circle(result, seed, 1, new Scalar(0, 255, 255), -1);
         //Core.circle(photoEdited, seed, 30, new Scalar(255,255,0), 50);
         int area = Imgproc.floodFill(photoOriginal, mask, seed, new Scalar(255, 0, 0), ccomp, new Scalar(loDiff, loDiff, loDiff), new Scalar(upDiff, upDiff, upDiff), flags);
         System.out.print(area + " ");
         }
         System.out.println();
         }

         List<Point> pointsOfInterest = new ArrayList<>();
         MatOfPoint2f mop2f = new MatOfPoint2f();
         int counter255 = 0;
         for (int i = 0; i < mask.rows(); i++) {
         for (int j = 0; j < mask.cols(); j++) {
         double[] pixel = new double[3];
         //mask.get(i, j, pixel);
         pixel = mask.get(i, j);

         if (pixel[0] == 255.0) {
         Point test = new Point(i, j);
         pointsOfInterest.add(test);
         //MatOfPoint mop = new MatOfPoint(test);
         //MatOfPoint2f mop2f = new MatOfPoint2f(mop.toArray());

         //pointsOfInterest.add(mop2f);
         counter255++;
         }

         //System.out.println("");
         }
         }
         mop2f.fromList(pointsOfInterest);
         RotatedRect minRect = Imgproc.minAreaRect(mop2f);

         System.out.println("Wykryto tyle rownych 255: " + counter255);
         int sum = mask.rows() * mask.cols();
         System.out.println("Suma pikseli: " + sum);
         //Zapis do pliku obrazków w celu widoku efektów
         Highgui.imwrite("result.jpg", result);
         Highgui.imwrite("mask.jpg", mask);
        
         */
    }

    /**
     * Cuts and saves all possible rectangles into .jpg files.
     *
     * @param rects List with rectangles.
     */
    public void cutAndSavePossiblePlate(List<RotatedRect> rects) {
        int rectCount = 0;
        for (RotatedRect rect : rects) {
            rectCount++;
            double r = (double) rect.size.width / (double) rect.size.height;
            double angle = rect.angle;
            if (r < 1) {
                angle = 90 + angle;
            }
            Mat rotmat = Imgproc.getRotationMatrix2D(rect.center, angle, 1);

            Mat rotatedImage = new Mat();
            Imgproc.warpAffine(photoOriginal, rotatedImage, rotmat, photoOriginal.size(), Imgproc.INTER_CUBIC);

            Size rectSize = rect.size;
            if (r < 1) {
                double temp;
                temp = rectSize.width;
                rectSize.width = rectSize.height;
                rectSize.height = temp;
            }
            Mat croppedImage = new Mat();
            //Wycinanka prostokąta.
            Imgproc.getRectSubPix(rotatedImage, rectSize, rect.center, croppedImage);

            //Nadanie ogólnych wymiarów każdemu z wyciętych prostokątów, w celu lepszego wykrywania
            //
            Mat equalizedImage = equalizeCroppedRect(croppedImage);

            Highgui.imwrite("cropped_images\\cropped" + rectCount + ".jpg", equalizedImage);
            //Highgui.imwrite("cropped_images\\cropped" + testPlateCount + "_" + rectCount + ".jpg", equalizedImage);
        }
    }

    /**
     * Equalize all possible rectangles into same size and light condition so
     * that detecting plate numbers could be easier.
     *
     * @param cropped Mat image with rectangle.
     * @return Equalized mat image with rectangle.
     */
    private Mat equalizeCroppedRect(Mat cropped) {
        Mat resultResized = new Mat();
        resultResized.create(33, 144, CvType.CV_8UC3);
        Imgproc.resize(cropped, resultResized, resultResized.size(), 0, 0, Imgproc.INTER_CUBIC);

        //Equalize cropped image with light histogram
        Mat grayResult = new Mat();
        Imgproc.cvtColor(resultResized, grayResult, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(grayResult, grayResult, new Size(3.0, 3.0), 0.0);
        Imgproc.equalizeHist(grayResult, grayResult);

        return grayResult;
    }

    /**
     * Verifies if rectangle has proper size, basing on polish plates and having
     * mistake error at 40%
     *
     * @param candidate Tested rectangle.
     * @return True if sizes are ok, otherwise false.
     */
    public static boolean verifySizes(RotatedRect candidate) {
        double error = 0.4;
        //Spain car plate size: 52x11 aspect 4,7272
        final double aspect = 4.5614;
        //Set a min and max area. All other patches are discarded
        //Polskie tablice mają 520mm x 114 mm, czyli stała to 4,5614
        double min = 15 * aspect * 15; // minimum area
        double max = 125 * aspect * 125; // maximum area
        //Get only patches that match to a respect ratio.
        double rmin = aspect - aspect * error;
        double rmax = aspect + aspect * error;
        double area = candidate.size.height * candidate.size.width;
        double r = (double) candidate.size.width / (double) candidate.size.height;
        if (r < 1) {
            r = 1 / r;
        }
        if ((area < min || area > max) || (r < rmin || r > rmax)) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * Filters all rectangles which might be plates, delete wrong ones and leave
     * only one photo with car plate.
     */
    public void photoFilter() {
        /*
        TaFileStorage tfs = new TaFileStorage();
        tfs.open("SVM.xml", TaFileStorage.READ);

        Mat SvmData = tfs.readMat("TrainingData");
        Mat SvmLabels = tfs.readMat("TrainingLabels");

        SvmData.convertTo(SvmData, CvType.CV_32FC1);
        SvmLabels.convertTo(SvmLabels, CvType.CV_32FC1);

        Size testa = SvmData.size();
        Size test = SvmLabels.size();

        CvSVMParams params = new CvSVMParams();
        params.set_svm_type(CvSVM.C_SVC);
        params.set_kernel_type(CvSVM.LINEAR);
        params.set_degree(0);
        params.set_gamma(1);
        params.set_coef0(0);
        params.set_C(1);
        params.set_nu(0);
        params.set_p(0);
        TermCriteria tc = new TermCriteria(1, 1000, 0.01);
        params.set_term_crit(tc);

        CvSVM svmClassifier = new CvSVM(SvmData, SvmLabels, new Mat(), new Mat(), params);

        for (int i = 1; i < 6; i++) {
            Mat cropImg = Highgui.imread("cropped_images\\cropped" + i + ".jpg");
            Size testb = cropImg.size();
            cropImg = cropImg.reshape(1, 1);

            cropImg.convertTo(cropImg, CvType.CV_32FC1);

            int response = (int) svmClassifier.predict(cropImg);
            if (response == 1) {
                System.out.println("Znaleziono blachę! Numer: " + i);
            }

        }
        */
        
        File folder = new File("cropped_images");
        File[] listOfFiles = folder.listFiles();
        
        for (File f : listOfFiles) {
            
        }
        
        SVMTrainCreator stc = new SVMTrainCreator();
        stc.loadTrainingData();
        

    }

    /**
     * @return the buffImage
     */
    public BufferedImage getBuffImage() {
        return buffImage;
    }

    /**
     * @param buffImage the buffImage to set
     */
    public void setBuffImage(BufferedImage buffImage) {
        this.buffImage = buffImage;
    }

    /**
     * @return the filePath
     */
    public static String getFilePath() {
        return filePath;
    }

    /**
     * @param aFilePath the filePath to set
     */
    public static void setFilePath(String aFilePath) {
        filePath = aFilePath;
    }

}
