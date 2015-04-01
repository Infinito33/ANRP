package pl.biai.logic;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
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
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

/**
 * @author Fufer
 * @version 1.0
 */
public class PhotoEdition {
//testing pull up

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

    /**
     * Constructor.
     */
    public PhotoEdition() {

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

            Mat img_rotated = new Mat();
            Imgproc.warpAffine(photoOriginal, img_rotated, rotmat, photoOriginal.size(), Imgproc.INTER_CUBIC);

            Size rect_size = rect.size;
            if (r < 1) {
                double temp;
                temp = rect_size.width;
                rect_size.width = rect_size.height;
                rect_size.height = temp;
            }
            Mat img_crop = new Mat();
            //Wycinanka prostokąta.
            Imgproc.getRectSubPix(img_rotated, rect_size, rect.center, img_crop);
            Highgui.imwrite("cropped_images\\cropped" + rectCount + ".jpg", img_crop);
        }
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
