package pl.biai.main;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.core.CvType;
import org.opencv.imgproc.Imgproc;
import pl.biai.gui.Gui;

/**
 * Automatic System of Plate Recognition
 * jazda jazda
 * @author Fufer
 */
public class ASPR {

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        Gui gui = new Gui();
        gui.createGui();
        /*
         VideoCapture camera = new VideoCapture(0);
    	
         if(!camera.isOpened()){
         System.out.println("Error");
         }
         else {
         Mat frame = new Mat();
         while(true){
         if (camera.read(frame)){
         System.out.println("Frame Obtained");
         System.out.println("Captured Frame Width " + 
         frame.width() + " Height " + frame.height());
         Highgui.imwrite("camera.jpg", frame);
         System.out.println("OK");
         break;
         }
         }	
         }
         camera.release();
         */

        Mat photo = new Mat();
        photo = Highgui.imread("src/resources/test1.jpg");

        photo.reshape(20);
        //photo.inv(Core.DECOMP_LU);
        Size size = new Size(0.0, 0.0);
        //Imgproc.blur(photo, photo, size);
        Mat greyScalePhoto = new Mat();
        //test
        Imgproc.cvtColor(photo, greyScalePhoto, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(photo, greyScalePhoto, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(photo, photo, size, 20.0);

        Mat sobelPhoto = new Mat();
        Imgproc.Sobel(greyScalePhoto, sobelPhoto, CvType.CV_8U, 1, 0, 3, 1, 0);

        
        
        Mat thresholdPhoto = new Mat();
        Imgproc.threshold(sobelPhoto, thresholdPhoto, 0, 255, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY);

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(17, 3));
        Imgproc.morphologyEx(thresholdPhoto, thresholdPhoto, Imgproc.MORPH_CLOSE, element);

        List<MatOfPoint> contoursList = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresholdPhoto, contoursList, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        List<RotatedRect> rects = new ArrayList<>();
        /*
        for (MatOfPoint mop : contoursList) {
            MatOfPoint2f mop2f = new MatOfPoint2f(mop.toArray());
            RotatedRect rr = Imgproc.minAreaRect(mop2f);
            
            if(!verifySizes(rr)) {
                //contoursList.remove(mop);
            } else {
                rects.add(rr);
            }
        }*/

        //imshow.showImage(thresholdPhoto);
    }

    

}
