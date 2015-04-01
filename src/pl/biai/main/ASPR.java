package pl.biai.main;

import org.opencv.core.*;
import pl.biai.gui.Gui;

/**
 * Automatic System of Plate Recognition
 *
 * @author Fufer
 */
public class ASPR {

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        //Loads native library - openCV won't work without this.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Gui gui = new Gui();
        gui.createGui();

    }

}
