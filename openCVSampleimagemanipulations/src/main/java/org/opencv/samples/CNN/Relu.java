package org.opencv.samples.CNN;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

/**
 * Created by Matthew on 6/29/2015.
 */
public class Relu implements Layer {

    public Relu() {
    }
    @Override
    public ArrayList<Mat> evaluate(ArrayList<Mat> in) {
        for(int i = 0; i < in.size();i++) {
            Core.max(in.get(i),new Scalar(0),in.get(i));
        }
        return in;
    }

    public String toString() {
        return "[Relu]";
    }
}
