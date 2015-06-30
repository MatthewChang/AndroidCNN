package org.opencv.samples.CNN;

import org.opencv.core.Mat;

import java.util.ArrayList;

/**
 * Created by Matthew on 6/29/2015.
 */
public interface Layer {
    ArrayList<Mat> evaluate(ArrayList<Mat> in);
}
