package org.opencv.samples.CNN;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.ArrayList;

/**
 * Created by Matthew on 6/29/2015.
 */
public class Net implements Layer {
    ArrayList<Layer> layers = new ArrayList<Layer>();
    Mat imageMean;

    public Net() {
        layers = new ArrayList<Layer>();
    }

    public void addLayer(Layer l) {
        layers.add(l);
    }
    public void setMean(Mat mean) {
        imageMean = mean;
    }
    public String toString() {
        String ret = "";
        for(int i = 0; i < imageMean.cols();i++) {
            ret+= imageMean.get(0,i)[0] + " ";
        }
        ret += "\n";
        for(int i = 0; i < layers.size(); i++) {
            ret += layers.get(i) + "\n";
        }
        return ret;
    }

    public ArrayList<Mat> evaluate(ArrayList<Mat> in) {
        ArrayList<Mat> temp = new ArrayList<Mat>();
        for(int i = 0; i < in.size(); i++) {
            temp.add(new Mat());
            Core.subtract(in.get(i), new Scalar(imageMean.get(0, i)[0]), temp.get(i));
        }
        for(int i = 0; i < layers.size(); i++) {
            temp = layers.get(i).evaluate(temp);
        }
        return temp;
    }
}
