package org.opencv.samples.CNN;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Scalar;

import java.util.ArrayList;

/**
 * Created by Matthew on 6/29/2015.
 */
public class MaxPool implements Layer {
    private Mat kernel;
    private int pad;
    private int stride;
    private int off;
    public MaxPool(int k,int s,int p) {
        kernel = Mat.ones(new Size(k,k), CvType.CV_8U);
        pad = p;
        stride = s;
        off = (kernel.rows()-1)-pad;
    }
    @Override
    public ArrayList<Mat> evaluate(ArrayList<Mat> in) {
        Mat temp;
        for(int i = 0; i < in.size();i++) {
            double min = Core.minMaxLoc(in.get(i)).minVal;
            temp = Mat.ones(in.get(i).rows() + 2 * pad, in.get(i).cols() + 2 * pad, in.get(i).type());
            Core.multiply(temp,new Scalar(min),temp);
            in.get(i).copyTo(temp.submat(pad, temp.rows() - pad, pad, temp.cols() - pad));
            Imgproc.dilate(temp, temp, kernel);
            temp = temp.submat(off,temp.rows()-off,off,temp.cols()-off);
            Imgproc.resize(temp,in.get(i),new Size(0,0),1.0/stride,1.0/stride,Imgproc.INTER_NEAREST);
        }
        return in;
    }

    public String toString() {
        return "[MaxPool "+kernel.rows()+"x"+kernel.cols() + " " + pad + " " + stride + "]";
    }
}
