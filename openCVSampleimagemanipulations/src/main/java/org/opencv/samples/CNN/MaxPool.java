package org.opencv.samples.CNN;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

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
        ArrayList<Mat> ret = new ArrayList<Mat>();
        for(int i = 0; i < in.size();i++) {
            ret.add(new Mat());
        }
        for(int i = 0; i < in.size();i++) {
            Imgproc.dilate(in.get(i),in.get(i),kernel);
            Mat temp = in.get(i).submat(off,in.get(i).rows()-off,off,in.get(i).cols()-off);
            Imgproc.resize(temp,ret.get(i),new Size(0,0),1.0/stride,1.0/stride,Imgproc.INTER_NEAREST);
        }
        return ret;
    }

    public String toString() {
        return "[MaxPool "+kernel.rows()+"x"+kernel.cols() + " " + pad + " " + stride + "]";
    }
}
