package org.opencv.samples.CNN;

import android.widget.ImageButton;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

/**
 * Created by Matthew on 6/29/2015.
 */
public class Mat3D {
    public ArrayList<Mat> filters;

    public  Mat3D(ArrayList<Mat> f) {
        filters = f;
    }

    public int rows() {
        if(filters.size() > 0) {
            return filters.get(0).rows();
        }
        return 0;
    }

    public int cols() {
        if(filters.size() > 0) {
            return filters.get(0).cols();
        }
        return 0;
    }

    public Mat filter(Mat3D in) {
        Mat ret = Mat.zeros(in.rows(),in.cols(), CvType.CV_32F);
        Mat temp = new Mat();
        for (int i = 0; i < filters.size(); i++) {
            Imgproc.filter2D(in.filters.get(i),temp,CvType.CV_32F,this.filters.get(i));
            Core.add(ret,temp,ret);
        }
        int off = (this.cols()-1)/2;
        ret.submat(off,ret.rows()-off,off,ret.cols()-off);
        return ret;
    }
}
