package org.opencv.samples.CNN;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

/**
 * Created by Matthew on 6/29/2015.
 */
public class Conv implements Layer {
    private ArrayList<ArrayList<Mat>> filters;
    private Mat biases;

    public Conv(ArrayList<ArrayList<Mat>> f,Mat b) {
        filters = f;
        biases = b;
    }

    public Mat filter(ArrayList<Mat> in,ArrayList<Mat> filter) {
        Mat ret = Mat.zeros(in.get(0).rows(),in.get(0).cols(), CvType.CV_32F);
        Mat temp = new Mat();
        for (int i = 0; i < filter.size(); i++) {
            /*Log.i("~~",""+filter.get(i).rows()+" " + filter.get(i).cols());
            Log.i("~~",""+in.get(i).rows()+" " + in.get(i).cols());*/
            Imgproc.filter2D(in.get(i), temp, CvType.CV_32F, filter.get(i));
            Core.add(ret, temp, ret);
        }
        int offr = (filter.get(0).rows()-1)/2;
        int offc = (filter.get(0).cols()-1)/2;
        return ret.submat(offr,ret.rows()-offr,offc,ret.cols()-offc);
    }

    public ArrayList<Mat> evaluate(ArrayList<Mat> in) {
        ArrayList<Mat> ret = new ArrayList<Mat>();
        for (int i = 0; i < filters.size(); i++) {
            ret.add(filter(in,filters.get(i)));
            Core.add(ret.get(i),new Scalar(biases.get(0,i)[0]),ret.get(i));
        }

        return ret;
    }

    public String toString() {
        return "[" + filters.get(0).get(0).rows() + " "+ filters.get(0).get(0).cols() + " "+ filters.get(0).size()+ " "+ filters.size() + "]";
    }
}