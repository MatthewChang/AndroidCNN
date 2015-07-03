package org.opencv.samples.CNN;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;
import org.opencv.samples.imagemanipulations.R;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;

public class CNNActivity extends Activity implements CvCameraViewListener2, View.OnTouchListener{
    private static final String TAG = "OCVSample::Activity";

   private CameraBridgeViewBase mOpenCvCameraView;


    private Mat mIntermediateMat;
    private Mat mIntermediateMat2;
    private Mat feedbackMat;
    private TreeNode tree;
    private Boolean toggle = false;
    private List<TreeNode> treeset;
    private Point2 center;
    private Mat previousLAB;
    private Boolean initialized = false;
    private Net net;
    private Mat input;
    private Mat filter;
    private Boolean tracking = false;
    private int buffer[] = {3,3,3,3,3,3,3,3,3,3};
    private int buffer_pos = 0;

    private byte bg_color[] = {0,0,0,0};
    private byte open_color[] = {127,127,0,0};
    private byte closed_color[] = {0,127,0,0};
    private byte gun_color[] = {0,0,127,0};
    private byte other_color[] = {127,0,0,0};
    private byte colors[][] = {bg_color,open_color,gun_color,closed_color,other_color};

    private Point point_add(Point o,Point t) {
        return new Point(o.x+t.x,o.y+t.y);
    }
    private Point point_sub(Point o,Point t) {
        return new Point(o.x-t.x,o.y-t.y);
    }
    private Point point_mul(Point o,double x) {
        return new Point(o.x*x,o.y*x);
    }
    private double point_dot(Point o, Point t) {return o.x*t.x + o.y*t.y;}
    private double point_length(Point o) {return Math.sqrt(point_dot(o, o));}
    private Point point_normalize(Point o) {return point_mul(o, 1 / point_length(o));}
    private Point point_projection(Point v, Point onto) {return point_mul(point_normalize(onto), point_dot(v, point_normalize(onto)));}
    private double dot(double lista[],double listb[]) {
        double val = 0;
        for(int i = 0; i < lista.length;i++) {
            val += lista[i]*listb[i];
        }
        return val;
    }
    private double len(double lista[]) {
        return Math.sqrt(dot(lista, lista));
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(CNNActivity.this);
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public CNNActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.image_manipulations_surface_view);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.image_manipulations_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

        center = new Point2(38/2,38/2);
        net = new Net();
        /*Mat A = new Mat(11,11,CvType.CV_32F);
        loadMatFromFile("layer1s1s1",11,11,A);
        Log.i(TAG,MatToString(A));*/
    }

    private Conv loadConv(String namebase,int kr, int kc, int t, int f) {
        ArrayList<ArrayList<Mat>> layer = new ArrayList<ArrayList<Mat>>();
        for(int l = 0; l < f; l++) {
            layer.add(new ArrayList<Mat>());
            for(int i = 0;i < t; i++) {
                String filename = namebase+"s"+(l+1)+"s"+(i+1);
                layer.get(l).add(loadMatFromFile(filename,kr,kc));
            }
        }
        Mat biases = loadMatFromFile(namebase+"b",1,f);
        return new Conv(layer,biases);
    }

    private MaxPool loadMaxPool(String filename) {
        Mat pool = loadMatFromFile(filename,1,4);
        return new MaxPool((int)pool.get(0,0)[0],(int)pool.get(0,2)[0],(int)pool.get(0,3)[0]);
    }

    private void init() {
        net.addLayer(loadConv("layer1", 11, 11, 1, 30));
        net.addLayer(new Relu());
        net.addLayer(loadMaxPool("layer3"));
        net.addLayer(loadConv("layer4", 7, 7, 30, 30));
        net.addLayer(new Relu());
        net.addLayer(loadConv("layer6", 1, 1, 30, 3));
        net.setMean(loadMatFromFile("config", 1, 1));
        input = new Mat();
        //input = loadMatFromFile("input",38,38);
        //filter = loadMatFromFile("layer1s1s1",11,11);
        Log.i(TAG,""+net);
        initialized = true;
    }

    private void addBuffer(int type) {
        buffer[buffer_pos++] = type;
        buffer_pos = buffer_pos%buffer.length;
    }

    private int buffer_mode() {
        int counts[] = {0,0,0,0};
        for(int i = 0; i < buffer.length; i++) {
            counts[buffer[i]]++;
        }
        int m = 0;
        for(int i = 1; i < counts.length; i++) {
            if(counts[i] > counts[m])
                m = i;
        }
        return m;
    }

    public Mat loadMatFromFile(String name,int row, int col) {
        int resID = getResources().getIdentifier(name, "raw", getPackageName());
        InputStream is = getResources().openRawResource(resID);
        Scanner scan = new Scanner(is).useDelimiter(",|\\n");
        Mat ret = new Mat(row,col,CvType.CV_32F);
        float data[] = new float[row*col];
        int i = 0;
        while(scan.hasNext()) {
            data[i++] = Float.parseFloat(scan.next());
        }
        ret.put(0,0,data);
        return ret;
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        Log.i(TAG, "onTouch event");
        toggle = !toggle;
        return false;
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mIntermediateMat = new Mat();
        mIntermediateMat2 = new Mat();
        previousLAB = new Mat();
        feedbackMat = Mat.zeros(38,38,CvType.CV_32F);
    }

    public void onCameraViewStopped() {
        // Explicitly deallocate Mats
        if (mIntermediateMat != null)
            mIntermediateMat.release();

        mIntermediateMat = null;

        if (mIntermediateMat2 != null)
            mIntermediateMat2.release();

        mIntermediateMat2 = null;
    }

    public String MatToString(Mat A) {
        String s = "";
        for(int r = 0; r < A.rows();r++) {
            for(int c= 0; c < A.cols();c++) {
                s += String.format("%.4f ",A.get(r,c)[0]);
            }
            s += "\n";
        }
        return s;
    }

    public String ArrayToString(double A[]) {
        String s = "[";
        for(int r = 0; r < A.length;r++) {
            s += A[r] + ", ";
        }
        s += "]";
        return s;
    }

    public String ArrayToString(int A[]) {
        String s = "[";
        for(int r = 0; r < A.length;r++) {
            s += A[r] + ", ";
        }
        s += "]";
        return s;
    }

    public boolean isExternalStorageWritable() {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state)) {
            return true;
        }
        return false;
    }

    public void mat2csvFile(Mat A,File file) {
        try {
            OutputStream outputStream = new FileOutputStream(file);

            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.cols(); c++) {
                    //Log.i(TAG,o);
                    outputStream.write(("" + (int)A.get(r, c)[0]).getBytes());
                    if(c != A.cols()-1) {
                        outputStream.write((",").getBytes());
                    }
                }
                if(r != A.rows()-1) {
                    outputStream.write(("\n").getBytes());
                }
            }
            outputStream.close();
            Log.i(TAG, "SUCCESS");
        } catch (Exception e) {
            Log.i(TAG,"FAILURE");
            e.printStackTrace();
        }
    }


    public Mat distanceFrom2(Mat A, Point2 kernel) {
        Mat distance = Mat.zeros(A.rows(), A.cols(), CvType.CV_32F);
        A.convertTo(A, CvType.CV_32F);
        int step1 = A.channels();
        int step2 = A.cols() * step1;
        float data[] = new float[A.rows() * A.cols() * A.channels()];
        A.get(0, 0, data);

        int cand[][] = {{0, 0},
                {-5, 0},
                {5, 0},
                {0, -5},
                {0, 5}};
        //int cand[][] = {{0, 0},{1,0}};
        List<Mat> channels = new ArrayList<Mat>();
        Core.split(A, channels);
        Mat ch = new Mat();

        for (int i = 0; i < cand.length; i++) {
            Mat distance_temp = Mat.zeros(A.rows(), A.cols(), CvType.CV_32F);
            for (int c = 0; c < A.channels(); c++) {
                float mean = data[(kernel.x + cand[i][0]) * step2 + (kernel.y + cand[i][1]) * step1 + c];
                Core.subtract(channels.get(c), new Scalar(mean), ch);
                Core.multiply(ch, ch, ch);
                Core.add(ch, distance_temp, distance_temp);
            }
            Core.sqrt(distance_temp, distance_temp);
            Core.addWeighted(distance_temp, 1.0 / cand.length, distance, 1, 0, distance);
        }
        ch.release();
        return distance;
    }

    public Mat distanceFrom(Mat A, Point2 kernel) {
        Mat distance = Mat.zeros(A.rows(), A.cols(), CvType.CV_32F);
        /*short distance_data[] = new short[A.rows() * A.cols()];
        distance.get(0, 0, distance_data);*/

        A.convertTo(A, CvType.CV_32F);
        int step1 = A.channels();
        int step2 = A.cols() * step1;
        float data[] = new float[A.rows() * A.cols() * A.channels()];

        A.get(0, 0, data);
        //Log.i(TAG,""+A.depth()+" "+A.step1());
        int mean[] = new int[A.channels()];
        /*int cand[][] = {{-2, 0},
                {-1, 1},
                {0, 2},
                {1, 1},
                {2, 0},
                {1, -1},
                {0, -2},
                {-1, -1},

                {-1, 0},
                {1, 0},
                {0, -1},
                {0, 1},
                {0, 0}};*/
        int cand[][] = {{0, 0}};

        for (int l = 0; l < A.channels(); l++) {
            for (int i = 0; i < cand.length; i++) {
                mean[l] += data[(kernel.x + cand[i][0]) * step2 + (kernel.y + cand[i][1]) * step1 + l];
            }
            mean[l] /= cand.length;
        }
        List<Mat> channels = new ArrayList<Mat>();
        Core.split(A, channels);
        for (int i = 0; i < A.channels(); i++){
            Core.subtract(channels.get(i), new Scalar(mean[i]), channels.get(i));
            Core.multiply(channels.get(i), channels.get(i), channels.get(i));
            Core.add(channels.get(i), distance, distance);
        }

        Core.sqrt(distance,distance);
        return distance;
        /*
        for(int r = 0; r < A.rows(); r++) {
            for(int c = 0; c < A.cols(); c++) {
                int dist = 0;
                for(int l = 0; l < A.channels(); l++) {
                    int v = data[r*step2 + c*step1 + l] - mean[l];
                    dist += v*v;
                }
                dist = (int)Math.sqrt(dist);
                distance_data[r * A.cols() + c] = (short)dist;
            }
        }
        distance.put(0,0,distance_data);

        return distance;*/
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        if(!initialized) {
            init();
        }
        Mat rgba = inputFrame.rgba();
        Core.flip(rgba, rgba, 0);

        Size sizeRgba = rgba.size();
        int width = 38;
        int thumbSize = width*4;
        Imgproc.resize(rgba, mIntermediateMat, new Size(width, width));
        Mat thumb = rgba.submat(0, thumbSize, 0, thumbSize);
        Imgproc.cvtColor(mIntermediateMat, mIntermediateMat, Imgproc.COLOR_RGB2Lab);
        ArrayList<Mat> ch = new ArrayList<Mat>();
        Core.split(mIntermediateMat,ch);
        ch.remove(0);
        Mat dist_lab = distanceFrom(mIntermediateMat,new Point2(width/2-1,width/2-1));
        Core.merge(ch,mIntermediateMat);
        Mat dist_ab = distanceFrom(mIntermediateMat,new Point2(width/2-1,width/2-1));
        Core.addWeighted(dist_lab, 1, dist_ab, 5, 0, feedbackMat);
        feedbackMat.copyTo(input);
        feedbackMat.convertTo(feedbackMat, CvType.CV_8U);
        Imgproc.cvtColor(feedbackMat, feedbackMat, Imgproc.COLOR_GRAY2RGBA);
        Imgproc.resize(feedbackMat, thumb, new Size(thumbSize, thumbSize), 0, 0, Imgproc.INTER_NEAREST);


        ArrayList<Mat> in = new ArrayList<Mat>();
        Core.flip(input, input, 1);
        in.add(input);
        in = net.evaluate(in);
        //Log.i(TAG, "" + in.get(0).get(0, 0)[0] + " " + in.get(1).get(0, 0)[0] + " " + in.get(2).get(0, 0)[0] + " " + in.get(3).get(0, 0)[0]);
        int max_pos = 0;
        for(int i = 1; i < in.size(); i++) {
            if(in.get(i).get(0, 0)[0] > in.get(max_pos).get(0, 0)[0])
                max_pos = i;
        }
        addBuffer(max_pos);

        Point c = new Point();
        c.x = rgba.rows()*center.y/width;
        c.y = rgba.cols()*center.x/width;
        Core.putText(rgba,""+max_pos,new Point(thumbSize,thumbSize),0,2,new Scalar(255,0,0,0),5);
        Core.circle(rgba,c,10,new Scalar(0,255,0,0),10);
        if(toggle) {
            toggle = !toggle;
            /*Log.i("~~","WRITING");
            File file = new File(Environment.getExternalStorageDirectory(), "data/test_input.data");
            mat2csvFile(input, file);*/
            center.x = width/2;
            center.y = width/2;
        }
        /*int off = 2;
        int stride = 4;
        Mat kernel = Mat.ones(5, 5, CvType.CV_8U);
        int pad = 2;

        double min = Core.minMaxLoc(in.get(0)).minVal;
        Mat temp = Mat.ones(in.get(0).rows() + 2 * pad, in.get(0).cols() + 2 * pad, in.get(0).type());
        Core.multiply(temp,new Scalar(min),temp);
        in.get(0).copyTo(temp.submat(pad, temp.rows() - pad, pad, temp.cols() - pad));
        Log.i(TAG, "" + MatToString(temp));
        Imgproc.dilate(temp, temp, kernel);
        Log.i(TAG, "" + MatToString(temp));
        temp = temp.submat(off,temp.rows()-off,off,temp.cols()-off);
        Log.i(TAG, "" + temp);
        int size = (int)Math.ceil(1.0*temp.rows()/stride);
        Imgproc.resize(temp,in.get(0),new Size(size,size),0,0,Imgproc.INTER_NEAREST);*/

                //mIntermediateMat = mIntermediateMat.submat(5,mIntermediateMat.rows()-5,5,mIntermediateMat.cols()-5);
        //Log.i(TAG,MatToString(mIntermediateMat));


        /*Imgproc.resize(rgba, mIntermediateMat, new Size(width, width));
        Mat filter = new Mat(new Size(11,11),CvType.CV_32F);
        Mat filter2 = Mat.ones(new Size(3,3),CvType.CV_8U);
        Mat dump = new Mat();
        Core.randu(filter, 0, 10);
        for (int i = 0; i < 150; i++) {
            Imgproc.filter2D(mIntermediateMat,dump,-1,filter);
        }
        Mat test = new Mat(new Size(10,10),CvType.CV_8U);
        Core.randu(test, 0, 100);
        Log.i(TAG, MatToString(test));
        Imgproc.dilate(test, test, filter2, new Point(1, 1), 1);
        test = test.submat(1,test.rows()-1,1,test.cols()-1);
        Log.i(TAG, MatToString(test));
        Imgproc.resize(test,test,new Size(0,0),0.25,0.25,Imgproc.INTER_NEAREST);
        //Imgproc.resize(test,test,new Size(3,3),0,0,Imgproc.INTER_MAX);
        Log.i(TAG,MatToString(test));*/
        //Mat test = loadMatFromFile("layer1s1s1",11,11);
        //Log.i(TAG,MatToString(test));


        Core.flip(rgba, rgba, -1);
        return rgba;
    }
}