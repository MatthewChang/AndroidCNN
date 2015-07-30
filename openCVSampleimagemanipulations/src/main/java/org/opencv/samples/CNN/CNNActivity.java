package org.opencv.samples.CNN;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
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
import android.content.res.AssetFileDescriptor;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;

public class CNNActivity extends Activity implements CvCameraViewListener2, View.OnTouchListener, SensorEventListener{
    private static final String TAG = "~~~";

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
    private ArrayList<Mat> input;
    private Mat filter;
    private Boolean tracking = false;
    private double buffer[][] = new double[12][];
    private int buffer_pos = 0;
    MediaPlayer mp;
    private int current_state = 0;

    private byte bg_color[] = {0,0,0,0};
    private byte open_color[] = {127,127,0,0};
    private byte closed_color[] = {0,127,0,0};
    private byte gun_color[] = {0,0,127,0};
    private byte other_color[] = {127,0,0,0};
    private byte colors[][] = {bg_color,open_color,gun_color,closed_color,other_color};
    private String label_sound_files[] = {"open.mp3","closed.mp3","gun.mp3","nohand.mp3"};

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

        center = new Point2(38 / 2, 38 / 2);
        net = new Net();
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = new double[4];
        }

        mp = new MediaPlayer();
        /*Mat A = new Mat(11,11,CvType.CV_32F);
        loadMatFromFile("layer1s1s1",11,11,A);
        Log.i(TAG,MatToString(A));*/
    }

    private Conv loadConv(String filters, String biases_name) {
        ArrayList<ArrayList<Mat>> layer = loadMat4FromFile(filters);
        Mat biases = loadMat2FromFile(biases_name);
        return new Conv(layer,biases);
    }


    /*private Conv loadConv(String namebase,int kr, int kc, int t, int f) {
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
    }*/

    private MaxPool loadMaxPool(String filename) {
        Mat pool = loadMat2FromFile(filename);
        return new MaxPool((int)pool.get(0,0)[0],(int)pool.get(0,2)[0],(int)pool.get(0,3)[0]);
    }

    private void init() {
        //Log.i("~~~",MatToString(loadMat2FromFile("input")));
        net = loadNetwork();
        input = loadMat3FromFile("input");
        //input = loadMatFromFile("input",38,38);
        //filter = loadMatFromFile("layer1s1s1",11,11);
        Log.i("~~~",""+net);
        initialized = true;
    }

    private Net loadNetwork() {
        int resID = getResources().getIdentifier("config", "raw", getPackageName());
        InputStream is = getResources().openRawResource(resID);
        Scanner config = new Scanner(is).useDelimiter(" |\\n");
        Net net = new Net();
        while(config.hasNext()) {
            String type = config.next();
            if(type.equals("mean")) {
                Mat mean = loadMat2FromFile(config.next());
                Log.i("~~~",""+mean);
                net.setMean(mean);
            } else if(type.equals("conv")) {
                net.addLayer(loadConv(config.next(),config.next()));
            } else if(type.equals("pool")) {
                net.addLayer(loadMaxPool(config.next()));
            } else if(type.equals("relu")) {
                net.addLayer(new Relu());
            } else {
                Log.e("~~~","Invalid type: "+type);
                return null;
            }
        }
        return net;
    }

    private void addBuffer(double[] in) {
        buffer[buffer_pos++] = in;
        buffer_pos = buffer_pos%buffer.length;
    }

    private double[] buffer_average() {
        double counts[] = {0,0,0,0};
        for(int i = 0; i < buffer.length; i++) {
            for(int c = 0; c < counts.length; c++) {
                counts[c] += buffer[i][c];
            }
        }
        for(int c = 0; c < counts.length; c++) {
            counts[c] /= buffer.length;
        }
        return counts;
    }

    /*private int buffer_label() {
        double counts[] = buffer_average();
        int max_pos = 0;
        for(int i = 1; i < counts.length; i++) {
            if(counts[i] > counts[max_pos])
                max_pos = i;
        }
        return max_pos;
    }*/

    private int buffer_label() {
        double counts[] = buffer_average();
        for(int i = 0; i < counts.length; i++) {
            if(counts[i] > 0.7) {
                return i;
            }
        }
        return -1;
    }

    public Mat loadMat2FromFile(String name) {
        int resID = getResources().getIdentifier(name, "raw", getPackageName());
        InputStream is = getResources().openRawResource(resID);
        Scanner scan = new Scanner(is).useDelimiter(",| |\\n");
        int dims = Integer.parseInt(scan.next());
        if(dims != 2) {
            Log.e("~~~","Dimention mismatch");
            return null;
        }
        int rows = Integer.parseInt(scan.next());
        int cols = Integer.parseInt(scan.next());
        //Log.i("~~~name",""+rows+ " " + cols);
        Mat res = new Mat(rows, cols, CvType.CV_32F);
        float data[] = new float[rows * cols];
        int el = 0;
        while (scan.hasNext()) { //Column major order
            String s= scan.next();
            //Log.i("~~~name",""+s);
            data[el++] = Float.parseFloat(s);
        }
        if(el != rows*cols) {
            Log.e("~~~","Size Mismatch");
        }
        res.put(0, 0, data);

        return res;
    }

    public ArrayList<Mat> loadMat3FromFile(String name) {
        int resID = getResources().getIdentifier(name, "raw", getPackageName());
        InputStream is = getResources().openRawResource(resID);
        Scanner scan = new Scanner(is).useDelimiter(",| |\\n");
        int dims = Integer.parseInt(scan.next());
        if(dims != 3) {
            Log.e("~~~","Dimention mismatch");
            return null;
        }
        int rows = Integer.parseInt(scan.next());
        int cols = Integer.parseInt(scan.next());
        int d3 = Integer.parseInt(scan.next());
        ArrayList<Mat> layer = new ArrayList<Mat>();
        for (int i = 0; i < d3; i++) {
            Mat res = new Mat(rows, cols, CvType.CV_32F);
            float data[] = new float[rows * cols];
            for(int el = 0; el < rows*cols; el++){ //Column major order
                if(!scan.hasNext()) {
                    Log.e(TAG,"Size Mismatch");
                }
                data[el] = Float.parseFloat(scan.next());
            }
            res.put(0, 0, data);
            layer.add(res);
        }

        if(scan.hasNext()) {
            Log.e(TAG,"Size Mismatch");
        }
        return layer;
    }

    public ArrayList<ArrayList<Mat>> loadMat4FromFile(String name) {
        int resID = getResources().getIdentifier(name, "raw", getPackageName());
        InputStream is = getResources().openRawResource(resID);
        Scanner scan = new Scanner(is).useDelimiter(",| |\\n");
        int dims = Integer.parseInt(scan.next());
        if(dims != 4) {
            Log.e("~~~","Dimention mismatch");
            return null;
        }
        int rows = Integer.parseInt(scan.next());
        int cols = Integer.parseInt(scan.next());
        int d3 = Integer.parseInt(scan.next());
        int filters = Integer.parseInt(scan.next());
        ArrayList<ArrayList<Mat>> layer = new ArrayList<ArrayList<Mat>>();
        for (int f = 0; f < filters; f++) {
            ArrayList<Mat> filter = new ArrayList<Mat>();
            for (int i = 0; i < d3; i++) {
                Mat res = new Mat(rows, cols, CvType.CV_32F);
                float data[] = new float[rows * cols];
                for(int el = 0; el < rows*cols; el++){ //Column major order
                    if(!scan.hasNext()) {
                        Log.e(TAG,"Size Mismatch");
                    }
                    data[el] = Float.parseFloat(scan.next());
                }
                res.put(0, 0, data);
                filter.add(res);
            }
            layer.add(filter);
        }
        if(scan.hasNext()) {
            Log.e(TAG,"Size Mismatch");
        }
        return layer;
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

        if(mp.isPlaying())
        {
            mp.stop();
        }
        return false;
    }

    @Override
    public void onAccuracyChanged(Sensor s,int accuracy) {

    }

    @Override
    public void onSensorChanged(SensorEvent s) {

    }

    public void playFile(String filename) {
        try {
            mp.reset();
            AssetFileDescriptor afd;
            afd = getAssets().openFd(filename);
            mp.setDataSource(afd.getFileDescriptor(),afd.getStartOffset(),afd.getLength());
            mp.prepare();
            mp.start();
        } catch (IllegalStateException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
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

        Core.sqrt(distance, distance);
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

    public void imgDist(Mat LAB,Point2 loc, Mat dest) {

    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        if(!initialized) {
            init();
        }
        Mat rgba = inputFrame.rgba();
        Core.flip(rgba, rgba, 0);
        Size sizeRgba = rgba.size();
        int width = 42;
        int thumbSize = width*4;

        Imgproc.resize(rgba, mIntermediateMat, new Size(width, width));
        Mat thumb = rgba.submat(0, thumbSize, 0, thumbSize);
        Imgproc.cvtColor(mIntermediateMat, mIntermediateMat, Imgproc.COLOR_RGB2Lab);
        Mat dist_lab = distanceFrom(mIntermediateMat,new Point2(width/2-1,width/2-1));

        ArrayList<Mat> ch = new ArrayList<Mat>();
        Core.split(mIntermediateMat,ch);
        ch.remove(0);
        Core.merge(ch,mIntermediateMat);
        Mat dist_ab = distanceFrom(mIntermediateMat,new Point2(width/2-1,width/2-1));
        Core.addWeighted(dist_lab, 1, dist_ab, 5, 0, feedbackMat);
        //feedbackMat.copyTo(input);
        feedbackMat.convertTo(feedbackMat, CvType.CV_8U);
        Imgproc.cvtColor(feedbackMat, feedbackMat, Imgproc.COLOR_GRAY2RGBA);
        Imgproc.resize(feedbackMat, thumb, new Size(thumbSize, thumbSize), 0, 0, Imgproc.INTER_NEAREST);

        ArrayList<Mat> in = new ArrayList<Mat>();
        //Core.flip(input,input,1);
        //in.add(input);

        /*Log.i(TAG,MatToString(input.get(0)));
        Log.i(TAG,MatToString(input.get(1)));
        Log.i(TAG,MatToString(input.get(2)));*/
        in = net.evaluate(input);
        Log.i(TAG, "" + in.get(0).get(0, 0)[0] + " " + in.get(1).get(0, 0)[0] + " " + in.get(2).get(0, 0)[0] + " " + in.get(3).get(0, 0)[0]);

        /*int max_pos = 0;
        double exp_sum = 0;
        for(int i = 0; i < in.size(); i++) {
            exp_sum += Math.exp(in.get(i).get(0, 0)[0]);
            if(in.get(i).get(0, 0)[0] > in.get(max_pos).get(0, 0)[0])
                max_pos = i;
        }
        double vals[] = {0,0,0,0};
        for(int i = 0; i < in.size(); i++) {
            vals[i] = Math.exp(in.get(i).get(0, 0)[0])/exp_sum;
        }
        addBuffer(vals);

        double res[] = buffer_average();
        String s = String.format("%.3f, %.3f, %.3f, %.3f", res[0], res[1], res[2], res[3]);
        Core.putText(rgba, s, new Point(thumbSize, thumbSize + 20), 0, 1, new Scalar(255, 0, 0, 0), 5);
        int new_label = buffer_label();
        if(new_label != current_state && new_label >= 0) {
            playFile(label_sound_files[new_label]);
            current_state = new_label;
        }
        switch(buffer_label()) {
            case 0:
                Core.putText(rgba,"Open",new Point(thumbSize,thumbSize),0,2,new Scalar(255,0,0,0),5);
                break;
            case 1:
                Core.putText(rgba,"Closed",new Point(thumbSize,thumbSize),0,2,new Scalar(255,0,0,0),5);
                break;
            case 2:
                Core.putText(rgba,"Gun",new Point(thumbSize,thumbSize),0,2,new Scalar(255,0,0,0),5);
                break;
            case 3:
                Core.putText(rgba,"No Hand",new Point(thumbSize,thumbSize),0,2,new Scalar(255,0,0,0),5);
                break;
            case -1:
                Core.putText(rgba,"Unsure...",new Point(thumbSize,thumbSize),0,2,new Scalar(255,0,0,0),5);
                break;

        }
        if(toggle) {
            toggle = !toggle;
            Log.i("~~","WRITING");
            File file = new File(Environment.getExternalStorageDirectory(), "data/RRR.data");
            mat2csvFile(ch.get(0), file);
            file = new File(Environment.getExternalStorageDirectory(), "data/GGG.data");
            mat2csvFile(ch.get(1), file);
            file = new File(Environment.getExternalStorageDirectory(), "data/BBB.data");
            mat2csvFile(ch.get(2), file);
        }*/

        Core.flip(rgba, rgba, -1);
        return rgba;
    }
}