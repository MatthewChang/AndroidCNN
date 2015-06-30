package org.opencv.samples.CNN;

/**
 * Created by Matthew on 6/9/2015.
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import android.util.Log;

import org.opencv.core.Mat;

public class TreeNode {
    public TreeNode true_node;
    public TreeNode false_node;
    public Learner learner;
    public int label;

    public TreeNode() {
        true_node = null;
        false_node = null;
        learner = null;
        label = -1;
    }

    public TreeNode(int label) {
        true_node = null;
        false_node = null;
        learner = null;
        this.label = label;
    }

    public TreeNode(TreeNode f, TreeNode t, Learner l) {
        true_node = t;
        false_node = f;
        learner = l;
        label = -1;
    }

    public static TreeNode loadTreeFromScanner(Scanner scanner) {
       int ox = Integer.parseInt(scanner.next());
        //Log.i("SOBELOUTPUT",""+ox);
        int oy = Integer.parseInt(scanner.next());
        //Log.i("SOBELOUTPUT",""+oy);
        int thresh = Integer.parseInt(scanner.next());
        //Log.i("SOBELOUTPUT",""+ox+" "+oy+" "+thresh);
        if(thresh < 0) {
            return new TreeNode(ox);
        }
        TreeNode f = loadTreeFromScanner(scanner);
        TreeNode t = loadTreeFromScanner(scanner);
        return new TreeNode(f,t,new Learner(ox,oy,thresh));
    }

    public void logSerialize() {
        if(this.label >= 0) {
            Log.i("LOG",""+this.label + " -1 -1");
        } else {
            Log.i("LOG",""+learner);
            this.false_node.logSerialize();
            this.true_node.logSerialize();
        }
    }

    public static TreeNode loadTreeFromFile(String filename) throws FileNotFoundException{
        Scanner scanner = new Scanner(new File(filename));
        return loadTreeFromScanner(scanner);
    }

    public int height() {
        if(this.true_node == null) {
            return 1;
        }
        return 1 + Math.max(this.true_node.height(),this.false_node.height());
    }

    public int evaluatePixel(Mat frame,int x, int y,int width) {
        if(label >= 0) {
            return label;
        } else {
            int nx = (x+learner.ox+width)%width;
            int ny = (y+learner.oy+width)%width;
            /*String out = "";
            for(int i = 0; i < frame.get(nx, ny).length;i++)
                out +=frame.get(nx, ny)[i];
            Log.i("LOG", out);*/
            if(frame.get(nx,ny)[0] >= learner.thresh) {
                return true_node.evaluatePixel(frame,x,y,width);
            } else {
                return false_node.evaluatePixel(frame,x,y,width);
            }

        }
    }

    public int evaluatePixelSerial(Mat frame,int x, int y,int width) {
        TreeNode node = this;
        while(node.label < 0) {
            node = (frame.get((x+node.learner.ox+width)%width,(y+node.learner.oy+width)%width)[0] >= node.learner.thresh) ? node.true_node : node.false_node;
        }
        return node.label;

    }

    public int evaluatePixelSerialData(byte data[],int x, int y,int width) {
        TreeNode node = this;
        while(node.label < 0) {
            int nx = (x+node.learner.ox+width)%width;
            int ny = (y+node.learner.oy+width)%width;
            //Log.i("DEBUG",""+data[nx*width + ny]+" "+node.learner.thresh+" "+(data[nx*width + ny] == node.learner.thresh));
            node = (data[nx*width + ny] == node.learner.thresh) ? node.true_node : node.false_node;
        }
        return node.label;
    }

    /*public int evaluatePixelSerialData(short data[],int x, int y,int width) {
        TreeNode node = this;
        while(node.label < 0) {

            //Log.i("LOG",""+learner.thresh);
            int nx = (x+node.learner.ox+width)%width;
            int ny = (y+node.learner.oy+width)%width;
            //Log.i("LOG",""+node.learner+" "+x+" "+y+" "+nx+" "+ny);
            int diff = Math.abs(data[nx*width + ny] - data[x*width + y]);
            if(diff > 127)
                diff = 255-diff;
            //float d = data[nx*width + ny];
            //Log.i("LOG",""+node.learner+" "+x+" "+y+" "+nx+" "+ny);
            node = (diff >= node.learner.thresh) ? node.true_node : node.false_node;
        }
        //return depth;
        return node.label;

    }*/

    /*public static void mod(Mat A,Scalar d,Mat dest) {
        Mat div = new Mat();
        Core.add(A,d,A);
        Core.divide(A,d,div);
        Core.multiply(div,d,div);
        Core.subtract(A,div,dest);
        div.release();
    }*/

   /*public Mat evaluatePoints(Mat frame,Mat points,int width) {

        if(this.label >= 0) {
            return new Mat(new Size(points.rows(),1), Core.DEPTH_MASK_8S,new Scalar(this.label));
        } else {
            Mat true_points = new Mat();
            Mat false_points = new Mat();
            for(int r = 0; r < points.rows(); r++) {
                if (frame.get((int)points.get(r, 0)[0], (int)points.get(r, 1)[0])[0] >= learner.thresh*255) {
                    true_points.push_back(points.row(r));
                } else {
                    false_points.push_back(points.row(r));
                }

            }
            Mat true_labels = true_node.evaluatePoints(frame,true_points,width);
            Mat false_labels = true_node.evaluatePoints(frame, true_points, width);
            true_labels.push_back(false_labels);
            return true_labels;
        }

    }*/
}
