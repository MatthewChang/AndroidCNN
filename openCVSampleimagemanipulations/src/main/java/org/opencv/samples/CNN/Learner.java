package org.opencv.samples.CNN;

/**
 * Created by Matthew on 6/9/2015.
 */
public class Learner {
    public int ox;
    public int oy;
    public int thresh;
    public Learner(int ox,int oy, int thresh) {
        this.ox = ox;
        this.oy = oy;
        this.thresh = thresh;

        //catch learners that need to be wrapped
        while(this.ox<=-128) {
            this.ox+=128;
        }
        while(this.oy<=-128) {
            this.oy+=128;
        }
    }

    public String toString() {
        return "["+ox +" " + oy + " " + thresh +"]";
    }
}
