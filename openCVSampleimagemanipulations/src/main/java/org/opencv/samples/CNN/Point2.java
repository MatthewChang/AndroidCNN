package org.opencv.samples.CNN;

/**
 * Created by Matthew on 6/15/2015.
 */
public class Point2 {
    public int x;
    public int y;
    public Point2(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public Point2 add(Point2 o) {
        return new Point2(this.x+o.x,this.y+o.y);
    }

    public Point2 add(int x, int y) {
        return new Point2(this.x+x,this.y+y);
    }

    public String toString() {
        return "["+x+","+y+"]";
    }
}
