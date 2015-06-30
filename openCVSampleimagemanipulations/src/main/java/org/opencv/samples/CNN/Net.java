package org.opencv.samples.CNN;

import java.util.ArrayList;

/**
 * Created by Matthew on 6/29/2015.
 */
public class Net {
    ArrayList<Layer> layers = new ArrayList<Layer>();

    public Net() {
        layers = new ArrayList<Layer>();
    }

    public void addLayer(Layer l) {
        layers.add(l);
    }

    public String toString() {
        String ret = "";
        for(int i = 0; i < layers.size(); i++) {
            ret += layers.get(i) + "\n";
        }
        return ret;
    }
}
