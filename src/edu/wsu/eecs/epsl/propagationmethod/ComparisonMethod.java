/**
 * author: Seyed Ali Rokni
 * 
 */
package edu.wsu.eecs.epsl.propagationmethod;


import java.util.ArrayList;
import java.util.List;

import edu.wsu.eecs.epsl.Utils;
import weka.core.Instances;

public class ComparisonMethod {
	Instances train;
	StringBuffer output;
	String name;
	List<Double> accuracies = new ArrayList<>();
	List<Double> percision = new ArrayList<>();
	List<Double> recalls = new ArrayList<>();
	List<Double> fMeasures = new ArrayList<>();
	double[][] confusionMat = new double[20][20];
	public ComparisonMethod(String name) {
		super();
		this.name = name;
		this.output = new StringBuffer(name + ": \t\t");
	}
	
	public String toString(){
		return "Acc: " + Utils.avgS(accuracies) + "\tPer: " + Utils.avgS(percision) + "\tRec: " + Utils.avgS(recalls) + "\tF1: " + Utils.avgS(fMeasures);
	}
	
}
