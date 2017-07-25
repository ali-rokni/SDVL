/**
 * author: Seyed Ali Rokni
 * 
 */
package edu.wsu.eecs.epsl;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Utils {

	
	public static NumberFormat formatter = new DecimalFormat("#00.00");

	public static double avg(List<Double> l) {
		return sum(l) / l.size();
	}

	public static String avgS(List<Double> l) {
		return formatter.format(100 * avg(l));
	}

	public static double sum(List<Double> l) {
		double total = 0;
		for (double d : l) {
			total += d;
		}
		return total;
	}

	public static String sumS(List<Double> l) {
		return formatter.format(sum(l));
	}

	public static Map<Integer, SemiLabel> computeSemiLabels(double[][] confusionMatrix) {
		Map<Integer, SemiLabel> result = new HashMap<>();
		for (int i = 0; i < confusionMatrix.length; i++) {
			result.put(i, new SemiLabel(i, confusionMatrix.length));
			for (int j = 0; j < confusionMatrix.length; j++) {
				result.get(i).possibles[j] = (int) confusionMatrix[j][i];
				result.get(i).sum += (int) confusionMatrix[j][i];
			}
		}

		return result;
	}

	public static void addColumnTitleToCsv(String filename) throws IOException {
		Scanner sc = new Scanner(new File(filename));
		String firstLine = sc.nextLine();
		String[] items = firstLine.split(",");
		FileWriter fw = new FileWriter(new File(filename.substring(0, filename.length() - 4) + "WithTitle.csv"));
		// String[] titles = new String[items.length];
		int i = 0;
		for (; i < items.length - 1; i++) {
			fw.write("a" + i + ", ");
		}
		fw.write("a" + i + "\n" + firstLine);
		while (sc.hasNextLine()) {
			fw.write("\n" + sc.nextLine());
		}
		fw.close();
		sc.close();
	}


	public static void printArray(int[] ar) {
		for (int i = 0; i < ar.length; i++) {
			System.out.print(ar[i] + " ");
		}
		System.out.println();
	}
	public static String matrixToString(double[][] m) {
		StringBuffer bf = new StringBuffer();
		for (int i = 0; i < m.length - 1; i++) {
			for (int j = 0; j < m.length - 1; j++) {
				bf.append((int) m[i][j] + "\t");
			}
			bf.append("\n");
		}
		return bf.toString();
	}
	public static String setToString(Set<Integer> set){
		String s = "";
		boolean first = true;
		for(int i: set){
			if(first){
				s += i;
				first = false;
			}else{
				s += "_" + i;
			}
		}
		return s;
	}

	public static void printArray(Object[] a) {
		for (int i = 0; i < a.length; i++) {
			System.out.print(a[i] + " ");
		}
		System.out.println();
	}


	public static Pair<Instances, Pair<List<Integer>, List<Integer>>> preProcessWithClustersAndClasses(
			Instances instances) throws Exception {
		instances.setClassIndex(instances.numAttributes() - 1);
		List<Integer> classesLeft = new ArrayList<Integer>();
		List<Integer> classesRight = new ArrayList<Integer>();
		for (int i = 0; i < instances.numInstances(); i++) {
			classesLeft.add((int) instances.instance(i).value(instances.numAttributes() - 2));
			classesRight.add((int) instances.instance(i).classValue());
		}

		Remove rm = new Remove();
		Integer mote = (Integer) (instances.numAttributes()) - 1;
		rm.setAttributeIndices((mote).toString() + "-last");
		rm.setInputFormat(instances);
		instances = Filter.useFilter(instances, rm);
		instances.setClassIndex(-1);
		return Pair.of(instances, Pair.of(classesLeft, classesRight));

	}
}
