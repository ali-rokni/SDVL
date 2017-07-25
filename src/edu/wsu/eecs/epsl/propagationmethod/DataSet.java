/**
 * author: Seyed Ali Rokni
 * 
 */
package edu.wsu.eecs.epsl.propagationmethod;


import java.util.Arrays;
import java.util.List;

public enum DataSet {
	ETH(Arrays.asList(1, 2, 4), 0.25, 1, 5), SENIOR(Arrays.asList(3, 5, 6, 7, 8, 9, 16, 18, 19), 0.25, 4, 22), SAD(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8), 0.25, 4, 15);

	public List<Integer> subjects;
	double alpha;
	int k;
	int numOfActivities;
	DataSet(List<Integer> subs, double alpha, int k, int n) {
		this.subjects = subs;
		this.alpha = alpha;
		this.k = k;
		this.numOfActivities = n;
	}
};