/**
 * author: Seyed Ali Rokni
 * 
 */
package edu.wsu.eecs.epsl.propagationmethod;

import java.util.ArrayList;
import java.util.List;



public class SemiGraphNode {
	public static List<SemiGraphNode> train = null;
	static double alpha = 0.5;
	public double[] currentWeights = null;
	public double[] futureWeights = null;;
	public List<Integer> neighbors = new ArrayList<Integer>();
	
	
	public static void propagate(int k){
		for (int c = 0; c < 5; c++) {
			for (int i = 0; i < train.size(); i++) {
				train.get(i).update();
			}
			for (int i = 0; i < train.size(); i++) {
				train.get(i).putFutureInCurrent();
			}
		}
	}
	public SemiGraphNode(int[] possibles, List<Integer> neighs){
		
		currentWeights = new double[possibles.length];
		futureWeights = new double[possibles.length];
		int sum = 0;
		for( int i = 0; i < possibles.length; i++){
			sum += possibles[i];
		}
		for( int i = 0; i < possibles.length; i++){
			currentWeights[i] = (double) possibles[i] / sum;
		}
		neighbors.addAll(neighs);
//		this.train = train;
	}
	
	public void computNeighbors(int k){
		//create a map for int (index) to its neighbors : in the experiments class
	}
	
	public void update(){
		for(int i =0; i < currentWeights.length; i++){
			futureWeights[i] = alpha * currentWeights[i] + (1 - alpha) * getNeighborAverage(i);
		}
	}
	
	public void putFutureInCurrent(){
		for(int i = 0; i < futureWeights.length; i++){
			currentWeights[i] =  futureWeights[i];
		}
	}
	
	public int getClassValue(){
		double max = 0;
		int index = -1;
		for(int i =0; i < currentWeights.length; i++){
			if(currentWeights[i] > max){
				max = currentWeights[i];
				index = i;
			}
		}
		return index;
	}
	
	private double getNeighborAverage(int i){
		double total = 0;
		for(int neighborIndex: neighbors){
			total += train.get(neighborIndex).currentWeights[i];
		}
		return total / neighbors.size();
	}
	
	public double getError(){
		double total = 0;
		for(Integer n: neighbors){
			for(int i =0; i < currentWeights.length; i++){
				total += Math.pow(currentWeights[i] - SemiGraphNode.train.get(n).currentWeights[i], 2);
			}
		}
		return total;
	}
}
