/**
 * author: Seyed Ali Rokni
 * 
 */
package edu.wsu.eecs.epsl;

public class SemiLabel {
		public int predicted;
		public int[] possibles;
		public int sum;

		public SemiLabel(int predicted, int size) {
			super();
			this.predicted = predicted;
			this.possibles = new int[size];
		}
		@Override
		public String toString() {
			StringBuffer bf = new StringBuffer("(" + predicted + "->[");
			for(int i = 0; i < possibles.length; i++){
				bf.append(possibles[i] + ", ");
			}
			bf.append("])");
			return bf.toString();
		}
	}