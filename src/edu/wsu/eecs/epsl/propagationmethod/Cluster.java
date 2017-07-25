/**
 * author: Seyed Ali Rokni
 * 
 */
package edu.wsu.eecs.epsl.propagationmethod;
import java.util.Set;


public class Cluster {
	public int id;
	public Set<Integer> members;
	public Cluster(int id, Set<Integer> members) {
		super();
		this.id = id;
		this.members = members;
	}
	
	
}
