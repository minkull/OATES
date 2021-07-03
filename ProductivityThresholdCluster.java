
/**
 * @author Leandro L. Minku, University of Birmingham, l.l.minku@cs.bham.ac.uk
 * 
 * Implements a simple cluster based on pre-defined productivity splits, as the criterion used to split CC data in the following paper: 
 * 
 * MINKU, L. L.; YAO, X.; "How to Make Best Use of Cross-company Data in Software Effort Estimation?", 
 * Proceedings of the 36th International Conference on Software Engineering (ICSE'14), p. 446-456, May 2014, doi: 10.1145/2568225.2568228.
 * 
 * Productivity is defined as effort / size. 
 * 
 * The left threshold is exclusive and the right is inclusive.
 * 
 */


package moa.cluster;

import java.util.Random;

import com.yahoo.labs.samoa.instances.Instance;

public class ProductivityThresholdCluster extends Cluster {

	private static final long serialVersionUID = 1L;

	protected double leftThreshold, rightThreshold;
	protected int effortAttIndex, sizeAttIndex;
	
	protected double weight;
	
	public ProductivityThresholdCluster() {
		super();
	}
	
	public ProductivityThresholdCluster(double leftThreshold, double rightThreshold, int effortAttIndex, int sizeAttIndex) {
		super();
		this.leftThreshold = leftThreshold;
		this.rightThreshold = rightThreshold;
		this.effortAttIndex = effortAttIndex;
		this.sizeAttIndex = sizeAttIndex; 
	}
	
	public double getLeftThreshold() {
		return leftThreshold;
	}
	
	public double getRightThreshold() {
		return rightThreshold;
	}
	
	public int getEffortAttIndex() {
		return effortAttIndex;
	}
	
	public int getSizeAttIndex() {
		return sizeAttIndex;
	}

	@Override
	public double[] getCenter() {
		throw new UnsupportedOperationException("Productivity threshold cluster has no center.");
	}

	@Override
	public double getWeight() {
		return weight;
	}
	
	public void setWeight(double weight) {
		this.weight = weight;
	}

	@Override
	public double getInclusionProbability(Instance instance) {
		
		double prod;
		
		if (instance.value(sizeAttIndex) != 0)
			prod = instance.value(effortAttIndex) / instance.value(sizeAttIndex);
		else prod = Double.MAX_VALUE;
		
		if (prod > leftThreshold && prod <= rightThreshold)
			return 1;
		
		return 0;
	}

	@Override
	public Instance sample(Random random) {
		throw new UnsupportedOperationException("Not supported yet.");
	}

}
