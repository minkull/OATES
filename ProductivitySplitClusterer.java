
/**
 * @author Leandro L. Minku, University of Birmingham, l.l.minku@cs.bham.ac.uk
 * 
 * Implements a simple cluster algorithm based on pre-defined productivity splits, as the criterion used to split CC data in the following paper: 
 * 
 * MINKU, L. L.; YAO, X.; "How to Make Best Use of Cross-company Data in Software Effort Estimation?", 
 * Proceedings of the 36th International Conference on Software Engineering (ICSE'14), p. 446-456, May 2014, doi: 10.1145/2568225.2568228.
 * 
 * Productivity is defined as effort / size. 
 * 
 * The left threshold is exclusive and the right is inclusive.
 * 
 */

package moa.clusterers.threshold;

import java.util.ArrayList;
import java.util.Collections;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.cluster.Clustering;
import moa.cluster.ProductivityThresholdCluster;
import moa.clusterers.AbstractClusterer;
import moa.core.Measurement;

public class ProductivitySplitClusterer extends AbstractClusterer {

	private static final long serialVersionUID = 1L;
	
	public StringOption prodThresholdsOption = new StringOption("prodThresholds", 't', 
			"Productivity (effort / size) thresholds to split CC data. Enter each threshold separated by semi-collon.", "2.85;6.6"); // the default values are for coc81
	
	public IntOption effortAttIndexOption = new IntOption("effortAttIndex", 'e',
            "Index of the effort attribute to compute productivity (starting from 0).", 16, 0, Integer.MAX_VALUE); // the default values are for coc81

	public IntOption sizeAttIndexOption = new IntOption("sizeAttIndex", 's',
            "Index of the size attribute to compute productivity (starting from 0).", 15, 0, Integer.MAX_VALUE); // the default values are for coc81
	
	protected ArrayList<Double> prodThresholds;
	
	public ProductivitySplitClusterer() {
		super();
		//resetLearningImpl();
	}
	
	@Override
	public void resetLearningImpl() {
		String []prodThresholdsStr = prodThresholdsOption.getValue().split(";");
		prodThresholds = new ArrayList<Double>(prodThresholdsStr.length);
		for (int i=0; i<prodThresholdsStr.length; ++i) {
			prodThresholds.add(Double.parseDouble(prodThresholdsStr[i]));
		}
		Collections.sort(prodThresholds);
	}

	public double getProductivityThreshold(int index) {
		return prodThresholds.get(index);
	}
	
	public int getNumberProductivityThresholds() {
		return prodThresholds.size();
	}
	
	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		throw new UnsupportedOperationException("Not supported yet.");
	}

	@Override
	public Clustering getClusteringResult() {
		
		Clustering clustering = new Clustering();

		if (prodThresholds == null || prodThresholds.size() == 0)
			return clustering;
		
		clustering.add(new ProductivityThresholdCluster(Double.MIN_VALUE, prodThresholds.get(0), effortAttIndexOption.getValue(), sizeAttIndexOption.getValue()));

		for ( int i = 0; i < this.prodThresholds.size()-1; i++ ) {
			clustering.add(new ProductivityThresholdCluster(prodThresholds.get(i), prodThresholds.get(i+1), effortAttIndexOption.getValue(), sizeAttIndexOption.getValue()));
		}
		
		clustering.add(new ProductivityThresholdCluster(prodThresholds.get(this.prodThresholds.size()-1), Double.MAX_VALUE, effortAttIndexOption.getValue(), sizeAttIndexOption.getValue()));
		
		return clustering;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return new Measurement[]{new Measurement("number of clusters",
                this.prodThresholds != null ? this.prodThresholds.size(): 0)};
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		throw new UnsupportedOperationException("Not supported yet.");
	}

}
