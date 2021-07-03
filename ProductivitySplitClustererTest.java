package moa.clusterers.threshold;

import com.yahoo.labs.samoa.instances.Instance;

import junit.framework.TestCase;
import moa.cluster.Clustering;
import moa.cluster.ProductivityThresholdCluster;
import moa.streams.ArffFileStream;

public class ProductivitySplitClustererTest extends TestCase {

	public ProductivitySplitClustererTest() {
		super();
	}

	public ProductivitySplitClustererTest(String name) {
		super(name);
	}
	
	private ArffFileStream dataStream;
	private String dataSetFileName = "test_clusterer_no_timestamp.arff";
	private int effIndex = 2; // note that the class index starts with 0
	private int sizeIndex = 1;

	
	protected void setUp() throws Exception {
		super.setUp();
		dataStream = new ArffFileStream(dataSetFileName, effIndex);
	}
	
	protected void tearDown() throws Exception {
		super.tearDown();
	}
    
	
	public void testGetInclusionProbability() {
		ProductivitySplitClusterer clusterer = new ProductivitySplitClusterer();
		clusterer.prodThresholdsOption.setValue("1.0;2.0");
		clusterer.effortAttIndexOption.setValue(effIndex);
		clusterer.sizeAttIndexOption.setValue(sizeIndex);
		clusterer.resetLearningImpl();
		
		assertEquals(2, clusterer.getNumberProductivityThresholds());
		assertEquals(1.0, clusterer.getProductivityThreshold(0));
		assertEquals(2.0, clusterer.getProductivityThreshold(1));
		
		Clustering clusters = clusterer.getClusteringResult();
		
		assertEquals(3, clusters.getClustering().size());
		assertEquals(Double.MIN_VALUE, ((ProductivityThresholdCluster) clusters.get(0)).getLeftThreshold());
		assertEquals(1.0, ((ProductivityThresholdCluster) clusters.get(0)).getRightThreshold());
		assertEquals(1.0, ((ProductivityThresholdCluster) clusters.get(1)).getLeftThreshold());
		assertEquals(2.0, ((ProductivityThresholdCluster) clusters.get(1)).getRightThreshold());
		assertEquals(2.0, ((ProductivityThresholdCluster) clusters.get(2)).getLeftThreshold());
		assertEquals(Double.MAX_VALUE, ((ProductivityThresholdCluster) clusters.get(2)).getRightThreshold());
		
		Instance inst = dataStream.nextInstance().instance; // 2,0,1
		assertEquals(1.0, inst.value(effIndex));
		assertEquals(0.0, inst.value(sizeIndex));
		
		assertEquals(0.0, clusters.get(0).getInclusionProbability(inst));
		assertEquals(0.0, clusters.get(1).getInclusionProbability(inst));
		assertEquals(1.0, clusters.get(2).getInclusionProbability(inst));
		
		inst = dataStream.nextInstance().instance; // 4,6,5
		assertEquals(5.0, inst.value(effIndex));
		assertEquals(6.0, inst.value(sizeIndex));
		
		assertEquals(1.0, clusters.get(0).getInclusionProbability(inst));
		assertEquals(0.0, clusters.get(1).getInclusionProbability(inst));
		assertEquals(0.0, clusters.get(2).getInclusionProbability(inst));
		
		inst = dataStream.nextInstance().instance; // 4,5,6
		assertEquals(6.0, inst.value(effIndex));
		assertEquals(5.0, inst.value(sizeIndex));
		
		assertEquals(0.0, clusters.get(0).getInclusionProbability(inst));
		assertEquals(1.0, clusters.get(1).getInclusionProbability(inst));
		assertEquals(0.0, clusters.get(2).getInclusionProbability(inst));
		
		inst = dataStream.nextInstance().instance; // 4,5,15
		assertEquals(15.0, inst.value(effIndex));
		assertEquals(5.0, inst.value(sizeIndex));
		
		assertEquals(0.0, clusters.get(0).getInclusionProbability(inst));
		assertEquals(0.0, clusters.get(1).getInclusionProbability(inst));
		assertEquals(1.0, clusters.get(2).getInclusionProbability(inst));
		
		inst = dataStream.nextInstance().instance; // 1,1,1
		assertEquals(1.0, inst.value(effIndex));
		assertEquals(1.0, inst.value(sizeIndex));
		
		assertEquals(1.0, clusters.get(0).getInclusionProbability(inst));
		assertEquals(0.0, clusters.get(1).getInclusionProbability(inst));
		assertEquals(0.0, clusters.get(2).getInclusionProbability(inst));
		
		
		
	}

}
