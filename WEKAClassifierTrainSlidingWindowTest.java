package moa.classifiers.meta;

import java.io.FileNotFoundException;
import java.io.FileReader;

import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instances;

import junit.framework.TestCase;

public class WEKAClassifierTrainSlidingWindowTest extends TestCase {

	WEKAClassifierTrainSlidingWindow classifier, classifier10;
	
	private Instances dataset;
	private String wcDataSetFileName = "test_wc_data_no_timestamp.arff";
	private int effIndex = 2; // note that this class index starts with 0

	
	public WEKAClassifierTrainSlidingWindowTest() {
	}

	public WEKAClassifierTrainSlidingWindowTest(String name) {
		super(name);
	}
	
	protected void setUp() throws Exception {
		super.setUp();

		// classifier to always train with all instances seen so far
		classifier = new WEKAClassifierTrainSlidingWindow();
		classifier.baseLearnerOption.setValueViaCLIString("weka.classifiers.lazy.IBk -K 1");
		classifier.widthOption.setValueViaCLIString("100000");
		classifier.resetLearning();

		// classifier to always train with only 3 instances
		classifier10 = new WEKAClassifierTrainSlidingWindow();
		classifier10.baseLearnerOption.setValueViaCLIString("weka.classifiers.lazy.IBk -K 1");
		classifier10.widthOption.setValueViaCLIString("3");
		classifier10.resetLearning();
		
		
		FileReader fr=null;
		try {
			fr = new FileReader(wcDataSetFileName);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		dataset = new Instances(fr,6,effIndex+1);
		

	}
	
	protected void tearDown() throws Exception {
		super.tearDown();
	}
	
	public void testTrainOnInstance() throws Exception {
		
		double [] attvals = {1,1,2};
		DenseInstance inst = new DenseInstance(1, attvals);
		inst.setDataset(dataset);
		weka.core.Instance winst = classifier.instanceConverter.wekaInstance(inst);
		
		double [] attvals2 = {2,2,3};
		DenseInstance inst2 = new DenseInstance(1, attvals2);
		inst2.setDataset(dataset);
		weka.core.Instance winst2 = classifier.instanceConverter.wekaInstance(inst2);
		
		double [] attvals3 = {3,3,4};
		DenseInstance inst3 = new DenseInstance(1, attvals3);
		inst3.setDataset(dataset);
		weka.core.Instance winst3 = classifier.instanceConverter.wekaInstance(inst3);
		
		double [] attvals4 = {4,4,5};
		DenseInstance inst4 = new DenseInstance(1, attvals4);
		inst4.setDataset(dataset);
		weka.core.Instance winst4 = classifier.instanceConverter.wekaInstance(inst4);
		
		double [] attvals5 = {5,5,6};
		DenseInstance inst5 = new DenseInstance(1, attvals5);
		inst5.setDataset(dataset);
		weka.core.Instance winst5 = classifier.instanceConverter.wekaInstance(inst5);
		
		classifier.trainOnInstance(inst);
		assertTrue(classifier.trainingHasStarted());
		assertEquals(1.0, classifier.trainingWeightSeenByModel());
		assertEquals(2.0, classifier.classifier.distributionForInstance(winst)[0]);
		
		classifier.trainOnInstance(inst2);
		assertEquals(2.0, classifier.trainingWeightSeenByModel());
		assertEquals(2.0, classifier.classifier.distributionForInstance(winst)[0]);
		assertEquals(3.0, classifier.classifier.distributionForInstance(winst2)[0]);
		
		classifier.trainOnInstance(inst3);
		assertEquals(3.0, classifier.trainingWeightSeenByModel());
		assertEquals(2.0, classifier.classifier.distributionForInstance(winst)[0]);
		assertEquals(3.0, classifier.classifier.distributionForInstance(winst2)[0]);
		assertEquals(4.0, classifier.classifier.distributionForInstance(winst3)[0]);
		
		classifier.trainOnInstance(inst4);
		assertEquals(4.0, classifier.trainingWeightSeenByModel());
		assertEquals(2.0, classifier.classifier.distributionForInstance(winst)[0]);
		assertEquals(3.0, classifier.classifier.distributionForInstance(winst2)[0]);
		assertEquals(4.0, classifier.classifier.distributionForInstance(winst3)[0]);
		assertEquals(5.0, classifier.classifier.distributionForInstance(winst4)[0]);
		
		classifier.trainOnInstance(inst5);
		assertEquals(5.0, classifier.trainingWeightSeenByModel());
		assertEquals(2.0, classifier.classifier.distributionForInstance(winst)[0]);
		assertEquals(3.0, classifier.classifier.distributionForInstance(winst2)[0]);
		assertEquals(4.0, classifier.classifier.distributionForInstance(winst3)[0]);
		assertEquals(5.0, classifier.classifier.distributionForInstance(winst4)[0]);
		assertEquals(6.0, classifier.classifier.distributionForInstance(winst5)[0]);
		
		
		classifier10.trainOnInstance(inst);
		assertTrue(classifier10.trainingHasStarted());
		assertEquals(1.0, classifier10.trainingWeightSeenByModel());
		assertEquals(2.0, classifier10.classifier.distributionForInstance(winst)[0]);
		
		classifier10.trainOnInstance(inst2);
		assertEquals(2.0, classifier10.trainingWeightSeenByModel());
		assertEquals(2.0, classifier10.classifier.distributionForInstance(winst)[0]);
		assertEquals(3.0, classifier10.classifier.distributionForInstance(winst2)[0]);
		
		classifier10.trainOnInstance(inst3);
		assertEquals(3.0, classifier10.trainingWeightSeenByModel());
		assertEquals(2.0, classifier10.classifier.distributionForInstance(winst)[0]);
		assertEquals(3.0, classifier10.classifier.distributionForInstance(winst2)[0]);
		assertEquals(4.0, classifier10.classifier.distributionForInstance(winst3)[0]);
		
		classifier10.trainOnInstance(inst4);
		assertEquals(4.0, classifier10.trainingWeightSeenByModel());
		assertEquals(3.0, classifier10.classifier.distributionForInstance(winst)[0]);
		assertEquals(3.0, classifier10.classifier.distributionForInstance(winst2)[0]);
		assertEquals(4.0, classifier10.classifier.distributionForInstance(winst3)[0]);
		assertEquals(5.0, classifier10.classifier.distributionForInstance(winst4)[0]);
		
		classifier10.trainOnInstance(inst5);
		assertEquals(5.0, classifier10.trainingWeightSeenByModel());
		assertEquals(4.0, classifier10.classifier.distributionForInstance(winst)[0]);
		assertEquals(4.0, classifier10.classifier.distributionForInstance(winst2)[0]);
		assertEquals(4.0, classifier10.classifier.distributionForInstance(winst3)[0]);
		assertEquals(5.0, classifier10.classifier.distributionForInstance(winst4)[0]);
		assertEquals(6.0, classifier10.classifier.distributionForInstance(winst5)[0]);
		
	}

	
	public void testGetVotesForInstance() {
		
		double [] attvals = {1,1,2};
		DenseInstance inst = new DenseInstance(1, attvals);
		inst.setDataset(dataset);
		
		double [] attvals2 = {2,2,3};
		DenseInstance inst2 = new DenseInstance(1, attvals2);
		inst2.setDataset(dataset);
		
		double [] attvals3 = {3,3,4};
		DenseInstance inst3 = new DenseInstance(1, attvals3);
		inst3.setDataset(dataset);
		
		double [] attvals4 = {4,4,5};
		DenseInstance inst4 = new DenseInstance(1, attvals4);
		inst4.setDataset(dataset);
		
		double [] attvals5 = {5,5,6};
		DenseInstance inst5 = new DenseInstance(1, attvals5);
		inst5.setDataset(dataset);
			
		
		classifier.trainOnInstance(inst);
		classifier.trainOnInstance(inst2);
		classifier.trainOnInstance(inst3);
		classifier.trainOnInstance(inst4);
		classifier.trainOnInstance(inst5);

		
		assertEquals(5.0, classifier.trainingWeightSeenByModel());

		
		assertEquals(2.0, classifier.getVotesForInstance(inst)[0]);
		assertEquals(3.0, classifier.getVotesForInstance(inst2)[0]);
		assertEquals(4.0, classifier.getVotesForInstance(inst3)[0]);
		assertEquals(5.0, classifier.getVotesForInstance(inst4)[0]);
		assertEquals(6.0, classifier.getVotesForInstance(inst5)[0]);
		
		
		classifier10.trainOnInstance(inst);
		classifier10.trainOnInstance(inst2);
		classifier10.trainOnInstance(inst3);
		classifier10.trainOnInstance(inst4);
		classifier10.trainOnInstance(inst5);
		
		
		assertEquals(5.0, classifier10.trainingWeightSeenByModel());

		
		assertEquals(4.0, classifier10.getVotesForInstance(inst)[0]);
		assertEquals(4.0, classifier10.getVotesForInstance(inst2)[0]);
		assertEquals(4.0, classifier10.getVotesForInstance(inst3)[0]);
		assertEquals(5.0, classifier10.getVotesForInstance(inst4)[0]);
		assertEquals(6.0, classifier10.getVotesForInstance(inst5)[0]);
	}
}
