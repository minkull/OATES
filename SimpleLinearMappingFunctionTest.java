package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;


import junit.framework.TestCase;
import junitx.util.PrivateAccessor;
import moa.classifiers.Classifier;
import moa.streams.ArffFileStream;

public class SimpleLinearMappingFunctionTest extends TestCase {

	private SimpleLinearMappingFunction mp;
	private Classifier ccLearner;
	private String ccTrainingSetFileName = "test_cc_data_no_timestamp.arff";
	private String wcTrainingSetFileName = "test_wc_data_no_timestamp.arff";
	private int classIndex = 3; // note that the class index starts with 1 when setting it to the arff file, but later on starts with 0 inside moa
	
	private ArffFileStream wcDataStream;
	
	public SimpleLinearMappingFunctionTest() {
		super();
	}

	public SimpleLinearMappingFunctionTest(String name) {
		super(name);
	}
	
	protected void setUp() throws Exception {
		super.setUp();
		ArffFileStream ccDataStream = new ArffFileStream(ccTrainingSetFileName, classIndex);
		wcDataStream = new ArffFileStream(wcTrainingSetFileName, classIndex);
		
		ccLearner = new moa.classifiers.lazy.kNN();
		((moa.classifiers.lazy.kNN) ccLearner).kOption.setValue(1);
		while (ccDataStream.hasMoreInstances()) {
			Instance inst = ccDataStream.nextInstance().instance;
			ccLearner.trainOnInstance(inst);
		}
		
		mp = new SimpleLinearMappingFunction();
		mp.resetLearning();
		mp.setCCLearner(ccLearner);
	}
	
	protected void tearDown() throws Exception {
        super.tearDown();
    }
	
	
	public void testTrainOnInstanceImpl() {
		assertEquals(1.0, mp.getB());
						
		// cc learner will predict the correct output for the first 3 wc instances, including prediction of 0 and output of 0
		double output = 0.5;
		Instance inst = wcDataStream.nextInstance().instance;
		assertEquals(output,inst.classValue());
		assertEquals(output,ccLearner.getVotesForInstance(inst)[0]);
		mp.trainOnInstanceImpl(inst);	
		assertEquals(1.0, mp.getB());
		assertEquals(2, inst.classIndex());
		
		output = 0.0;
		inst = wcDataStream.nextInstance().instance;
		assertEquals(output,inst.classValue());
		assertEquals(output,ccLearner.getVotesForInstance(inst)[0]);
		mp.trainOnInstanceImpl(inst);	
		assertEquals(1.0, mp.getB());
		assertEquals(2, inst.classIndex());
		
		output = 1.0;
		inst = wcDataStream.nextInstance().instance;
		assertEquals(output,inst.classValue());
		assertEquals(output,ccLearner.getVotesForInstance(inst)[0]);
		mp.trainOnInstanceImpl(inst);	
		assertEquals(1.0, mp.getB());
		assertEquals(2, inst.classIndex());
		
		
		// cc learner will make an error and predict 0.0 instead of 0.1 for this instance, and require b to be updated. However, the prediction is zero, meaning that b remains at 1.
		inst = wcDataStream.nextInstance().instance;
		assertEquals(0.1,inst.classValue());
		assertEquals(0.0,ccLearner.getVotesForInstance(inst)[0]);
		mp.trainOnInstanceImpl(inst);
		assertEquals(1.0, mp.getB());
				
		// cc learner will make an error and predict 0.5 instead of 0.4 for this instance, and require b to be updated
		inst = wcDataStream.nextInstance().instance;
		assertEquals(0.4,inst.classValue());
		assertEquals(0.5,ccLearner.getVotesForInstance(inst)[0]);
		mp.trainOnInstanceImpl(inst);
		assertEquals((1-mp.learningRate.getValue()) * 1 + mp.learningRate.getValue() * 0.4d/0.5d, mp.getB());
		
		// cc learner will make an error and predict 1 instead of 0.8 for this instance, and require b to be updated
		inst = wcDataStream.nextInstance().instance;
		assertEquals(0.8,inst.classValue());
		assertEquals(1.0,ccLearner.getVotesForInstance(inst)[0]);
		mp.trainOnInstanceImpl(inst);
		assertEquals(Math.round(((1-mp.learningRate.getValue()) * 0.98 + mp.learningRate.getValue() * 0.8d/1d) * 1000000), Math.round(mp.getB() * 1000000));
				
		assertEquals(false, wcDataStream.hasMoreInstances());
	}

	public void testGetVotesForInstance() throws NoSuchFieldException {
		wcDataStream.restart();
		PrivateAccessor.setField(mp, "b", 1.0);
		assertEquals(1.0,mp.getB());
		
		// Mapping function gives the correct prediction
		Instance inst = wcDataStream.nextInstance().instance;
		assertEquals(0.5,inst.classValue());
		assertEquals(0.5,mp.getVotesForInstance(inst)[0]);
		
		// Mapping function gives the correct prediction
		inst = wcDataStream.nextInstance().instance;
		PrivateAccessor.setField(mp, "b", 0.9);
		assertEquals(0.9,mp.getB());
		assertEquals(0.0,mp.getVotesForInstance(inst)[0]);
		
		// Mapping function predicts incorrectly		
		inst = wcDataStream.nextInstance().instance;
		assertEquals(1.0,inst.classValue());
		assertEquals(0.9 * 1.0,mp.getVotesForInstance(inst)[0]);
	}
	
}
