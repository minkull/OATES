

/*
 * Example of command line:
 * EvaluatePrequentialRegression -l (meta.OATES -c (threshold.ProductivitySplitClusterer -t 2.85;6.6) -d (/Users/llm11/Leandro's Files/Work/Approaches/MOA-2016.04-leandro/promise_nominal_edited0_sorted_timestamp.arff) -q 500) -s (ArffFileStream -f (/Users/llm11/Leandro's Files/Work/Approaches/MOA-2016.04-leandro/cocomonasa_nominal_edited_timestamp.arff) -c 17) -f 1 -o (/Users/llm11/Leandro's Files/Work/Approaches/MOA-2016.04-leandro/tmp.pred) -O (/Users/llm11/Leandro's Files/Work/Approaches/MOA-2016.04-leandro/tmp2.pred)
 */

package moa.classifiers.meta;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.AttributesInformation;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceInformation;
import com.yahoo.labs.samoa.instances.Instances;

import junit.framework.TestCase;
import junitx.util.PrivateAccessor;
import moa.cluster.Clustering;
import moa.clusterers.streamkm.StreamKM;
import moa.streams.ArffFileStream;

public class OATESTest extends TestCase {
	
	public OATESTest() {
		super();
	}

	public OATESTest(String name) {
		super(name);
	}
	
	
	private ArffFileStream wcDataStream;
	private String wcDataSetFileName = "test_wc_data.arff";
	private String ccDataSetFileName = "test_cc_data.arff";
	private String ccDataSetFileNameEmpty = "test_cc_data_empty.arff";
	private int effIndex = 2; // note that this class index starts with 0
	private int sizeIndex = 1;
	private Instances dataset;
	
	private OATES oates, oates2; //dycom uses ProductivitySplitClusterer, and dycom2 uses StreamKM
	private OATES oatesNoCC; // same as oates, but without any cc data stream
	
	protected void setUp() throws Exception {
		super.setUp();
		wcDataStream = new ArffFileStream(wcDataSetFileName, effIndex);
		
		oates = new OATES();
		oates.baseLearnerOption.setValueViaCLIString("moa.classifiers.lazy.kNN -k 1");
		oates.mappingFunctionOption.setValueViaCLIString("moa.classifiers.meta.SimpleLinearMappingFunction -r 0.1");
		oates.clustererOption.setValueViaCLIString("moa.clusterers.threshold.ProductivitySplitClusterer -t \"1.0;2.0\" -e " + effIndex + " -s " + sizeIndex);
		oates.ccDataStreamFile.setValueViaCLIString(ccDataSetFileName);
		oates.betaWC.setValue(0.6);
		oates.betaCC.setValue(0.5);
		oates.period.setValue(1);
		oates.wcPastInstancesQueueSize.setValue(3);
		oates.prepareForUse();
		oates.resetLearning();
		
		oates2 = new OATES();
		oates2.baseLearnerOption.setValueViaCLIString("moa.classifiers.lazy.kNN -k 1");
		oates2.mappingFunctionOption.setValueViaCLIString("moa.classifiers.meta.SimpleLinearMappingFunction -r 0.1");
		oates2.clustererOption.setValueViaCLIString("moa.clusterers.clustream.WithKmeans -h 10 -k 3 -m 10");
		oates2.ccDataStreamFile.setValueViaCLIString(ccDataSetFileName);
		oates2.betaWC.setValue(0.5);
		oates2.betaCC.setValue(0.5);
		oates2.period.setValue(1);
		oates2.wcPastInstancesQueueSize.setValue(3);
		oates2.prepareForUse();
		oates2.resetLearning();
		
		oatesNoCC = new OATES();
		oatesNoCC.baseLearnerOption.setValueViaCLIString("moa.classifiers.lazy.kNN -k 1");
		oatesNoCC.mappingFunctionOption.setValueViaCLIString("moa.classifiers.meta.SimpleLinearMappingFunction -r 0.1");
		oatesNoCC.clustererOption.setValueViaCLIString("moa.clusterers.threshold.ProductivitySplitClusterer -t \"1.0;2.0\" -e " + effIndex + " -s " + sizeIndex);
		oatesNoCC.ccDataStreamFile.setValueViaCLIString(ccDataSetFileNameEmpty);
		oatesNoCC.betaWC.setValue(0.6);
		oatesNoCC.betaCC.setValue(0.5);
		oatesNoCC.period.setValue(1);
		oatesNoCC.wcPastInstancesQueueSize.setValue(3);
		oatesNoCC.prepareForUse();
		oatesNoCC.resetLearning();
		
		FileReader fr=null;
		try {
			fr = new FileReader(wcDataSetFileName);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		dataset = new Instances(fr,6,effIndex+1); // WARNING!!! The "Instances" class is initialised with class attribute starting from 1, and later on in the code the class attribute of each instance is actually starting from 0
															// It seems that the class ArffLoader is to blame for that.
		
	}
	
	protected void tearDown() throws Exception {
		super.tearDown();
	}
	
	public void testResetLearning() {
		
		assertTrue(oates.mappingFunctions != null);
		assertTrue(oates.weights != null);
		assertTrue(oates.learners != null);
		assertTrue(oates.wcPastInstancesQueue != null);
		assertTrue(oates.ccPastInstancesQueue != null);
		
		// When using ProductivitySplitClusterer, the clusters will be initialized before training starts
		assertEquals(3,oates.mappingFunctions.size());
		assertEquals(4,oates.learners.size());
		assertEquals(4,oates.weights.size());
		assertEquals(0,oates.ccPastInstancesQueue.size());
		assertEquals(0,oates.wcPastInstancesQueue.size());
		
		assertFalse(oates.isCCClassIndexSet);
		
		for (int i=0; i<3; ++i) {
			assertEquals(oates.learners.get(i),((MappingFunction) oates.mappingFunctions.get(i)).getCCLearner());
		}
		

		assertTrue(oates2.clusterer.getClusteringResult() != null);
		assertEquals(3,((moa.clusterers.clustream.WithKmeans) oates2.clusterer).kOption.getValue());
		assertEquals(0,oates2.clusterer.getClusteringResult().size());
		
		assertTrue(oates2.mappingFunctions != null);
		assertTrue(oates2.weights != null);
		assertTrue(oates2.learners != null);
		assertTrue(oates2.wcPastInstancesQueue != null);
		assertTrue(oates2.ccPastInstancesQueue != null);
		
		// When using StremKM, the clusters will not be initialized before training starts
		assertEquals(0,oates2.mappingFunctions.size());
		assertEquals(1,oates2.learners.size());
		assertEquals(1,oates2.weights.size());
		assertEquals(0,oates2.ccPastInstancesQueue.size());
		assertEquals(0,oates2.wcPastInstancesQueue.size());
		
		assertFalse(oates2.isCCClassIndexSet);	
		
		assertFalse(oatesNoCC.ccDataStream.hasMoreInstances());
		assertFalse(oatesNoCC.learners.get(0).trainingHasStarted());
		assertFalse(oatesNoCC.learners.get(1).trainingHasStarted());
		assertFalse(oatesNoCC.learners.get(2).trainingHasStarted());
		
	}
	
	
	public void testIndexClusterInstanceBelongsTo() {
		
		Clustering clusters = oates.clusterer.getClusteringResult();
		
		double [] atts = {1,2,3};
		DenseInstance ccInst = new DenseInstance(1, atts);
		assertEquals(1,oates.indexClusterInstanceBelongsTo(clusters, ccInst));
		
		double []atts2 = {1,3,2};
		ccInst = new DenseInstance(1, atts2);
		assertEquals(0,oates.indexClusterInstanceBelongsTo(clusters, ccInst));

		double []atts3 = {1,3,10};
		ccInst = new DenseInstance(1, atts3);
		assertEquals(2,oates.indexClusterInstanceBelongsTo(clusters, ccInst));
	}
	
	public void testTrainOnCCInstance() {
		
		// The first tests are with dycom rather than dycom2. This means that the trainOnCCInstance method will not result in a new cluster being created
		double [] attvals = {1,2,3,4};
		DenseInstance ccInst = new DenseInstance(1, attvals);
		ccInst.setDataset(dataset);
		
		oates.trainOnCCInstance(ccInst);
		
		assertEquals(3,oates.mappingFunctions.size());
		assertEquals(4,oates.learners.size());
		assertEquals(4,oates.weights.size());
		
		ccInst.deleteAttributeAt(3);
				// Delete attribute from dataset, as some base learners will make use of the information from the dataset.
				List<Attribute> atts = null;
				List<Integer> indexes = null;
				AttributesInformation attinfo = null;
				try {
					InstanceInformation instanceinfo = (InstanceInformation) PrivateAccessor.getField(dataset, "instanceInformation");
					attinfo = (AttributesInformation) PrivateAccessor.getField(instanceinfo, "attributesInformation");
					atts = (List<Attribute>) PrivateAccessor.getField(attinfo, "attributes");
					indexes = (List<Integer>) PrivateAccessor.getField(attinfo, "indexValues");
				} catch (NoSuchFieldException e) {
					System.err.println("Error while trying to delete attribute from dataset.");
					e.printStackTrace();
				}
				Attribute attbk = atts.remove(atts.size()-1); // remove last attribute, which corresponds to the timestamp
				Integer indexbk = indexes.remove(indexes.size()-1);
				attinfo.setAttributes(atts, indexes);
		assertTrue(oates.learners.get(1).trainingHasStarted());				
		assertEquals(3.0,oates.learners.get(1).getVotesForInstance(ccInst)[0]);
				atts.add(attbk);
				indexes.add(indexbk);
				attinfo.setAttributes(atts, indexes);
		
				
				
		attvals[0] = 1;
		attvals[1] = 3;
		attvals[2] = 2;
		attvals[3] = 5;
		ccInst = new DenseInstance(1, attvals);
		ccInst.setDataset(dataset);
		
		oates.trainOnCCInstance(ccInst);
		
		assertEquals(3,oates.mappingFunctions.size());
		assertEquals(4,oates.learners.size());
		assertEquals(4,oates.weights.size());
		ccInst.deleteAttributeAt(3);
				// Delete attribute from dataset, as some base learners will make use of the information from the dataset.
				atts = null;
				indexes = null;
				attinfo = null;
				try {
					InstanceInformation instanceinfo = (InstanceInformation) PrivateAccessor.getField(dataset, "instanceInformation");
					attinfo = (AttributesInformation) PrivateAccessor.getField(instanceinfo, "attributesInformation");
					atts = (List<Attribute>) PrivateAccessor.getField(attinfo, "attributes");
					indexes = (List<Integer>) PrivateAccessor.getField(attinfo, "indexValues");
				} catch (NoSuchFieldException e) {
					System.err.println("Error while trying to delete attribute from dataset.");
					e.printStackTrace();
				}
				attbk = atts.remove(atts.size()-1); // remove last attribute, which corresponds to the timestamp
				indexbk = indexes.remove(indexes.size()-1);
				attinfo.setAttributes(atts, indexes);
		assertTrue(oates.learners.get(0).trainingHasStarted());
		assertEquals(2.0,oates.learners.get(0).getVotesForInstance(ccInst)[0]);
				atts.add(attbk);
				indexes.add(indexbk);
				attinfo.setAttributes(atts, indexes);
				
		assertFalse(oates.learners.get(2).trainingHasStarted());
		
		
		
		
		attvals[0] = 1.1;
		attvals[1] = 3.1;
		attvals[2] = 2.1;
		attvals[3] = 5;
		ccInst = new DenseInstance(1, attvals);
		ccInst.setDataset(dataset);
		
				ccInst.deleteAttributeAt(3);
				// Delete attribute from dataset, as some base learners will make use of the information from the dataset.
				atts = null;
				indexes = null;
				attinfo = null;
				try {
					InstanceInformation instanceinfo = (InstanceInformation) PrivateAccessor.getField(dataset, "instanceInformation");
					attinfo = (AttributesInformation) PrivateAccessor.getField(instanceinfo, "attributesInformation");
					atts = (List<Attribute>) PrivateAccessor.getField(attinfo, "attributes");
					indexes = (List<Integer>) PrivateAccessor.getField(attinfo, "indexValues");
				} catch (NoSuchFieldException e) {
					System.err.println("Error while trying to delete attribute from dataset.");
					e.printStackTrace();
				}
				attbk = atts.remove(atts.size()-1); // remove last attribute, which corresponds to the timestamp
				indexbk = indexes.remove(indexes.size()-1);
				attinfo.setAttributes(atts, indexes);
		assertTrue(oates.learners.get(0).trainingHasStarted());
		assertEquals(2.0,oates.learners.get(0).getVotesForInstance(ccInst)[0]);
				atts.add(attbk);
				indexes.add(indexbk);
				attinfo.setAttributes(atts, indexes);

				
		// The second set of tests are with dycom2. This means that the trainOnCCInstance method will result in a new cluster being created
		attvals[0] = 1;
		attvals[1] = 3;
		attvals[2] = 2;
		attvals[3] = 5;
		ccInst = new DenseInstance(1, attvals);
		ccInst.setDataset(dataset);
		
		for (int i=0; i<20; ++i)
			oates2.trainOnCCInstance(ccInst);
		
		// There will be only one cluster, because the clustering algorithm has been trained with 20 copies of the same instance
		assertEquals(1,oates2.mappingFunctions.size());
		assertEquals(2,oates2.learners.size());
		assertEquals(2,oates2.weights.size());
		ccInst.deleteAttributeAt(3);
				// Delete attribute from dataset, as some base learners will make use of the information from the dataset.
				atts = null;
				indexes = null;
				attinfo = null;
				try {
					InstanceInformation instanceinfo = (InstanceInformation) PrivateAccessor.getField(dataset, "instanceInformation");
					attinfo = (AttributesInformation) PrivateAccessor.getField(instanceinfo, "attributesInformation");
					atts = (List<Attribute>) PrivateAccessor.getField(attinfo, "attributes");
					indexes = (List<Integer>) PrivateAccessor.getField(attinfo, "indexValues");
				} catch (NoSuchFieldException e) {
					System.err.println("Error while trying to delete attribute from dataset.");
					e.printStackTrace();
				}
				attbk = atts.remove(atts.size()-1); // remove last attribute, which corresponds to the timestamp
				indexbk = indexes.remove(indexes.size()-1);
				attinfo.setAttributes(atts, indexes);
		
		int trainingStarted = -1;
		for (int i=0; i<oates2.learners.size(); ++i) {
			if(oates2.learners.get(i).trainingHasStarted())
				trainingStarted = i;
		}
		assertTrue(trainingStarted != -1);
		assertEquals(2.0,oates2.learners.get(trainingStarted).getVotesForInstance(ccInst)[0]);
				atts.add(attbk);
				indexes.add(indexbk);
				attinfo.setAttributes(atts, indexes);
				
				

		// Try to make k-means algorithm create all 3 clusters
		for (int i=0; i<10; ++i) {
			attvals[0] = i;
			attvals[1] = i;
			attvals[2] = i;
			attvals[3] = i;
			ccInst = new DenseInstance(1, attvals);
			ccInst.setDataset(dataset);
			oates2.trainOnCCInstance(ccInst);
		}
			
		
		assertEquals(3,oates2.mappingFunctions.size());
		assertEquals(4,oates2.learners.size());
		assertEquals(4,oates2.weights.size());		
	}


	public void testUpdateWeights() {
		
		for (int i=0; i<oates.weights.size(); ++i)
			assertEquals(1.0, oates.weights.get(i));

		double [] attvals = {1,2,3,4};
		DenseInstance inst = new DenseInstance(1, attvals);
		inst.setDataset(dataset);
		oates.makeChecks(inst);
		
		List<Attribute> atts = new ArrayList<Attribute>();
		for (int i=0; i<inst.numAttributes()-1; ++i) {
			atts.add(inst.attribute(i));
		}
		Instances datasetNoTimeStamp = new Instances(inst.dataset().getRelationName(), atts, 0);
		datasetNoTimeStamp.setClassIndex(inst.classIndex());
		inst.deleteAttributeAt(3);
		inst.setDataset(datasetNoTimeStamp);

		oates.wcPastInstancesQueue.add(inst);
		oates.learners.get(1).trainOnInstance(inst);
		
		attvals = new double[3];
		attvals[0] = 5;
		attvals[1] = 6;
		attvals[2] = 7;
		inst = new DenseInstance(1, attvals);
		inst.setDataset(datasetNoTimeStamp);
		
		oates.learners.get(0).trainOnInstance(inst);
		oates.updateWeights();
		
		assertEquals(1.0/1.5,oates.weights.get(1));
		assertEquals(0.5/1.5,oates.weights.get(0));
		assertEquals(1.0,oates.weights.get(2));
		assertEquals(1.0,oates.weights.get(3));
		
		attvals[0] = 10;
		attvals[1] = 11;
		attvals[2] = 12;
		inst = new DenseInstance(1, attvals);
		inst.setDataset(datasetNoTimeStamp);
		
		oates.learners.get(3).trainOnInstance(inst);
		oates.updateWeights();
				
		assertEquals(1.0/2.1,oates.weights.get(1));
		assertEquals(0.5/2.1,oates.weights.get(0));
		assertEquals(0.6/2.1,oates.weights.get(3));
		assertEquals(1.0,oates.weights.get(2));
		
	}

	
	public void testTrainMappingFunctions() {
		
		for (int i=0; i<oates.mappingFunctions.size(); ++i) 
			assertEquals(1.0,((SimpleLinearMappingFunction)oates.mappingFunctions.get(i)).getB());
		
		
		double [] attvals = {1,2,3,4};
		DenseInstance inst = new DenseInstance(1, attvals);
		inst.setDataset(dataset);
		oates.makeChecks(inst);
		
		List<Attribute> atts = new ArrayList<Attribute>();
		for (int i=0; i<inst.numAttributes()-1; ++i) {
			atts.add(inst.attribute(i));
		}
		Instances datasetNoTimeStamp = new Instances(inst.dataset().getRelationName(), atts, 0);
		datasetNoTimeStamp.setClassIndex(inst.classIndex());
		inst.deleteAttributeAt(3);
		inst.setDataset(datasetNoTimeStamp);

		oates.wcPastInstancesQueue.add(inst);
		oates.learners.get(1).trainOnInstance(inst);
		assertTrue(oates.learners.get(1).trainingHasStarted());
		
		double []attvals2 = {10,20,30};
		DenseInstance inst2 = new DenseInstance(1, attvals2);
		inst2.setDataset(dataset);
		oates.wcPastInstancesQueue.add(inst2);
		
		oates.trainMappingFunctions();
		
		assertEquals(1.9,((SimpleLinearMappingFunction)oates.mappingFunctions.get(1)).getB());
		assertEquals(1.0,((SimpleLinearMappingFunction)oates.mappingFunctions.get(0)).getB());
		assertEquals(1.0,((SimpleLinearMappingFunction)oates.mappingFunctions.get(2)).getB());
		
		oates.learners.get(1).trainOnInstance(inst2);
		oates.trainMappingFunctions();

		assertEquals(1.0,((SimpleLinearMappingFunction)oates.mappingFunctions.get(1)).getB());
		assertEquals(1.0,((SimpleLinearMappingFunction)oates.mappingFunctions.get(0)).getB());
		assertEquals(1.0,((SimpleLinearMappingFunction)oates.mappingFunctions.get(2)).getB());
		
		// wc instances will include only <15,25,35> and <150,250,350>, whereas CC lerner 1 will have been trained on <1,2,3> and <10,20,30>
		oates.wcPastInstancesQueue.clear();
		double []attvals3 = {15,25,35};
		DenseInstance inst3 = new DenseInstance(1, attvals3);
		inst3.setDataset(dataset);
		oates.wcPastInstancesQueue.add(inst3);
		
		double []attvals4 = {150,250,350};
		DenseInstance inst4 = new DenseInstance(1, attvals4);
		inst4.setDataset(dataset);
		oates.wcPastInstancesQueue.add(inst4);

		oates.trainMappingFunctions();
		
		assertEquals(Math.round(2.216666666666667 * 1000000),Math.round(((SimpleLinearMappingFunction)oates.mappingFunctions.get(1)).getB() * 1000000));
		assertEquals(1.0,((SimpleLinearMappingFunction)oates.mappingFunctions.get(0)).getB());
		assertEquals(1.0,((SimpleLinearMappingFunction)oates.mappingFunctions.get(2)).getB());		
		
	}

	public void testTrainOnInstanceImpl() {
		oates.resetLearning();
		assertEquals(4,oates.learners.size());
		
		
		oates.trainOnInstanceImpl(wcDataStream.nextInstance().instance);
		assertEquals(1,oates.wcPastInstancesQueue.size());
		assertTrue(oates.learners.get(3).trainingHasStarted());
		int numCCModelsTrained = 0, numMappingModelsTrained = 0;
		for (int i=0; i<3; ++i) {
			if (oates.learners.get(i).trainingHasStarted()) {
				numCCModelsTrained++;
				numMappingModelsTrained++;
			}
		}
		// Only 1 CC model should have been trained, as only 1 CC instance had timestamp before the WC instance
		assertEquals(1,numCCModelsTrained);
		assertEquals(1,numMappingModelsTrained);
		assertEquals(0.5/1.5,oates.weights.get(3));
		
		
		oates.trainOnInstanceImpl(wcDataStream.nextInstance().instance);
		assertEquals(2,oates.wcPastInstancesQueue.size());
		assertTrue(oates.learners.get(3).trainingHasStarted());
		int numCCInstsTrainingCCModel = 0, numCCInstsTrainingMappingModel = 0;
		for (int i=0; i<3; ++i) {
			if (oates.learners.get(i).trainingHasStarted()) {
				numCCInstsTrainingCCModel += oates.learners.get(i).trainingWeightSeenByModel();
				numCCInstsTrainingMappingModel += oates.mappingFunctions.get(i).trainingWeightSeenByModel();
			}
		}
		// All 3 CC training instances should have been used for training, as all of them had timestamp before the WC instance
		assertEquals(3,numCCInstsTrainingCCModel);
		
		
		while (wcDataStream.hasMoreInstances()) {
			Instance inst = wcDataStream.nextInstance().instance;
			oates.trainOnInstanceImpl(inst);
			assertEquals(3,oates.wcPastInstancesQueue.size());
			assertEquals(inst.classValue(),oates.wcPastInstancesQueue.get(2).classValue());
			
		}
		
		numCCInstsTrainingCCModel = 0; numCCInstsTrainingMappingModel = 0;
		for (int i=0; i<3; ++i) {
			if (oates.learners.get(i).trainingHasStarted()) {
				numCCInstsTrainingCCModel += oates.learners.get(i).trainingWeightSeenByModel();
				numCCInstsTrainingMappingModel += oates.mappingFunctions.get(i).trainingWeightSeenByModel();				
			}
		}
		// No more than the previous 3 CC training instances should have been used for training, as there are no other CC instances
		assertEquals(3,numCCInstsTrainingCCModel);
		
		wcDataStream.restart();
		
		while(wcDataStream.hasMoreInstances()) {
			Instance inst = wcDataStream.nextInstance().instance;
			inst.deleteAttributeAt(inst.numAttributes()-1);
			assertEquals(inst.classValue(),oates.learners.get(3).getVotesForInstance(inst)[0]);
		}
		
		
		// Case where there is no cc instance
		oatesNoCC.resetLearning();
		wcDataStream.restart();
		oatesNoCC.trainOnInstanceImpl(wcDataStream.nextInstance().instance);
		assertTrue(oatesNoCC.learners.get(3).trainingHasStarted());
		assertFalse(oatesNoCC.learners.get(0).trainingHasStarted());
		assertFalse(oatesNoCC.learners.get(1).trainingHasStarted());
		assertFalse(oatesNoCC.learners.get(2).trainingHasStarted());
		for (int i=0; i<4; ++i)
			assertEquals(1.0, oatesNoCC.weights.get(i));
	}

	
	public void testGetVotesForInstance() {
		oates.resetLearning();
		assertEquals(4,oates.learners.size());
		
		double [] attvals = {1,2,3,4};
		DenseInstance inst = new DenseInstance(1, attvals);
		inst.setDataset(dataset);
		oates.makeChecks(inst);
		
		List<Attribute> atts = new ArrayList<Attribute>();
		for (int i=0; i<inst.numAttributes()-1; ++i) {
			atts.add(inst.attribute(i));
		}
		Instances datasetNoTimeStamp = new Instances(inst.dataset().getRelationName(), atts, 0);
		datasetNoTimeStamp.setClassIndex(inst.classIndex());
		Instance instNoTimeStamp = inst.copy();
		instNoTimeStamp.deleteAttributeAt(3);
		instNoTimeStamp.setDataset(datasetNoTimeStamp);

		//dycom.wcPastInstancesQueue.add(inst);
		// Train one CC model
		oates.learners.get(1).trainOnInstance(instNoTimeStamp);
		assertTrue(oates.learners.get(1).trainingHasStarted());		
		
		double [] attvals2 = {10,20,30};
		DenseInstance instNoTimeStamp2 = new DenseInstance(1, attvals2);
		instNoTimeStamp2.setDataset(datasetNoTimeStamp);
		
		// Train the WC model
		oates.learners.get(3).trainOnInstance(instNoTimeStamp2);
		assertTrue(oates.learners.get(3).trainingHasStarted());
				
		assertFalse(oates.learners.get(0).trainingHasStarted());		
		assertFalse(oates.learners.get(2).trainingHasStarted());	
				
		// Ensure that no further CC training will take place in OATES, besides the training done manually above
		while(oates.ccDataStream.hasMoreInstances())
			oates.ccDataStream.nextInstance();
		
		// Add one instance to the wc queue, to be used to update OATES' weights when asking to get the votes
		// The CC mapped model will give a correct prediction to this instance
		// The WC model should have its weight multiplied by 0.6
		// The new weights should be 1/1.6 for the CC model 1, and 0.6/1.6 for the WC model
		oates.wcPastInstancesQueue.add(instNoTimeStamp);
		
		assertEquals(Math.round((3 * 1/1.6 + 30 * 0.6/1.6)*1000000), Math.round(oates.getVotesForInstance(inst)[0]*1000000));
		
	}
	
	public void testTrainCCModels() {
		
		// Case where all CC data should be used for training
		double [] attvals = {1,2,3,4};
		DenseInstance inst = new DenseInstance(1, attvals);
		inst.setDataset(dataset);
		
		oates.resetLearning();
		oates.makeChecks(inst);
		
		oates.trainCCModels(inst);
		int numCCInstancesTrained = 0;
		
		for (int i=0; i<oates.learners.size(); i++)
			numCCInstancesTrained += oates.learners.get(i).trainingWeightSeenByModel();
		assertEquals(3, numCCInstancesTrained);
		assertEquals(0, oates.ccPastInstancesQueue.size());
		
		// Case where only 1 CC instance should be used for training
		double [] attvals2 = {1,2,3,2};
		DenseInstance inst2 = new DenseInstance(1, attvals2);
		inst2.setDataset(dataset);
		
		oates.resetLearning();
		oates.makeChecks(inst2);
		
		oates.trainCCModels(inst2);
		numCCInstancesTrained = 0;
		
		for (int i=0; i<oates.learners.size(); i++)
			numCCInstancesTrained += oates.learners.get(i).trainingWeightSeenByModel();
		assertEquals(1, numCCInstancesTrained);
		assertEquals(1, oates.ccPastInstancesQueue.size());
		assertTrue(oates.ccDataStream.hasMoreInstances());

		// Case where only 0 CC instances should be used for training
		double [] attvals3 = {1,2,3,0};
		DenseInstance inst3 = new DenseInstance(1, attvals3);
		inst3.setDataset(dataset);

		oates.resetLearning();
		oates.makeChecks(inst3);

		oates.trainCCModels(inst3);
		numCCInstancesTrained = 0;

		for (int i=0; i<oates.learners.size(); i++)
			numCCInstancesTrained += oates.learners.get(i).trainingWeightSeenByModel();
		assertEquals(0, numCCInstancesTrained);
		assertEquals(1, oates.ccPastInstancesQueue.size());
		assertTrue(oates.ccDataStream.hasMoreInstances());

		
		// Case where there are no CC instances
		oatesNoCC.resetLearning();
		oatesNoCC.makeChecks(inst);
		
		assertFalse(oatesNoCC.ccDataStream.hasMoreInstances());
		oatesNoCC.trainCCModels(inst);
		
		assertFalse(oatesNoCC.learners.get(0).trainingHasStarted());
		assertFalse(oatesNoCC.learners.get(1).trainingHasStarted());
		assertFalse(oatesNoCC.learners.get(2).trainingHasStarted());		
		
		assertEquals(1.0, oatesNoCC.weights.get(3));
		
	}
		
}
