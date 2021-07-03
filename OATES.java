
/**
 * @author Leandro L. Minku, University of Birmingham, l.l.minku@cs.bham.ac.uk
 * 
 * This class implements OATES. Both WC and CC data arrive online.
 * It requires each WC and CC data stream to have an attribute to represent the timestamp when the training example was received.
 * 
 * Note that it is possible to run this without using CC data by entering an arff CC file containing only the headers, without actual data.
 * If you do that and select wcQueueSize = 1, this will be equivalent to running a WC learner using a certain period. E.g.:
 * 
 *  EvaluatePrequentialRegression -l (meta.OATES -l (meta.WEKAClassifierTrainSlidingWindow -l (weka.classifiers.trees.REPTree -M 1 -V 1.0E-4 -N 3 -S 1 -L -1 -P -I 0.0) -w 100000) -c (threshold.ProductivitySplitClusterer -t 5.6;13.0 -e 4 -s 3) -d (/Users/llm11/Leandro's Files/Work/Approaches/MOA-2016.04-leandro/isbsg-empty.arff) -p 1 -q 1) -s (ArffFileStream -f (/Users/llm11/Leandro's Files/Work/Approaches/MOA-2016.04-leandro/single_projects_data_proc3_after_2000_timestamp.arff) -c 5) -e (WindowRegressionPerformanceEvaluator -w 1) -f 1
 *  
 * If period = 1 as in the example above, this is equivalent to using MOA to run a WC learner directly rather than through oates, as in the example below:
 * 
 *  EvaluatePrequentialRegression -l (meta.WEKAClassifierTrainSlidingWindow -l (weka.classifiers.trees.REPTree -M 1 -V 1.0E-4 -N 3 -S 1 -L -1 -P -I 0.0) -w 10000) -s (ArffFileStream -f (/Users/llm11/Leandro's Files/Work/Approaches/MOA-2016.04-leandro/single_projects_data_proc3_after_2000.arff) -c 5) -e (WindowRegressionPerformanceEvaluator -w 1) -f 1
 * 
 * If period > 1, this will run the WC learner using only every period WC training examples.
 * 
 * The implementation can also be tuned to run the original Dycom from ICSE 2014, based on the option wcPastInstancesQueueSize.
 * 
 * 
 * * Instructions to run Dycom as in the following paper:
 * 
 * MINKU, L. L.; YAO, X.; "How to Make Best Use of Cross-company Data in Software Effort Estimation?", Proceedings of the 36th International Conference on Software Engineering (ICSE'14), p. 446-456, May 2014, doi: 10.1145/2568225.2568228. 
 * 
 * ***** Set wcQueueSize option to 1 and choose an appropriate period (p) value. *****
 * 
 * Data sets must have an attribute to represent the timestamp when the training example was received.
 * 
 * Note that it is possible to run this without using CC data by entering an arff CC file containing only the headers, without actual data.
 * If you do that and select wcQueueSize = 1, this will be equivalent to running a WC learner using a certain period.   
 * If period = 1, this is equivalent to using MOA to run a WC learner directly rather than through oates, as in the example below:
 * If period > 1, this will run the WC learner using only every period WC training examples.
 * 
 * To use WEKA's base learners, choose WEKAClassifierTrainSlidingWindow as base learner.
 * 
 * For evaluation, use EvaluatePrequentialRegression + WindowClassificationPerformanceEvaluator with a window of size 10.
 * 
 * Example of command line for CocNasaCoc81:
 * 
 * EvaluatePrequentialRegression -l (meta.OATES -l (meta.WEKAClassifierTrainSlidingWindow -l (weka.classifiers.trees.REPTree -M 1 -V 1.0E-4 -N 3 -S 1 -L -1 -P -I 0.0) -w 100000) -c (threshold.ProductivitySplitClusterer -t 2.85;6.60) -d (/Users/llm11/Leandro's Files/Work/Approaches/MOA-2016.04-leandro/promise_nominal_edited0_sorted_timestamp.arff) -q 1) -s (ArffFileStream -f (/Users/llm11/Leandro's Files/Work/Approaches/MOA-2016.04-leandro/cocomonasa_nominal_edited_timestamp.arff) -c 17) -e (WindowRegressionPerformanceEvaluator -w 10) -f 1
 * 
 */

package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.List;

import org.joda.time.Instant;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.Regressor;
import moa.cluster.Clustering;
import moa.clusterers.Clusterer;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.streams.ArffFileStream;

public class OATES extends AbstractClassifier implements Regressor {

	private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Type of base learner to train WC and CC data.", Regressor.class, "lazy.kNN -k 1"); // Classifier.class  trees.HoeffdingTree

    public ClassOption mappingFunctionOption = new ClassOption("mappingFunction", 'm',
            "Learner to use as mapping function.", Regressor.class, "meta.SimpleLinearMappingFunction");
   
    public ClassOption clustererOption = new ClassOption("clusterer", 'c',
            "Clusterer to split CC data. Warning: this is unlikely to work well in cases where clusters can be merged over time, " +
            "or where training instances used with a given cluster may become members of another cluster over time. This is " +
            "because the learner corresponding to each cluster does not get redistributed to other clusters when this happens. "+
            "So, even though we can choose any cluster option here, most clustering algorithms are likely not to work well with " +
            "OATES.",
            Clusterer.class, "threshold.ProductivitySplitClusterer");
    
    //public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
    //        "The number of models in the ensemble. It should be the number of CC models + 1 WC model.", 10, 1, Integer.MAX_VALUE);
    
	public FileOption ccDataStreamFile = new FileOption("ccDataStream", 'd', 
			"Name of the cc data stream.", "", ".arff", false);
	//without numeric identifier and extension. " +
	//		"NumId.arff will be added to the file name. " +
	//		"Warning: these files must follow the same format as the WC data stream and the last attribute must be the timestamp.", "");
	
	public FloatOption betaWC = new FloatOption("betaWC", 'b', 
			"Beta value for updating WC weights", 0.5, 0.0, 1.0);
	
	public FloatOption betaCC = new FloatOption("betaCC", 'g', 
			"Beta value for updating CC weights", 0.5, 0.0, 1.0);
	
	public IntOption period = new IntOption("period", 'p',
            "Period between WC training examples.", 10, 1, Integer.MAX_VALUE);

	public IntOption wcPastInstancesQueueSize = new IntOption("wcQueueSize", 'q',
            "Number of past WC instances to store for evaluating updates in CC models. If set to 1, the code is changed to make this equal " + 
            "to the original Dycom and do not perform any training with CC instances after WC training has started (not completely tested). " +
            "If set to 1 while using empty CC data sets containing only the arff headers, this will run a WC approach using the specified period.", 
            Integer.MAX_VALUE, 1, Integer.MAX_VALUE);
	
	// The first learners are the CC learners and the last learner is the WC learner.
	protected ArrayList<Classifier> learners;
	private Classifier baseLearner;
	
	// Clustering algorithm to use for clustering CC instances
	protected Clusterer clusterer;
	private Clustering currentClusters;
	
	// Mapping functions for CC learners
	// All mapping functions should be classifiers implementing the MappingFunction interface.
	protected ArrayList<Classifier> mappingFunctions;
	private Classifier baseMappingFunction;
	
	// Weight associated with each learner
	protected ArrayList<Double> weights;
	
	
	protected ArffFileStream ccDataStream;
	protected boolean isCCClassIndexSet;
	
	// Used to set the weights of the learners, instead of the beta values.
	// This is needed because CC training examples may be received before any WC training example is received.
	// In order to use these examples, we need to have access to some past WC training examples.
	// Moreover, as CC training examples may be learnt over time, simply updating the weights based on a single
	// most recent WC training example may be insufficient to properly test each updated CC model.
	// The mapping function faces a similar issue.
	// Stores instances without time stamps.
	protected ArrayList<Instance> wcPastInstancesQueue;
	
	// We need to read the next instance of each CC data stream to check whether it can already be used for training.
	// However, if the timestamp of this CC instance indicates that it cannot be used for training yet, we need to store it in
	// this bk so that it can be used for training later.
	// These instances have not been used for training the clusters and the CC learners. 
	// Stores instances with time stamps.
	protected ArrayList<Instance> ccPastInstancesQueue;
	
	// These instances have already been used to train the clusters, but not yet to train the corresponding CC learners.
	// Stores instances without time stamps.
	protected ArrayList<Instance> ccInstancesWaitingForWindowTraining;
	
	protected int timeStep;
	
	// Maintains attribute info about instances without timestamp
	protected Instances datasetNoTimeStamp;
	
	public OATES() {
		super();
	}
	
	// This method is inspired by OzaBag's.
	@Override
	public void resetLearningImpl() {
		
		timeStep = 0; // no training done yet
		
        learners = new ArrayList<Classifier>(); 
        weights = new ArrayList<Double>();
        mappingFunctions = new ArrayList<Classifier>();
        
		// Create WC learner and reset its weight
        baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        learners.add(baseLearner.copy());
        weights.add(1.0);
        
        baseMappingFunction = (Classifier) getPreparedClassOption(this.mappingFunctionOption);
        baseMappingFunction.resetLearning();
		
		clusterer = (Clusterer) getPreparedClassOption(this.clustererOption);
        clusterer.resetLearning();
        currentClusters = clusterer.getClusteringResult(); 
        
        // Some clustering algorithms will have created clusters even before using any example for training.
        // In this case, we can already create the corresponding CC learners here
        if (currentClusters != null && currentClusters.size() != 0)         		
       		addNewCCLearners(currentClusters.size());
        
		// Load CC data stream
        isCCClassIndexSet = false;
        ccDataStream = new ArffFileStream(ccDataStreamFile.getValue(),-1);
        
        
        // Reset the queue of WC training examples and the BK CC training examples
        this.wcPastInstancesQueue = new ArrayList<Instance>();
        this.ccPastInstancesQueue = new ArrayList<Instance>();
        this.ccInstancesWaitingForWindowTraining = new ArrayList<Instance>();
        
        datasetNoTimeStamp = null;
        
	}	
	
	@Override
    public Classifier[] getSubClassifiers() {
		Classifier []learnersArray = new Classifier[learners.size()];
        return learners.toArray(learnersArray);
    }
	
	public int getNumTrainedLearners() {
		int numTrainedLearners = 0;
		for (int i=0; i<learners.size(); ++i)
			if (learners.get(i).trainingHasStarted())
				numTrainedLearners++;
		return numTrainedLearners;
	}
	
	// Create and add <num> new CC learners to the beginning of the learners array list
	// Add its corresponding weight and mapping function too
	protected void addNewCCLearners(int num) {
		for (int i=0; i<num; ++i) {
			learners.add(0,baseLearner.copy());
			weights.add(0,1.0); 
			mappingFunctions.add(0,baseMappingFunction.copy());
			((MappingFunction) mappingFunctions.get(0)).setCCLearner(learners.get(0));
		}
	}
	
	// OATES itself is not randomisable, even though its base learners may be.
	// This method is the same as AccuracyUpdatedEnsemble's.
	@Override
	public boolean isRandomizable() {
		return false;
	}
	
	
	// Set the class index of the CC data streams to be the same as that of inst
	protected void setCCClassIndex(Instance inst) throws Exception {

		if (inst.numAttributes() != ccDataStream.getHeader().numAttributes()) 
			throw new Exception("Number of attributes in CC data stream is different from that of the WC data stream.");

		ccDataStream.getHeader().setClassIndex(inst.classIndex());
		//System.out.println(this.ccDataStreams[i].nextInstance().instance.classIndex());
	}
	
	// Make checks to see if oates is prepared to train or test with inst.
	// Tries to set CC class index if not yet set.
	public boolean makeChecks(Instance inst) {
		if (inst.classAttribute().isNominal()) {
			System.err.println("This is a classification problem. OATES can only be used for regression problems.");
			return false;
		}
		
		// Set class attribute of cc data streams, if not set
		if (!isCCClassIndexSet) {
			try {
				setCCClassIndex(inst);
			} catch (Exception e) {
				System.err.println("Error: Unable to set CC class index.");
				e.printStackTrace();
				return false;
			}
		}
		
		// Create a dataset containing attribute information about instances without timestamp
		if (datasetNoTimeStamp == null) {
			List<Attribute> atts = new ArrayList<Attribute>();
			for (int i=0; i<inst.numAttributes()-1; ++i) {
				atts.add(inst.attribute(i));
			}
			datasetNoTimeStamp = new Instances(inst.dataset().getRelationName(), atts, 0);
			datasetNoTimeStamp.setClassIndex(inst.classIndex());
		}
						
		return true;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		
		if (!makeChecks(inst)) {
			System.err.println("Halting training.");
			return;
		}
		
		// Train CC models and update mapping functions accordingly
		if (this.wcPastInstancesQueueSize.getValue() != 1 || timeStep == 0) {// TEST<--- added this if to make this like the original dycom
			trainCCModels(inst);
			trainMappingFunctions();
			updateWeights(); // Update weights based on the up-to-date CC models and mapping functions
		}
		
		// Add WC instances for training only at every period time steps
		if (timeStep % this.period.getValue() == 0) {
			// Store this instance in the list of most recent WC training instances
			if (this.wcPastInstancesQueue.size() == this.wcPastInstancesQueueSize.getValue())
				wcPastInstancesQueue.remove(0);
			Instance instNoTimeStamp = inst.copy();
			deleteTimeStamp(instNoTimeStamp);
			wcPastInstancesQueue.add(instNoTimeStamp);
		}
		else { 
			this.trainingWeightSeenByModel = this.trainingWeightSeenByModel - inst.weight(); // correct the statistics about how much WC weight has been used by OATES for training. 
		}
			
		// <--- before the update weights was here, which would cause weights to be wrongly updated based on the most recent wc instance, even if it was not used for training!
			
		// Use WC instances for training only at every period time steps
		if (timeStep % this.period.getValue() == 0) {
			updateWeights(); // update the weights of the learners based on the most recent WC instance before using this WC instance to train any mapping function or WC learner.
			
			trainMappingFunctions();
			
			// Train WC model
			learners.get(learners.size()-1).trainOnInstance(wcPastInstancesQueue.get(wcPastInstancesQueue.size()-1));
			
			// The weight of the WC learner will be used for the first time, so all weights need to be normalised
			if (timeStep == 0) {
				weights.set(weights.size()-1, 1.0/getNumTrainedLearners()); // <---le18Aug2017 this makes this equal to the original dycom
				normaliseWeights();
			}
			
			//this.recoverDeletedAttDataset();
		}
		
		timeStep++;

	}
	
	// Retrain mapping functions using the instances in wcPastInstancesQueue
	protected void trainMappingFunctions() {

		for (int i=0; i<this.mappingFunctions.size(); ++i) {
			// Only mapping functions corresponding to CC models that have already been trained should be trained
			if (learners.get(i).trainingHasStarted()) {
				if (wcPastInstancesQueueSize.getValue() != 1) // TEST<--- added this if to make this like the original dycom  
					mappingFunctions.get(i).resetLearning();
				((MappingFunction) mappingFunctions.get(i)).setCCLearner(learners.get(i));
				for (int j=0; j<wcPastInstancesQueue.size(); ++j) 
					mappingFunctions.get(i).trainOnInstance(wcPastInstancesQueue.get(j));
			}
		}
	}

	
	// Update weights based on the past WC examples
	protected void updateWeights() {
		
		if (wcPastInstancesQueueSize.getValue() != 1) { // TEST<--- added this if to make this like the original dycom
			for (int i=0; i<weights.size(); ++i) {
				weights.set(i, 1.0);
			}
		}
		
		for (int j=0; j<wcPastInstancesQueue.size(); ++j) {
			
			Instance instNoTimeStamp = wcPastInstancesQueue.get(j);
			
			double minError = Double.MAX_VALUE;
			int indexMinError = -1;
			for (int i=0; i<weights.size(); ++i) {
				if (learners.get(i).trainingHasStarted()) {
					double []votes = null;
					
					if (i < weights.size()-1)
						votes = mappingFunctions.get(i).getVotesForInstance(instNoTimeStamp);
					else votes = learners.get(i).getVotesForInstance(instNoTimeStamp);
					
					double error = Math.abs(votes[0] - instNoTimeStamp.classValue());
					
					if (error < minError) {
						minError = error;
						indexMinError = i;
					}
				}
			}
			
			// Multiply weights of losers by beta
			for (int i=0; i<weights.size(); ++i) {
				if (i != indexMinError && i < weights.size()-1 && learners.get(i).trainingHasStarted())
					weights.set(i, weights.get(i) * betaCC.getValue());
				else if (i != indexMinError && i == weights.size()-1 && learners.get(i).trainingHasStarted())
					weights.set(i, weights.get(i) * betaWC.getValue());
			}
			
			//recoverDeletedAttDataset();
		}
		
		normaliseWeights();
		
	}

	protected void normaliseWeights() {
		double sumWeights = 0.0;
		
		for (int i=0; i<weights.size(); ++i) {
			if (learners.get(i).trainingHasStarted())
				sumWeights += weights.get(i);
		}
		
		for (int i=0; i<weights.size(); ++i) {
			if (learners.get(i).trainingHasStarted())
				weights.set(i, weights.get(i) / sumWeights);
		}
	}
	
	
	// Find the cluster to which inst belongs.
	// inst should be an instance without timestamp
	protected int indexClusterInstanceBelongsTo(Clustering clusters, Instance inst) {
		
		double maxProb = -1.0;
		int indexMaxProb = -1;
		
		for (int c=0; c<clusters.size(); ++c) {
			double prob = clusters.get(c).getInclusionProbability(inst);
			if (prob > maxProb) {
				maxProb = prob;
				indexMaxProb = c;
			}
		}
		
		return indexMaxProb;
	}

	// Delete an attribute from inst, and update its corresponding dataset to be the one containing the list of attributes without timestamp
	protected void deleteTimeStamp(Instance inst) {
		inst.deleteAttributeAt(inst.numAttributes()-1);
		inst.setDataset(datasetNoTimeStamp);
	}
	
//	// Delete an attribute from inst, and from its corresponding dataset
//	Attribute attbk;
//	Integer indexbk;
//	AttributesInformation attinfo;
//	List<Attribute> atts;
//	List<Integer> indexes;
//	protected void deleteAtt(Instance inst) {
//		inst.deleteAttributeAt(inst.numAttributes()-1);
//		
//		// Delete attribute from dataset, as some base learners will make use of the information from the dataset.
//		try {
//			InstanceInformation instanceinfo = (InstanceInformation) PrivateAccessor.getField(inst.dataset(), "instanceInformation");
//			attinfo = (AttributesInformation) PrivateAccessor.getField(instanceinfo, "attributesInformation");
//			atts = (List<Attribute>) PrivateAccessor.getField(attinfo, "attributes");
//			indexes = (List<Integer>) PrivateAccessor.getField(attinfo, "indexValues");
//		} catch (NoSuchFieldException e) {
//			System.err.println("Error while trying to delete attribute from dataset.");
//			e.printStackTrace();
//		}
//		attbk = atts.remove(atts.size()-1); // remove last attribute, which corresponds to the timestamp
//		indexbk = indexes.remove(indexes.size()-1);
//		attinfo.setAttributes(atts, indexes);
//	}
//	
//	// Recover the attribute in the dataset
//	protected void recoverDeletedAttDataset() {
//		atts.add(attbk);
//		indexes.add(indexbk);
//		attinfo.setAttributes(atts, indexes);
//	}
	
	// Train clusterer and CC learners on new CC Instance
	// This instance should contain the timestamp
	// Assumes that this instance came after the current WC instances
	// Does not make any weight updates
	protected void trainOnCCInstance(Instance inst) {
		if (!makeChecks(inst)) {
			System.out.println("Error while training CC models.");
			return;
		}
		
		Instance instNoTimeStamp = inst.copy();
		// NEEDS TO UPDATE THE DATASET TO CONTAIN INFORMATION ABOUT INSTANCES WITH ONE LESS ATT,
		// AS SOME BASE LEARNERS MAY NEED TO USE INFORMATION ABOUT THE DATASET FOR TRAINING OR PREDICTIONS.
		deleteTimeStamp(instNoTimeStamp);

		int currentSize = 0;
		if (currentClusters != null)
			currentSize = currentClusters.size();
		
		clusterer.trainOnInstance(instNoTimeStamp);
		currentClusters = clusterer.getClusteringResult();		
		addNewCCLearners(currentClusters.size() - currentSize);
		
		// note that the index of the cluster to which this instance belong is not necessarily the index of a newly added cluster, even
		// if this instance resulted in the addition of a new cluster.
		int indexCluster = indexClusterInstanceBelongsTo(currentClusters, instNoTimeStamp);
		
		// SOME CLUSTERERS WORK WITH WINDOWS, AND MAY HAVE BEEN TRAINED ONLY AFTER A CERTAIN NUMBER OF CC PROJECTS WERE RECEIVED.
		// SO, AT THIS POINT HERE, THERE MAY BE NO CLUSTERS YET AND INDEXCLUSTER WILL BE -1.
		// SO, NEEDS TO STORE SOME OF THE CC INSTANCES TO USE FOR TRAINING ONLY ONCE CLUSTERS HAVE BEEN CREATED.
		if (indexCluster != -1) {
			
			if (!learners.get(indexCluster).trainingHasStarted() && timeStep != 0) // <---le18Aug2017
				weights.set(indexCluster, 1.0/getNumTrainedLearners()); // <---le18Aug2017 Any learner that is trained for the first time gets its weight set to this value
			else if (!learners.get(indexCluster).trainingHasStarted() && timeStep == 0) // <---le18Aug2017
				weights.set(indexCluster, 1.0/weights.size()); // <---le18Aug2017
			
			learners.get(indexCluster).trainOnInstance(instNoTimeStamp);
			
			// Check if any instances waiting to be used for training should be used for training learners now.
			// They have already been sent to train the clusterer, so no need to send them again.
			for (int i=0; i< ccInstancesWaitingForWindowTraining.size(); ++i) {
				instNoTimeStamp = ccInstancesWaitingForWindowTraining.get(i);
				indexCluster = indexClusterInstanceBelongsTo(currentClusters, instNoTimeStamp);
				if (indexCluster != -1) {
					learners.get(indexCluster).trainOnInstance(instNoTimeStamp);
					ccInstancesWaitingForWindowTraining.remove(i);
					--i;
				}
			}
		}
		else ccInstancesWaitingForWindowTraining.add(instNoTimeStamp);
		
		// Reinsert attribute to the dataset
		//recoverDeletedAttDataset();
	}
	
	// Train the CC models with their corresponding instances until the timestamp of inst.
	protected void trainCCModels(Instance inst) {
		Instant timestampWCInstance = new Instant((long)inst.value(inst.numAttributes()-1)*1000);
		Instant timestampCCInstance = null;

		// If we have previously stored past CC instances, we need to use them for training, if their time stamps are before that of inst
		while (ccPastInstancesQueue.size() != 0) {
			Instance ccInstance = ccPastInstancesQueue.get(0);
			timestampCCInstance = new Instant((long)ccInstance.value(inst.numAttributes()-1)*1000); 

			// Train on CC instance only if its timestamp is before that of inst
			if (timestampCCInstance.isBefore(timestampWCInstance)) {
				trainOnCCInstance(ccInstance);
				ccPastInstancesQueue.remove(0); // remove that instance, as we won't need to use it for training anymore
			}
			else break;
		}
		

		// if we have used the bk instances and there are still more instances in the CC data stream, check if we can use them for training
		if (ccPastInstancesQueue.size() == 0 && ccDataStream.hasMoreInstances()) {

			while (ccDataStream.hasMoreInstances()) {

				Instance ccInstance = ccDataStream.nextInstance().instance;
				timestampCCInstance = new Instant((long)ccInstance.value(inst.numAttributes()-1)*1000);

				// Train on CC instance only if its timestamp is before that of inst
				if (timestampCCInstance.isBefore(timestampWCInstance)) {
					trainOnCCInstance(ccInstance);
				}
				else {
					ccPastInstancesQueue.add(ccInstance);
					break;
				}

			}  
		}

	}
	
	@Override
	public double[] getVotesForInstance(Instance inst) {
		
		if (!makeChecks(inst)) {
			System.err.println("Halting get votes.");
			return new double[inst.numClasses()];
		}
		
		// Update the CC models with CC training examples up to the timestamp of this instance
		if (wcPastInstancesQueueSize.getValue() != 1|| timeStep == 0) { // TEST<--- added this if to make this like the original dycom
			trainCCModels(inst);
			trainMappingFunctions();
			updateWeights();
		}
				
		double []votes = new double[1]; // this is a regression problem
		
		Instance instNoTimeStamp = inst.copy();
		deleteTimeStamp(instNoTimeStamp);
		
		// Get predictions of the CC mapped models
		for (int i=0; i<learners.size()-1; ++i) {
			if (learners.get(i).trainingHasStarted())
				votes[0] += mappingFunctions.get(i).getVotesForInstance(instNoTimeStamp)[0] * weights.get(i);
		}
		
		// Get prediction on the WC model
		if (learners.get(learners.size()-1).trainingHasStarted())
			votes[0] += learners.get(learners.size()-1).getVotesForInstance(instNoTimeStamp)[0] * weights.get(learners.size()-1);
		
		return votes; 
	
	}





	@Override
	protected Measurement[] getModelMeasurementsImpl() {

		Measurement [] measurement = null;
		Measurement [][]measurementLearners = new Measurement[learners.size()][];

		int numMeasurementLearners = 0;
		for (int i=0; i<learners.size(); ++i) {
			measurementLearners[i] = learners.get(i).getModelMeasurements();
			if (measurementLearners[i] != null)
				numMeasurementLearners += measurementLearners[i].length; 
		}
		
		if (mappingFunctions.size() != 0 && mappingFunctions.get(0) instanceof SimpleLinearMappingFunction)
			measurement = new Measurement[1+numMeasurementLearners+weights.size()+mappingFunctions.size()]; 
		else measurement = new Measurement[1+numMeasurementLearners+weights.size()]; 

		measurement[0] = new Measurement("ensemble size", this.learners != null ? this.learners.size() : 0);
		
		numMeasurementLearners = 0;
		for (int i=0; i<learners.size(); ++i) {
			if (measurementLearners[i] != null) {
				for (int j=0; j<measurementLearners[i].length;++j) {
					measurement[numMeasurementLearners+1] = new Measurement("learner " + i + "'s " + measurementLearners[i][j].getName(), measurementLearners[i][j].getValue());
					numMeasurementLearners++;
				}
			}
		}
		for (int i=0; i<weights.size(); ++i)
			measurement[i+numMeasurementLearners+1] = new Measurement("weight " + i, this.weights.get(i));
		
		if (mappingFunctions.size() != 0 && mappingFunctions.get(0) instanceof SimpleLinearMappingFunction) {
			for (int i=0; i<mappingFunctions.size(); ++i) 
				measurement[i+weights.size()+numMeasurementLearners+1] = new Measurement("factor b " + i, ((SimpleLinearMappingFunction)mappingFunctions.get(i)).getB()); 
		}
		
		return measurement;
		
		//return new Measurement[]{new Measurement("ensemble size",
        //        this.learners != null ? this.learners.size() : 0)};
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub

	}

}
