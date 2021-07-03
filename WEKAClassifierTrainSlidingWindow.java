
/**
 * Author: Leandro L. Minku, University of Leicester, leandro.minku@leicester.ac.uk
 * 
 * Change in the WEKAClassifier in order to train on a sliding window of examples if the WEKA classifier.
 * If the window is not full yet, at each time step, will train on all examples seen so far. 
 * 
 * The WEKAClassifier will train non-updateable WEKA classifiers as chunk-based learners.
 */

package moa.classifiers.meta;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;
import moa.core.Measurement;
import moa.options.WEKAClassOption;
import weka.classifiers.Classifier;

public class WEKAClassifierTrainSlidingWindow  
extends AbstractClassifier 
implements Regressor {

	private static final long serialVersionUID = 1L;

	protected SamoaToWekaInstanceConverter instanceConverter;

	public WEKAClassOption baseLearnerOption = new WEKAClassOption("baseLearner", 'l',
			"Classifier to train.", weka.classifiers.Classifier.class, "weka.classifiers.bayes.NaiveBayesUpdateable");

	public IntOption widthOption = new IntOption("width",
			'w', "Size of Window for training learner.", 10, 1, Integer.MAX_VALUE);
	
	public IntOption minInstancesForTraining = new IntOption("minInstances",
			'm', "Minimum number of instances to use for allowing to build a classifier before the sliding window is full. Should be smaller or equal to the window width.", 1, 1, Integer.MAX_VALUE);

	protected Classifier classifier;

	protected weka.core.Instances instancesBuffer;
	
	protected boolean isClassificationEnabled;

	@Override
	public String getPurposeString() {
		return "Classifier from Weka trained on a sliding window.";
	}

	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public void resetLearningImpl() {
		resetWekaClassifier();
		this.instanceConverter = new SamoaToWekaInstanceConverter();
		instancesBuffer = null;
		isClassificationEnabled = false;
	}

	protected void resetWekaClassifier() {
		try {
			//System.out.println(baseLearnerOption.getValue());
			String[] options = weka.core.Utils.splitOptions(baseLearnerOption.getValueAsCLIString());
			createWekaClassifier(options);
			isClassificationEnabled = false;
		} catch (Exception e) {
			System.err.println("Creating a new classifier: " + e.getMessage());
		}
	}


	public void createWekaClassifier(String[] options) throws Exception {
		String classifierName = options[0];
		String[] newoptions = options.clone();
		newoptions[0] = "";
		this.classifier = weka.classifiers.AbstractClassifier.forName(classifierName, newoptions);
	}

	@Override
	public void trainOnInstanceImpl(Instance samoaInstance) {
		weka.core.Instance inst = this.instanceConverter.wekaInstance(samoaInstance);
		try {
			if (instancesBuffer == null) 
				instancesBuffer = new weka.core.Instances(inst.dataset());

			if (instancesBuffer.size() < this.widthOption.getValue()) {
				instancesBuffer.add(inst);
			} else {
				instancesBuffer.remove(0);
				instancesBuffer.add(inst);
			}

			if (instancesBuffer.size() >= this.minInstancesForTraining.getValue()) {
				resetWekaClassifier();
				classifier.buildClassifier(instancesBuffer);
				this.isClassificationEnabled = true;
			}

		} catch (Exception e) {
			System.err.println("Training: " + e.getMessage());
		}

	}

	@Override
	public double[] getVotesForInstance(Instance samoaInstance) {
		weka.core.Instance inst = this.instanceConverter.wekaInstance(samoaInstance);
        double[] votes = new double[inst.numClasses()];
        if (isClassificationEnabled == false) {
            for (int i = 0; i < inst.numClasses(); i++) {
                votes[i] = 1.0 / inst.numClasses();
            }
		} else {
			try {
				votes = this.classifier.distributionForInstance(inst);
			} catch (Exception e) {
				System.err.println(e.getMessage());
			}
		}
		return votes;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		Measurement[] m = new Measurement[0];
		return m;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		if (classifier != null) {
			out.append(classifier.toString());
		}

	}




}
