/**
 * @author Leandro L. Minku, University of Birmingham, l.l.minku@cs.bham.ac.uk
 * 
 * This class implements a simple linear mapping function for use with OATES.
 * This mapping function is the same as the one published in the following paper:
 * 
 * MINKU, L. L.; YAO, X.; "How to Make Best Use of Cross-company Data in Software Effort Estimation?", Proceedings of the 36th International Conference on Software Engineering (ICSE'14), p. 446-456, May 2014, doi: 10.1145/2568225.2568228. 
 * 
 * However, note that the experiments run in that paper did not use MOA. They used the WEKA implementation.
 * This MOA implementation is a new implementation created after the ICSE'14 paper.
 * 
 * This works only for regression problems.
 * 
 */

package moa.classifiers.meta;

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.Measurement;

public class SimpleLinearMappingFunction extends AbstractClassifier implements MappingFunction {

	private static final long serialVersionUID = 1L;

	public FloatOption learningRate = new FloatOption("learningRate", 'r', 
			"Learning rate of the adjustment factor of the multi-company learners predictions (between 0 and 1).", 0.1, 0.0, 1.0);

	protected double b;
	protected int trainingInstancesSeenByModel;

	// This is the cc learner whose predictions should be mapped to the WC context.
	private Classifier ccLearner;

	public SimpleLinearMappingFunction() {
		super();
	}

	@Override
	public void resetLearningImpl() {
		b = 1.0;
		trainingInstancesSeenByModel = 0;
		ccLearner = null;
	}

	public double getB() {
		return b;
	}
	
	public Classifier getCCLearner() {
		return ccLearner;
	}
	
	// This method should be called before starting to use an object of this class for training or predictions.
	public void setCCLearner(Classifier ccLearner) {
		this.ccLearner = ccLearner;
	}

	@Override
	public boolean isRandomizable() {
		return false;
	}

	// Checks whether this learner is properly set for learning and making predictions for the given instance
	public boolean makeChecks(Instance inst) {
		if (ccLearner == null) {
			System.err.println("Error: trying to use mapping function without having set ccLearner.");
			return false;
		}	

		if (inst.classAttribute().isNominal()) {
			System.err.println("Error: trying to use SimpleLinearMappingFunction for classification. It can only be used for regression.");
			return false;
		}

		return true;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {

		if (!makeChecks(inst)) {
			System.err.println("Halting get votes.");
			return new double[inst.numClasses()];
		}

		double []votes = ccLearner.getVotesForInstance(inst);
		votes[0] *= b;

		return votes;
	}
	
	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (!makeChecks(inst)) {
			System.err.println("Halting training.");
			return;
		}

		double []ccVotes = ccLearner.getVotesForInstance(inst);

		if (trainingInstancesSeenByModel != 0) {
			if (ccVotes[0] != 0)
				b = (1 - learningRate.getValue()) * b + learningRate.getValue() * inst.classValue() / ccVotes[0];
			else //if (inst.classValue() == 0)
				b = (1 - learningRate.getValue()) * b + learningRate.getValue() * 1.0;
			//else
			//	b = (1 - learningRate.getValue()) * b + learningRate.getValue() * 0.0;
		}
		else {
			if (ccVotes[0] != 0)
				b = inst.classValue() / ccVotes[0];
			else 
				b = 1.0;
		}
		
		trainingInstancesSeenByModel++;
		
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub

	}

}
