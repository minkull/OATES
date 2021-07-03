/**
 * Author: Leandro L. Minku, University of Leicester (leandro.minku@leicester.ac.uk)
 * 
 * This class applies natural logarithm to any numeric input and output attributes before calling WEKA's classifier.
 * It also converts predictions back to non-logarithmic scale when WEKA is used to make predictions.
 * 
 * It assumes that numeric attributes are >= 0. It sums 1 to all numeric attributes to avoid problems with log(0).
 * 
 * Uses WEKALogClassifier class for achieving the following: 
 * Predictions that, when converted to non-logarithmic scale, are infinite or NaN, are replaced by 1000000.
 * This was for using with software effort estimation. A more appropriate choice could be to use Double.MAX_VALUE
 * 
 */

package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;

public class WEKALogClassifierTrainSlidingWindow extends WEKAClassifierTrainSlidingWindow {

	private static final long serialVersionUID = 1L;

	@Override
    public String getPurposeString() {
        return "Natural Logarithm of Classifier from Weka trained on a sliding window. Warning: numeric attributes must be >= 0.";
    }
	
	public WEKALogClassifierTrainSlidingWindow() {
		super();
	}
	
	@Override
    public void trainOnInstanceImpl(Instance samoaInstance) {
		
		Instance logInstance = WEKALogClassifier.createLogInstance(samoaInstance);
		super.trainOnInstanceImpl(logInstance);
	}
	
	@Override
    public double[] getVotesForInstance(Instance samoaInstance) {
		
		Instance logInstance = WEKALogClassifier.createLogInstance(samoaInstance);
		double[] prediction = super.getVotesForInstance(logInstance);
		
		// de-log the predictions of numerical variables
		// The -1 is needed because we summed 1 to numeric values before applying the log.
		if (samoaInstance.classAttribute().isNumeric()) {
			prediction[0] = WEKALogClassifier.getUnlogPrediction(prediction[0]);
			
		}

		return prediction;		
	}

}
