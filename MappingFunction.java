/**
 * @author Leandro L. Minku, University of Birmingham, l.l.minku@cs.bham.ac.uk
 * 
 * Interface for mapping functions used by OATES. All mapping functions should implement this interface.
 */

package moa.classifiers.meta;

import moa.classifiers.Classifier;
import moa.classifiers.Regressor;

public interface MappingFunction extends Regressor {

	// This is the CC learner whose predictions will be mapped to the WC context.
	public void setCCLearner(Classifier ccLearner);
	
	public Classifier getCCLearner();
}
