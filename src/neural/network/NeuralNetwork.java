package neural.network;

import java.lang.reflect.Array;
import java.util.Random;

public class NeuralNetwork 
{

	private int numInput;
	private int numHidden;
	private int numOutput;
	
	private static Random rnd;
	
	private double[] inputs;

	private double[][] ihWeights; // input-hidden
	private double[] hBiases;
	private double[] hOutputs;
	
	private double[][] hoWeights; // hidden-output
    private double[] oBiases;

    private double[] outputs;

    // back-prop specific arrays (these could be local to method UpdateWeights)
    private double[] oGrads; // output gradients for back-propagation
    private double[] hGrads; // hidden gradients for back-propagation

    // back-prop momentum specific arrays (these could be local to method Train)
    private double[][] ihPrevWeightsDelta;  // for momentum with back-propagation
    private double[] hPrevBiasesDelta;
    private double[][] hoPrevWeightsDelta;
    private double[] oPrevBiasesDelta;
	
	public NeuralNetwork(int numInput, int numHidden, int numOutput)
	{
		rnd = new Random(0);
		
		this.numInput = numInput;
		this.numHidden = numHidden;
		this.numOutput = numOutput;
		
		this.inputs = new double[numInput];

	    this.ihWeights = makeMatrix(numInput, numHidden);
	    this.hBiases = new double[numHidden];
	    this.hOutputs = new double[numHidden];
	    
	    this.hoWeights = makeMatrix(numHidden, numOutput);
	    this.oBiases = new double[numOutput];

	    this.outputs = new double[numOutput];

	    // back-prop related arrays below
	    this.hGrads = new double[numHidden];
	    this.oGrads = new double[numOutput];

	    this.ihPrevWeightsDelta = makeMatrix(numInput, numHidden);
	    this.hPrevBiasesDelta = new double[numHidden];
	    this.hoPrevWeightsDelta = makeMatrix(numHidden, numOutput);
	    this.oPrevBiasesDelta = new double[numOutput];
	}
	
	static attData makeTrainTest(double[][] allData, attData data)
    {
		double[][] trainData = null;
		double[][] testData = null;
		
		
      // split allData into 80% trainData and 20% testData
      Random rnd = new Random(0);
      int totRows = allData.length;
      int numCols = allData[0].length;

      int trainRows = (int)(totRows * 0.80); // hard-coded 80-20 split
      int testRows = totRows - trainRows;

      trainData = new double[trainRows][];
      testData = new double[testRows][];

      int[] sequence = new int[totRows]; // create a random sequence of indexes
      for (int i = 0; i < sequence.length; ++i)
        sequence[i] = i;

      for (int i = 0; i < sequence.length; ++i)
      {
        int r = rnd.nextInt(sequence.length - i) + i;
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }

      int si = 0; // index into sequence[]
      int j = 0; // index into trainData or testData

      for (; si < trainRows; ++si) // first rows to train data
      {
        trainData[j] = new double[numCols];
        int idx = sequence[si];
        System.arraycopy(allData[idx], 0, trainData[j], 0, numCols);
        ++j;
      }

      j = 0; // reset to start of test data
      for (; si < totRows; ++si) // remainder to test data
      {
        testData[j] = new double[numCols];
        int idx = sequence[si];
        System.arraycopy(allData[idx], 0, testData[j], 0, numCols);
        ++j;
      }
      data.setData(trainData, testData);
      return data;
    } // MakeTrainTest
	
	public void initWeights()
	{
		int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
		double[] initWeights = new double[numWeights];
		double lo = -0.01;
		double hi = 0.01;
		for (int i=0; i < initWeights.length ; ++i)
			initWeights[i] = (hi - lo) * rnd.nextDouble() + lo;
		this.setWeights(initWeights);
	}
	
	public void setWeights(double[] weights)
	{
		int k = 0;
		
		for (int i=0; i < numInput; ++i)
			for(int j=0; j < numHidden; ++j)
				ihWeights[i][j] = weights[k++];
		for (int i=0; i < numHidden; ++i)
			hBiases[i] = weights[k++];
		for(int i=0; i < numHidden; ++i)
			for(int j=0; j < numOutput; ++j)
				hoWeights[i][j] = weights[k++];
		for (int i=0; i < numOutput; ++i)
			oBiases[i] = weights[k++];
	}
	
	
	private static double[][] makeMatrix(int rows, int cols)
    {
      double[][] result = new double[rows][];
      for (int r = 0; r < result.length; ++r)
        result[r] = new double[cols];
      return result;
    }

	public void train(double[][] trainData, int maxEpochs, double learnRate,double momentum) 
	{
		//train a backpropagation style NN classifier using learning rate + momentum
		//no weight decay
		
		int epoch = 0;
		double[] xValues = new double[numInput];
		double[] tValues = new double[numOutput];
		
		int[] sequence = new int[trainData.length];
		for (int i=0; i < sequence.length; ++i)
			sequence[i] = i;
		
		while (epoch < maxEpochs)
		{
			double mse = meanSquaredError(trainData);
			if (mse < 0.020) break; // consider passing value in as parameter
			//if (mse < 0.001) break; // consider passing value in as parameter
			
			shuffle(sequence); // visit each training data in random order 
			
			for (int i=0; i < trainData.length; ++i)
			{
				int idx = sequence[i];
				System.arraycopy(trainData[idx], 0, xValues, 0, numInput); // extract x's and y's.
		        System.arraycopy(trainData[idx], numInput, tValues, 0, numOutput);
		        computeOutputs(xValues); // copy xValues in, compute outputs (and store them internally)
		        updateWeights(tValues, learnRate, momentum); // use back-prop to find better weights
			}//each training tuple
			++epoch;
		}
	}

	private void updateWeights(double[] tValues, double learnRate,double momentum) 
	{
		// update the weights and biases using back-propagation, with target values, eta (learning rate),
	      // alpha (momentum)
	      // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays and
	      // matrices have values (other than 0.0)
	      
	      // 1. compute output gradients
	      for (int i = 0; i < oGrads.length; ++i)
	      {
	        // derivative of softmax = (1 - y) * y (same as log-sigmoid)
	        double derivative = (1 - outputs[i]) * outputs[i]; 
	        // 'mean squared error version'. research suggests cross-entropy is better here . . .
	        oGrads[i] = derivative * (tValues[i] - outputs[i]); 
	      }

	      // 2. compute hidden gradients
	      for (int i = 0; i < hGrads.length; ++i)
	      {
	        double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]); // derivative of tanh = (1 - y) * (1 + y)
	        double sum = 0.0;
	        for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
	        {
	          double x = oGrads[j] * hoWeights[i][j];
	          sum += x;
	        }
	        hGrads[i] = derivative * sum;
	      }

	      // 3a. update hidden weights (gradients must be computed right-to-left but weights
	      // can be updated in any order)
	      for (int i = 0; i < ihWeights.length; ++i) // 0..2 (3)
	      {
	        for (int j = 0; j < ihWeights[0].length; ++j) // 0..3 (4)
	        {
	          double delta = learnRate * hGrads[j] * inputs[i]; // compute the new delta
	          ihWeights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
	          // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
	          ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j]; 
	          // weight decay would go here
	          ihPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum 
	        }
	      }

	      // 3b. update hidden biases
	      for (int i = 0; i < hBiases.length; ++i)
	      {
	        // the 1.0 below is the constant input for any bias; could leave out
	        double delta = learnRate * hGrads[i] * 1.0; 
	        hBiases[i] += delta;
	        hBiases[i] += momentum * hPrevBiasesDelta[i]; // momentum
	        // weight decay here
	        hPrevBiasesDelta[i] = delta; // don't forget to save the delta
	      }

	      // 4. update hidden-output weights
	      for (int i = 0; i < hoWeights.length; ++i)
	      {
	        for (int j = 0; j < hoWeights[0].length; ++j)
	        {
	          // see above: hOutputs are inputs to the nn outputs
	          double delta = learnRate * oGrads[j] * hOutputs[i];  
	          hoWeights[i][j] += delta;
	          hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j]; // momentum
	          // weight decay here
	          hoPrevWeightsDelta[i][j] = delta; // save
	        }
	      }

	      // 4b. update output biases
	      for (int i = 0; i < oBiases.length; ++i)
	      {
	        double delta = learnRate * oGrads[i] * 1.0;
	        oBiases[i] += delta;
	        oBiases[i] += momentum * oPrevBiasesDelta[i]; // momentum
	        // weight decay here
	        oPrevBiasesDelta[i] = delta; // save
	      }
	}

	private void shuffle(int[] sequence) 
	{
		for (int i=0; i < sequence.length; ++i)
		{
			int r = rnd.nextInt(sequence.length - i) + i;
			int tmp = sequence[r];
			sequence[r] = sequence[i];
			sequence[i] = tmp;
		}
	}

	private double meanSquaredError(double[][] trainData) 
	{
		//average squared error per training tuple
		double sumSquaredError = 0.0;
		double[] xValues = new double[numInput]; // first numInput values in trainData
	    double[] tValues = new double[numOutput]; // last numOutput values
	    
	    for (int i=0; i < trainData.length; ++i){
	    	// walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
	        //  where the parens are not really there
	        System.arraycopy(trainData[i], 0, xValues, 0, numInput); // get xValues.
	        System.arraycopy(trainData[i], numInput, tValues, 0, numOutput); // get target values
	        double[] yValues = this.computeOutputs(xValues); // compute output using current weights
	        for (int j = 0; j < numOutput; ++j)
	        {
	          double err = tValues[j] - yValues[j];
	          sumSquaredError += err * err;
	        }
	    }
	    
		return sumSquaredError / trainData.length;
	}

	private double[] computeOutputs(double[] xValues) 
	{
		double[] hSums = new double[numHidden];
		double[] oSums = new double[numOutput];
		
		for (int i = 0; i < xValues.length; ++i) // copy x-values to inputs
	        this.inputs[i] = xValues[i];

	    for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
	    	for (int i = 0; i < numInput; ++i)
	          hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

	    for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
	        hSums[i] += this.hBiases[i];

	    for (int i = 0; i < numHidden; ++i)   // apply activation
	        this.hOutputs[i] = hyperTanFunction(hSums[i]); // hard-coded

	    for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
	    	for (int i = 0; i < numHidden; ++i)
	    		oSums[j] += hOutputs[i] * hoWeights[i][j];

	    for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
	    	oSums[i] += oBiases[i];
		
	    double[] softOut = softmax(oSums); // softmax activation does all outputs at once for efficiency
	    System.arraycopy(softOut, 0, outputs, 0, softOut.length);

	    double[] retResult = new double[numOutput]; // could define a GetOutputs method instead
	    System.arraycopy(this.outputs, 0, retResult, 0, retResult.length);
	    return retResult;
	}

	private double hyperTanFunction(double x) 
	{
		if (x < - 20.0) return -1.0; // approx correct to 30 decimals 
		else if (x > 20.0) return 1.0;
		else return Math.tanh(x);
	}
	
	private double[] softmax(double[] oSums) 
	{
		// does all output nodes at once so scale doesn't have to be re-computed each time
	    // 1. determine max output sum
		double max = oSums[0];
	    for (int i = 0; i < oSums.length; ++i)
	      if (oSums[i] > max) max = oSums[i];

	    // 2. determine scaling factor -- sum of exp(each val - max)
	    double scale = 0.0;
	    for (int i = 0; i < oSums.length; ++i)
	    	scale += Math.exp(oSums[i] - max);

	    double[] result = new double[oSums.length];
	    for (int i = 0; i < oSums.length; ++i)
	    	result[i] = Math.exp(oSums[i] - max) / scale;

	    return result; // now scaled so that xi sum to 1.0
	}
	
	
	 public double[] getWeights()
	 {
		 // returns the current set of wweights, presumably after training
		 int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
	     double[] result = new double[numWeights];
	     int k = 0;
	     for (int i = 0; i < ihWeights.length; ++i)
	    	 for (int j = 0; j < ihWeights[0].length; ++j)
	    		 result[k++] = ihWeights[i][j];
	     for (int i = 0; i < hBiases.length; ++i)
	    	 result[k++] = hBiases[i];
	     for (int i = 0; i < hoWeights.length; ++i)
	    	 for (int j = 0; j < hoWeights[0].length; ++j)
	    		 result[k++] = hoWeights[i][j];
	     for (int i = 0; i < oBiases.length; ++i)
	    	 result[k++] = oBiases[i];
	     return result;
	 }
	 
	 public double accuracy(double[][] testData)
	    {
	      // percentage correct using winner-takes all
	      int numCorrect = 0;
	      int numWrong = 0;
	      double[] xValues = new double[numInput]; // inputs
	      double[] tValues = new double[numOutput]; // targets
	      double[] yValues; // computed Y

	      for (int i = 0; i < testData.length; ++i)
	      {
	        System.arraycopy(testData[i], 0, xValues, 0, numInput); // parse test data into x-values and t-values
	        System.arraycopy(testData[i], numInput, tValues, 0, numOutput);
	        yValues = this.computeOutputs(xValues);
	        int maxIndex = maxIndex(yValues); // which cell in yValues has largest value?

	        if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
	          ++numCorrect;
	        else
	          ++numWrong;
	      }
	      return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
	    }
	 
	 	private static int maxIndex(double[] vector) // helper for Accuracy()
	    {
	      // index of largest value
	      int bigIndex = 0;
	      double biggestVal = vector[0];
	      for (int i = 0; i < vector.length; ++i)
	      {
	        if (vector[i] > biggestVal)
	        {
	          biggestVal = vector[i]; bigIndex = i;
	        }
	      }
	      return bigIndex;
	    }
	
}








