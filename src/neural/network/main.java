package neural.network;

public class main {
	
	public static void main(String[] args) {
		
		attData data = new attData(IrisData.getData());
		data = NeuralNetwork.makeTrainTest(data.getAllData(), data);
		// Data really should be normalized here
		
		int inNodes = 4;
		int hidNodes = 7;
		int outNodes = 3;
		
		NeuralNetwork nn = new NeuralNetwork(inNodes, hidNodes, outNodes);
		nn.initWeights();
		
		int maxEpochs = 4000;
		double learnRate = 0.01;
		double momentum = 0.001;
		
		nn.train(data.getTrainData(), maxEpochs, learnRate, momentum);
		
		double trainAcc = nn.accuracy(data.getTrainData());
		System.out.println(trainAcc);
		
		double testAcc = nn.accuracy(data.getTestData());
		System.out.println(testAcc);
	}
}
