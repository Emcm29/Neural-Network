package neural.network;

// All Train Test Data
public class attData {
	double[][] allData = null;
	double[][] trainData = null;
    double[][] testData = null;
    
    public attData(double[][] allData){
    	this.allData = allData;
    }
    
    public void setData(double[][] trainData, double[][] testData){
    	this.trainData = trainData;
    	this.testData = testData;
    }
    
    public double[][] getAllData(){
    	return allData;
    }
    
    public double[][] getTrainData(){
    	return trainData;
    }
    
    public double[][] getTestData(){
    	return testData;
    }
}
