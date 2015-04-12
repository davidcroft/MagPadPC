package magPadReference;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;

import processing.core.*;
import oscP5.*;
import netP5.*;

public class ProcessingSketch extends PApplet {
	private static final long serialVersionUID = 1L;
	
	private int fftCnt = 0;
	private int fftIndex = 0;
	private boolean startFFT = false;
	
	// OSC
	public OscP5 oscP5;
	public NetAddress remoteLocation;

	// Buffers
	public Buffer magDataBuf;
	
	// Fir filters group
	public FIRFilterGroup filterGroup;

	// Thread
	public FFTThread fftThread;
	public FFTThreadCls fftThreadCls;
	
	// SVM regression
	WekaClassifier SVMClassfier;
	WekaRegression[] SVMRegressionRow;
	WekaRegression[] SVMRegressionCol;
	
	private boolean isTrainedCls = false;
	private boolean isTrainedReg = false;
	private TrainingHashTable trainTable;
	private int isTrainingIndex = -2;	// default value: -2; train classifier: -1; train blocks: 0-3;
	private ArrayList<Integer> unTrainedList = new ArrayList<Integer>();
	
	public void setup() {
		// OSC
		// start oscP5, telling it to listen for incoming messages at port 5001 */
		OscProperties properties = new OscProperties();
		properties.setListeningPort(GlobalConstants.RECVPORT);
		properties.setDatagramSize(102400);
		oscP5 = new OscP5(this,properties);
		// set the remote location to be the localhost on port 3001
		remoteLocation = new NetAddress(GlobalConstants.SENDHOST,GlobalConstants.SENDPORT);
		
		// MagBuffer
		// init buffers
		magDataBuf = new Buffer(GlobalConstants.BUFFERSIZE);
		
		// init fir filters
		filterGroup = new FIRFilterGroup();
		
		////////////////////////////////////
		// SVM classifier and regression model
		// train/load csv classifier
		String modelPathClassifier = Paths.get("files", "trainingSetClassifier.csv.model").toAbsolutePath().toString();
		String modelPathClassifierSet = Paths.get("files", "trainingSetClassifier.csv").toAbsolutePath().toString();
		File fClass = new File(modelPathClassifier);
		File sClass = new File(modelPathClassifierSet);
		if(fClass.exists() && !fClass.isDirectory()) {
			// load models
			SVMClassfier = new WekaClassifier(modelPathClassifierSet, modelPathClassifier);
			// set isTrainedCls
			isTrainedCls = true;
		} else if (sClass.exists() && !sClass.isDirectory()){
			// load training set and train model
			SVMClassfier = new WekaClassifier(modelPathClassifierSet);
			// set isTrainedCls
			isTrainedCls = true;
		} else {
			// set isTrainedCls
			isTrainedCls = false;
			isTrainingIndex = -1;
		}
		System.out.println("isTrainedCls = "+(isTrainedCls?"true":"false"));
		
		
		// train/load csv regression array
		SVMRegressionRow = new WekaRegression[GlobalConstants.TRAININGCLSNUM];
		SVMRegressionCol = new WekaRegression[GlobalConstants.TRAININGCLSNUM];
		
		boolean isTrainedRegTemp = true;
		for (int block = 0; block < GlobalConstants.TRAININGCLSNUM; block++) {
			String pathModelRow =  "trainingSetRowB" + block + ".csv.model";
			String pathModelCol =  "trainingSetColB" + block + ".csv.model";
			String pathSetRow =  "trainingSetRowB" + block + ".csv";
			String pathSetCol =  "trainingSetColB" + block + ".csv";
			
			String modelPathRow = Paths.get("files", pathModelRow).toAbsolutePath().toString();
			String modelPathCol = Paths.get("files", pathModelCol).toAbsolutePath().toString();
			String modelPathRowSet = Paths.get("files", pathSetRow).toAbsolutePath().toString();
			String modelPathColSet = Paths.get("files", pathSetCol).toAbsolutePath().toString();
			
			// check if model has been trained in the system
			File fRow = new File(modelPathRow);
			File fCol = new File(modelPathCol);
			File sRow = new File(modelPathRowSet);
			File sCol = new File(modelPathColSet);
			
			if(fRow.exists() && !fRow.isDirectory() && fCol.exists() && !fCol.isDirectory()) {
				// load models	
				SVMRegressionRow[block] = new WekaRegression(Paths.get("files", pathSetRow).toAbsolutePath().toString(), 
					Paths.get("files", pathModelRow).toAbsolutePath().toString());
				SVMRegressionCol[block] = new WekaRegression(Paths.get("files", pathSetCol).toAbsolutePath().toString(), 
					Paths.get("files", pathModelCol).toAbsolutePath().toString());
			} else if (sRow.exists() && !sRow.isDirectory() && sCol.exists() && !sCol.isDirectory()){
				SVMRegressionRow[block] = new WekaRegression(Paths.get("files", pathSetRow).toAbsolutePath().toString()); 
				SVMRegressionCol[block] = new WekaRegression(Paths.get("files", pathSetCol).toAbsolutePath().toString());
			} else {
				// set isTrainedReg
				isTrainedRegTemp = false;
				// add index into unTrainedList
				unTrainedList.add(block);
				isTrainingIndex = block;
			}
		}
		isTrainedReg = isTrainedRegTemp;
		System.out.println("isTrainedReg = "+(isTrainedReg?"true":"false"));
		if (!isTrainedReg) {
			System.out.print("unTrainedBlocks: ");
			for (int i = 0; i < unTrainedList.size(); i++) {
				System.out.print(unTrainedList.get(i).intValue()+", ");
			}
			System.out.print("\n");
		}
		////////////////////////////////
		
		// init hashtable
		trainTable = new TrainingHashTable();
	    size(displayWidth,displayHeight);
	}

	public void draw() {
	}
	
	public void mousePressed() {
		if (mouseButton == LEFT) {
			// start doing fft
			fftIndex++;
			fftCnt = 0;
			if (isTrainingIndex == -1) {
				if (fftIndex <= GlobalConstants.TRAININGCLSNUM) {
					startFFT = true;
					System.out.println("CLS: start FFT at position " + fftIndex);
				} else {
					startFFT = false;
				}
			} else if (isTrainingIndex >= 0 && isTrainingIndex < GlobalConstants.TRAININGCLSNUM) {
				if (fftIndex <= trainTable.getBlockTrainNumSum(isTrainingIndex)) {
					startFFT = true;
					System.out.println("REG: start FFT at position " + fftIndex);
				} else {
					startFFT = false;
				}
			} else {
				startFFT = false;
			}
		} else if (mouseButton == RIGHT) {
		    // stop doing fft
			startFFT = false;
			System.out.println("stop FFT");
		}
	}
	
	void oscEvent(OscMessage theOscMessage) {
		// osc data
	    float[] bufX = new float[GlobalConstants.BUFFERSIZE];
	    float[] bufY = new float[GlobalConstants.BUFFERSIZE];
	    float[] bufZ = new float[GlobalConstants.BUFFERSIZE];
	    
	    // apply filter
	    for (int i = 0; i < GlobalConstants.BUFFERSIZE; i++) {
	    	bufX[i] = filterGroup.applyToFilter(theOscMessage.get(i*3+0).floatValue(), 0);
	    	bufY[i] = filterGroup.applyToFilter(theOscMessage.get(i*3+1).floatValue(), 1);
	    	bufZ[i] = filterGroup.applyToFilter(theOscMessage.get(i*3+2).floatValue(), 2);
	    }
	    
	    magDataBuf.addToBuffer(bufX, bufY, bufZ);
	    
	    // identify if model has been trained
	    if (isTrainedCls && isTrainedReg) {
		    // TESTING
		    // axis par: 1 for x axis, 2 for y axis and 3 for z axis
		    float[] dataX = magDataBuf.genBufferForFFT(magDataBuf.m_bufferIndex, 1);
		    float[] dataY = magDataBuf.genBufferForFFT(magDataBuf.m_bufferIndex, 2);
		    float[] dataZ = magDataBuf.genBufferForFFT(magDataBuf.m_bufferIndex, 3);
		    // fftIndex: -1 testing, >=0 training (and fftIndex used for identify training index)
		    Pair<Float, Float> loc = Pair.createPair((float)-1.0, (float)-1.0);
		    bufferFFT(dataX, dataY, dataZ, loc);
		    
		    // predict
		    if (!GlobalConstants.testingSet.isEmpty()) {
		    	float[] testDataRow = GlobalConstants.testingSet.get(0);
		    	GlobalConstants.testingSet.remove(0);
		    	// get classification first
		    	int locClass = SVMClassfier.classifyGesture(testDataRow);
		    	System.out.println("predict block class: " + locClass);
		    	// predict location using corresponding regression
		    	double predictLocRow = SVMRegressionRow[locClass].classifyGesture(testDataRow);
		    	double predictLocCol = SVMRegressionCol[locClass].classifyGesture(testDataRow);
		    	
		    	// normalize location to A4 size
		    	predictLocRow = predictLocRow * GlobalConstants.MAXHEIGHT;
		    	predictLocCol = predictLocCol * GlobalConstants.MAXWIDTH;
		    	System.out.println("predict location: " + predictLocRow + " " + predictLocCol);
		    	
		    	// send a OSC message
	    		OscMessage myMessage = new OscMessage("/loc");
	    		myMessage.add((float)predictLocRow);
	    		myMessage.add((float)predictLocCol);
	    		oscP5.send(myMessage, remoteLocation);
		    }
	    }
	    
	    if (!isTrainedReg) {
	    	if (!unTrainedList.isEmpty()) {
	    		if (startFFT) {
	    			isTrainingIndex = unTrainedList.get(0).intValue();
		    		// fft
				    float[] dataX = magDataBuf.genBufferForFFT(magDataBuf.m_bufferIndex, 1);
				    float[] dataY = magDataBuf.genBufferForFFT(magDataBuf.m_bufferIndex, 2);
				    float[] dataZ = magDataBuf.genBufferForFFT(magDataBuf.m_bufferIndex, 3);
				    Pair<Float, Float> loc = Pair.createPair(trainTable.getReg(isTrainingIndex, fftIndex).getX(), trainTable.getReg(isTrainingIndex, fftIndex).getY());
				    bufferFFT(dataX, dataY, dataZ, loc);
				    if (fftCnt++ >= GlobalConstants.FFTFOREACHPOS) {
					    // stop fft
					    startFFT = false;
					    System.out.println("Bock " + isTrainingIndex + " Pos " + fftIndex + " FFT collection finished!");
				    }
	    		}
	    		
			    int trainNumSum = trainTable.getBlockTrainNumSum(isTrainingIndex);
	    		if (fftIndex > trainNumSum) {
			    	fftIndex = 0;
			    	startFFT = false;
			    	// save training set to local file
			    	generateCsvFile(isTrainingIndex);
			    	// train model
			    	System.out.println("train SVM for block " + isTrainingIndex + " ...");
			    	String pathSetRow =  "trainingSetRowB" + isTrainingIndex + ".csv";
					String pathSetCol =  "trainingSetColB" + isTrainingIndex + ".csv";
			    	SVMRegressionRow[isTrainingIndex] = new WekaRegression(pathSetRow);
					SVMRegressionCol[isTrainingIndex] = new WekaRegression(pathSetCol);
			    	System.out.println("SVM training for block " + isTrainingIndex + " finished");

			    	// remove from unTrainedList
			    	unTrainedList.remove(0);
			    	if (unTrainedList.isEmpty()) {
			    		// set isTrainedReg
			    		isTrainedReg = true;
			    		isTrainingIndex = -2;
			    		System.out.println("all blocks training finished");
			    	} else {
			    		System.out.print("unTrainedBlocks: ");
						for (int i = 0; i < unTrainedList.size(); i++) {
							System.out.print(unTrainedList.get(i).intValue()+", ");
						}
						System.out.print("\n");
			    	}
			    }
	    	}
	    }
	    
	    // regression training has higher priority
	    // train classification classifier
	    if (isTrainedReg && !isTrainedCls){
		    // TRAIN CLASSIFICATION
	    	isTrainingIndex = -1;
		    // axis par: 1 for x axis, 2 for y axis and 3 for z axis
		    // begin fft after left click mouse
		    if (startFFT) {
			    float[] dataX = magDataBuf.genBufferForFFT(magDataBuf.m_bufferIndex, 1);
			    float[] dataY = magDataBuf.genBufferForFFT(magDataBuf.m_bufferIndex, 2);
			    float[] dataZ = magDataBuf.genBufferForFFT(magDataBuf.m_bufferIndex, 3);
			    bufferFFTCls(dataX, dataY, dataZ, trainTable.getCls(fftIndex), fftCnt);
			    if (fftCnt++ >= GlobalConstants.FFTCLSFOREACHPOS) {
				    // stop fft
				    startFFT = false;
				    System.out.println("Class " + fftIndex + " FFT collection finished!");
			    }
		    }

		    // train model in interrupt handler, might have some problem
		    if (fftIndex > GlobalConstants.TRAININGCLSNUM) {
		    	fftIndex = -1;
		    	startFFT = false;
		    	// save training set to local file
		    	generateClsCsvFile();
		    	// train model
		    	System.out.println("train SVM for classification ...");
		    	SVMClassfier = new WekaClassifier(Paths.get("files", "trainingSetClassifier.csv").toAbsolutePath().toString());
		    	System.out.println("SVM classification training finished");
		    	
		    	// set isTrainedCls and fftIndex
		    	isTrainedCls = true;
		    	System.out.println("set isTrainedCls to be true");
		    }
 	    }
	}

	void bufferFFT(float[] fftDataX, float[] fftDataY, float[] fftDataZ, Pair<Float, Float> location) {
		fftThread = new FFTThread(this, 10, GlobalConstants.BUFFERSIZE * Buffer.BUFFERNUM, GlobalConstants.SAMPLERATE, fftDataX, fftDataY, fftDataZ, location);
		fftThread.start();
	}
	
	void bufferFFTCls(float[] fftDataX, float[] fftDataY, float[] fftDataZ, String label, int fftCnt) {
		fftThreadCls = new FFTThreadCls(this, 10, GlobalConstants.BUFFERSIZE * Buffer.BUFFERNUM, GlobalConstants.SAMPLERATE, fftDataX, fftDataY, fftDataZ, label, fftCnt);
		fftThreadCls.start();
	}
	
	void generateClsCsvFile() {
		try
		{
			// row file
			FileWriter writer = new FileWriter(Paths.get("files", "trainingSetClassifier.csv").toAbsolutePath().toString());
		    String title = "";
		    for(int i = 1; i <= GlobalConstants.NNINPUTNUM-18; i++) {
		    	title += "input "+i+",";
		    }
		    // define other feature titles
		    title += "meanX,meanY,meanZ,meanDiffXY,meanDiffYZ,meanDiffZX,maxIndexX,maxValX,maxIndexY,maxValY,maxIndexZ,maxValZ,varX,varY,varZ,kurtosisX,kurtosisY,kurtosisZ,label\n";
		    
		    writer.append(title);
		    for(int i = 0; i < GlobalConstants.trainingClsSet.size(); i++){
		    	writer.append(GlobalConstants.trainingClsSet.get(i));
		    }
		    writer.flush();
		    writer.close();
		    
		    // clear trainingClsSet
		    GlobalConstants.trainingClsSet.clear();
		}
		catch(IOException e)
		{
		    e.printStackTrace();
		}
	}
	
	void generateCsvFile(int index) {
		try
		{	
			String pathSetRow =  "trainingSetRowB" + index + ".csv";
			String pathSetCol =  "trainingSetColB" + index + ".csv";
			// row file
			FileWriter writer = new FileWriter(Paths.get("files", pathSetRow).toAbsolutePath().toString());
		    String title = "";
		    for(int i = 1; i <= GlobalConstants.NNINPUTNUM-18; i++) {
		    	title += "input "+i+",";
		    }
		    // define other feature titles
		    title += "meanX,meanY,meanZ,meanDiffXY,meanDiffYZ,meanDiffZX,maxIndexX,maxValX,maxIndexY,maxValY,maxIndexZ,maxValZ,varX,varY,varZ,kurtosisX,kurtosisY,kurtosisZ,label\n";
		    writer.append(title);
		    for(int i = 0; i < GlobalConstants.trainingRowSet.size(); i++){
		    	writer.append(GlobalConstants.trainingRowSet.get(i));
		    }
		    writer.flush();
		    writer.close();
		    
		    
		    // col file
		    writer = new FileWriter(Paths.get("files", pathSetCol).toAbsolutePath().toString());
		    writer.append(title);
		    for(int i = 0; i < GlobalConstants.trainingColSet.size(); i++){
		    	writer.append(GlobalConstants.trainingColSet.get(i));
		    }
		    writer.flush();
		    writer.close();
		    
		    // clear trainingRowSet and trainingColSet
		    GlobalConstants.trainingRowSet.clear();
		    GlobalConstants.trainingColSet.clear();
		}
		catch(IOException e)
		{
		    e.printStackTrace();
		}
	}
	
	public static void main(String args[]) {
	    PApplet.main(new String[] { "--present", "magPadReference.ProcessingSketch" });
	}
}
