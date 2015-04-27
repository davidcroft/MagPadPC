package magPadReference;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;

import processing.core.*;
import processing.serial.Serial;
import oscP5.*;
import netP5.*;

public class ProcessingSketch extends PApplet {
	private static final long serialVersionUID = 1L;
	
	private int fftCnt = 0;
	private int fftIndex = 0;
	private boolean startFFT = false;
	
	private boolean stopRecv = false;
	
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
	
	// GUI
	public static float[] guiData = new float[32];
	public static int[] maxValIdx = new int[2];
	private PFont font;
	private int locClass = 0;
	private double predictLocRow = 0;
	private double predictLocCol = 0;
	private String logStr = "";
	// sine wave
	float[][] sineValues;  // Using an array to store height values for the wave
	
	// save to file
	private float[][] calibratStdData = new float[6][32];
	private boolean loadStdData = false;
	private boolean showStdData = false;
	private int showStdDataIndex = 0;	// index for position
	
	// serial
	private Serial myPort;
	private int lf = 10;    // Linefeed in ASCII
	private String myString = null;
	
	public void setup() {
		// OSC
		// start oscP5, telling it to listen for incoming messages at port 5001 */
		OscProperties properties = new OscProperties();
		properties.setListeningPort(GlobalConstants.RECVPORT);
		properties.setDatagramSize(102400);
		oscP5 = new OscP5(this,properties);
		// set the remote location to be the localhost on port 3001
		remoteLocation = new NetAddress(GlobalConstants.SENDHOST,GlobalConstants.SENDPORT);
		
		// serial
		// List all the available serial ports:
		//System.out.println(Serial.list());
		// Open the port you are using at the rate you want:
		try {
			myPort = new Serial(this, GlobalConstants.SERIALPORT, 115200);
			myPort.clear();
			// Throw out the first reading, in case we started reading 
			// in the middle of a string from the sender.
			myString = myPort.readStringUntil(lf);
			myString = null;
		} catch (Exception e) {
			//e.printStackTrace();
			System.out.println("cannot open serial port!");
		}
		
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
	    
	    // GUI
	    frameRate(4);
	    maxValIdx[0] = maxValIdx[1] = 0;
	    font = createFont(Paths.get("files", "OpenSans-Regular.ttf").toAbsolutePath().toString(),16);
	    
	    // load calibratedData.csv
		loadStdData = loadStdDataFromFile();
		System.out.println("loadStdData = " + (loadStdData? "true":"false"));
		/*if (loadStdData) {
			for (int i = 0; i < 6; i++) {
				System.out.println("calibratStdData " + i);
				for (int k = 0; k < 3; k++) {
					for (int j = 0; j < 32; j++) {
						System.out.print(calibratStdData[i][k][j]+",");
					}
					System.out.println("\n");
				}
			}
		}*/
		
		// sine wave
		// init period
		int widthPts = 100;
		float heightPts = 25;
		sineValues = new float[6][widthPts];
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < widthPts; j++) {
				float x = TWO_PI * (i+1) * ((float)j/widthPts);
				sineValues[i][j] = sin(x)*heightPts;
				//println(sineValues[i][j]);
			}
		}
	}

	public void draw() {
		background(0);
		
		// FFT bands 
		int frameWidth = 510;
		int frameHeight = 660;
		float normRate = ((float)((height-200)/3))/120000;
		int offsetCol = frameWidth;
		int barWidth = offsetCol / 32;
		int offsetColLeft = (width-frameWidth)/2;
		
		int offsetRow = 100;
		// draw baseline
		stroke(0xFF,0xFF,0xF0);
		strokeWeight(1);
		line(offsetColLeft-10, offsetRow+1, offsetColLeft+offsetCol+10, offsetRow+1);
		
		// draw rect
		for(int i = 0; i < 32; i++) {
			int ratio = (int)map(guiData[i], 0, guiData[maxValIdx[0]], 0, 255);
			if (i == 4 || i == 5) {
				// motor 1
				//fill(0xFE, 0x6F, 0x5E, ratio);
				fill(0xFE, 0x6F, 0x5E);
			} else if (i == 19 || i == 20) {
				// motor 2
				//fill(0xff, 0x66, 0x33, ratio);
				fill(0xff, 0x66, 0x33);
			} else if (i == 10 || i == 11) {
				// motor 3
				//fill(0xff, 0xff, 0x99, ratio);
				fill(0xff, 0xff, 0x99);
			} else if (i == 7 || i == 8) {
				// motor 4
				//fill(0x99, 0xcc, 0x99, ratio);
				fill(0x99, 0xcc, 0x99);
			} else if (i == 16 || i == 17) {
				// motor 5
				//fill(0x33, 0x99, 0xcc, ratio);
				fill(0x33, 0x99, 0xcc);
			} else if (i == 13 || i == 14) {
				// motor 6
				//fill(0x66, 0x33, 0x66, ratio);
				fill(0x66, 0x33, 0x66);
			} else {
				// others
				fill(0xFF,0xFF,0xF0, ratio);
			}
			noStroke();
			rect(offsetColLeft+i*barWidth+1, offsetRow-guiData[i]*normRate, barWidth-2, guiData[i]*normRate);
				
			if (loadStdData && showStdData) {
			    noFill();
			    stroke(255);
			    rect(offsetColLeft+i*barWidth+1, offsetRow-calibratStdData[showStdDataIndex][i], barWidth-2, calibratStdData[showStdDataIndex][i]);
			    noStroke();
			}
	    }
		
		// jig(frame) display
		offsetCol = offsetColLeft;
		offsetRow = 130;
		frameWidth = 510;
		frameHeight = 660;
		int offset1 = 40;
		int offset2 = 200;
		int circleRadius = 60;
		int radius = 20;
		//fill(0x66, 0x99, 0xcc, 180);
		//fill(0x16, 0x40, 0x6B, 200);
		fill(0x33, 0x66, 0x99, 200);
		rect(offsetCol, offsetRow, frameWidth, frameHeight);
		
		// motor 1
		int moveOffset = 40;
		int x = offsetCol-offset2+circleRadius-10;
		int y = offsetRow+offset1+moveOffset;
		int pointOffset = 50;
		int ratio = (int)map(guiData[4], 0, guiData[maxValIdx[0]], 80, 220);
		fill(255);
		noStroke();
		ellipse(offsetCol+pointOffset, y-moveOffset, radius, radius);
		stroke(255);
		line(x+circleRadius+40, y, offsetCol+pointOffset-30, y);
		line(offsetCol+pointOffset-30, y, offsetCol+pointOffset, y-moveOffset);
		fill(0xFE, 0x6F, 0x5E, ratio);
		//fill(0x33, 0x66, 0x99, 200);
		noStroke();
		ellipse(x, y, circleRadius*2, circleRadius*2);
		if(ratio > 150) {
			noFill();
			stroke(0xFE, 0x6F, 0x5E, ratio);
			strokeWeight(3);
			ellipse(x, y, (circleRadius+10)*2, (circleRadius+10)*2);
		}
		if(ratio > 200) {
			noFill();
			stroke(0xFE, 0x6F, 0x5E, ratio);
			strokeWeight(5);
			ellipse(x, y, (circleRadius+25)*2, (circleRadius+25)*2);
		}
		renderWave(0, x-circleRadius+10, y);
		
		// motor 2
		x = offsetCol+frameWidth+offset2-100+circleRadius-10;
		y = offsetRow+offset1+moveOffset;
		ratio = (int)map(guiData[20], 0, guiData[maxValIdx[0]], 80, 220);
		fill(255);
		noStroke();
		ellipse(offsetCol+frameWidth-pointOffset, y-moveOffset, radius, radius);
		stroke(255);
		line(offsetCol+frameWidth-pointOffset, y-moveOffset, offsetCol+frameWidth-pointOffset+30, y);
		line(offsetCol+frameWidth-pointOffset+30, y, x-circleRadius-40, y);
		//fill(0x33, 0x66, 0x99, 200);
		fill(0xff, 0x66, 0x33, ratio);
		noStroke();
		ellipse(x, y, circleRadius*2, circleRadius*2);
		if(ratio > 150) {
			noFill();
			stroke(0xff, 0x66, 0x33, ratio);
			strokeWeight(3);
			ellipse(x, y, (circleRadius+10)*2, (circleRadius+10)*2);
		}
		if(ratio > 200) {
			noFill();
			stroke(0xff, 0x66, 0x33, ratio);
			strokeWeight(5);
			ellipse(x, y, (circleRadius+25)*2, (circleRadius+25)*2);
		}
		renderWave(5, x-circleRadius+10, y);
		
		// motor 3
		x = offsetCol+frameWidth+offset2-100+circleRadius-10+100;
		y = offsetRow+frameHeight/2;
		ratio = (int)map(guiData[11], 0, guiData[maxValIdx[0]], 80, 220);
		fill(255);
		noStroke();
		ellipse(offsetCol+frameWidth-pointOffset, y, radius, radius);
		stroke(255);
		line(offsetCol+frameWidth-pointOffset, y, x-circleRadius-40, y);
		//fill(0x33, 0x66, 0x99, 200);
		fill(0xff, 0xff, 0x99, ratio);
		noStroke();
		ellipse(x, y, circleRadius*2, circleRadius*2);
		if(ratio > 150) {
			noFill();
			stroke(0xff, 0xff, 0x99, ratio);
			strokeWeight(2);
			ellipse(x, y, (circleRadius+10)*2, (circleRadius+10)*2);
		}
		if(ratio > 200) {
			noFill();
			stroke(0xff, 0xff, 0x99, ratio);
			strokeWeight(3);
			ellipse(x, y, (circleRadius+25)*2, (circleRadius+25)*2);
		}
		renderWave(2, x-circleRadius+10, y);
		
		// motor 4
		x = offsetCol+frameWidth+offset2-100+circleRadius-10;
		y = offsetRow+frameHeight-offset1-moveOffset;
		ratio = (int)map(guiData[7], 0, guiData[maxValIdx[0]], 80, 220);
		fill(255);
		noStroke();
		ellipse(offsetCol+frameWidth-pointOffset, y+moveOffset, radius, radius);
		stroke(255);
		line(offsetCol+frameWidth-pointOffset, y+moveOffset, offsetCol+frameWidth-pointOffset+30, y);
		line(offsetCol+frameWidth-pointOffset+30, y, x-circleRadius-40, y);
		
		//fill(0x33, 0x66, 0x99, 200);
		fill(0x99, 0xcc, 0x99, ratio);
		noStroke();
		ellipse(x, y, circleRadius*2, circleRadius*2);
		if(ratio > 150) {
			noFill();
			stroke(0x99, 0xcc, 0x99, ratio);
			strokeWeight(2);
			ellipse(x, y, (circleRadius+10)*2, (circleRadius+10)*2);
		}
		if(ratio > 200) {
			noFill();
			stroke(0x99, 0xcc, 0x99, ratio);
			strokeWeight(3);
			ellipse(x, y, (circleRadius+25)*2, (circleRadius+25)*2);
		}
		renderWave(1, x-circleRadius+10, y);
		
		// motor 5
		x = offsetCol-offset2+circleRadius-10;
		y = offsetRow+frameHeight-offset1-moveOffset;
		ratio = (int)map(guiData[16], 0, guiData[maxValIdx[0]], 80, 220);
		fill(255);
		noStroke();
		ellipse(offsetCol+pointOffset, y+moveOffset, radius, radius);
		stroke(255);
		line(x+circleRadius+40, y, offsetCol+pointOffset-30, y);
		line(offsetCol+pointOffset-30, y, offsetCol+pointOffset, y+moveOffset);
		//fill(0x33, 0x66, 0x99, 200);
		fill(0x33, 0x99, 0xcc, ratio);
		noStroke();
		ellipse(x, y, circleRadius*2, circleRadius*2);
		if(ratio > 150) {
			noFill();
			stroke(0x33, 0x99, 0xcc, ratio);
			strokeWeight(2);
			ellipse(x, y, (circleRadius+10)*2, (circleRadius+10)*2);
		}
		if(ratio > 200) {
			noFill();
			stroke(0x33, 0x99, 0xcc, ratio);
			strokeWeight(3);
			ellipse(x, y, (circleRadius+25)*2, (circleRadius+25)*2);
		}
		renderWave(4, x-circleRadius+10, y);
		
		// motor 6
		x = offsetCol-offset2+circleRadius-10-100;
		y = offsetRow+frameHeight/2;
		ratio = (int)map(guiData[13], 0, guiData[maxValIdx[0]], 80, 220);
		fill(255);
		noStroke();
		ellipse(offsetCol+pointOffset, y, radius, radius);
		stroke(255);
		line(x+circleRadius+40, y, offsetCol+pointOffset, y);
		//fill(0x33, 0x66, 0x99, 200);
		fill(0x66, 0x33, 0x66, ratio);
		noStroke();
		ellipse(x, y, circleRadius*2, circleRadius*2);
		if(ratio > 150) {
			noFill();
			stroke(0x66, 0x33, 0x66, ratio);
			strokeWeight(2);
			ellipse(x, y, (circleRadius+10)*2, (circleRadius+10)*2);
		}
		if(ratio > 200) {
			noFill();
			stroke(0x66, 0x33, 0x66, ratio);
			strokeWeight(3);
			ellipse(x, y, (circleRadius+25)*2, (circleRadius+25)*2);
		}
		renderWave(3, x-circleRadius+10, y);
		
		// texts
		fill(0xFE,0x6F,0x5E);
		textFont(font, 32);
		offsetRow = 400;
		//offsetCol = width*3/5+130;
		offsetCol = 510; 
		if (isTrainedCls && isTrainedReg) {
			text("TEST", offsetCol, offsetRow);
		} else {
			if (!isTrainedReg) {
				text("TRAIN REG "+unTrainedList.get(0), offsetCol+35, offsetRow);
			} else if (!isTrainedCls){
				text("TRAIN CLS", offsetCol+35, offsetRow);
			}
		}
		
		fill(255);
		textFont(font, 16);
		offsetRow += 70;
		if (unTrainedList.isEmpty()) {
			text("Regression Untrained Blocks: None", offsetCol, offsetRow);
		} else {
			String temp = "";
			for (int i = 0; i < unTrainedList.size(); i++) {
				temp +=unTrainedList.get(i)+", ";
			}
			text("Regression Untrained Blocks: "+temp, offsetCol, offsetRow);
		}
		offsetRow += 30;
		if (isTrainedCls) {
			text("Classification: Trained", offsetCol, offsetRow);
		} else {
			text("Classification: Untrained", offsetCol, offsetRow);
		}
		offsetRow += 30;
		text("fftIndex: "+fftIndex, offsetCol, offsetRow);
		offsetRow += 30;
		text("fftCnt: "+fftCnt, offsetCol, offsetRow);
		
		offsetRow += 50;
		if (isTrainedCls) {
			text("Classification: "+locClass, offsetCol, offsetRow);
		} else {
			text("Classification: -", offsetCol, offsetRow);
		}
		offsetRow += 30;
		if (isTrainedCls && isTrainedReg) {
			text("predictLocRow: "+predictLocRow, offsetCol, offsetRow);
			offsetRow += 30;
			text("predictLocCol: "+predictLocCol, offsetCol, offsetRow);
		} else {
			text("predictLocRow: -", offsetCol, offsetRow);
			offsetRow += 30;
			text("predictLocCol: -", offsetCol, offsetRow);
		}
		
		offsetRow += 50;
		textFont(font, 12);
		text("LOG: "+logStr, offsetCol, offsetRow);
		
		// text
		textFont(font, 16);
		offsetCol = (width-frameWidth)/2 + frameWidth - 160;
		offsetRow = 50;
	    text("Max bins index: "+maxValIdx[0]+", "+maxValIdx[1], offsetCol, offsetRow);
	    textFont(font, 48);
	    fill(0xFE, 0x6F, 0x5E);
	    text("FFT", (width-frameWidth)/2 - 180, 100);
	    
	    
	    // serial
	    try {
	    	while (myPort.available() > 0) {
				myString = myPort.readStringUntil(lf);
			    if (myString != null) {
			      println(myString);
			      logStr = myString;
			    }
			}
		} catch (Exception e) {
		}
	}

	void renderWave(int index, float offsetX, float offsetY) {
		stroke(255);
		strokeWeight(1);
		// A simple way to draw the wave with an ellipse at each location
		for (int x = 0; x < sineValues[index].length-1; x++) {
			line(offsetX+x, offsetY+sineValues[index][x], offsetX+x+1, offsetY+sineValues[index][x+1]);
		}
	}
	
	public void keyPressed() {
		if (key == 'a') {
			// remove last collection of data
			logStr = "remove last capture pts";
			System.out.println("Size before removal: "+GlobalConstants.trainingColSet.size());
			int cnt = GlobalConstants.FFTFOREACHPOS+1;
			if (GlobalConstants.trainingColSet.size() > 0 && GlobalConstants.trainingColSet.size()%cnt == 0 && 
				GlobalConstants.trainingRowSet.size() > 0 && GlobalConstants.trainingRowSet.size()%cnt == 0) {
				while(cnt > 0) {
					GlobalConstants.trainingRowSet.remove(GlobalConstants.trainingRowSet.size()-1);
					GlobalConstants.trainingColSet.remove(GlobalConstants.trainingColSet.size()-1);
					cnt -= 1;
				}
				fftIndex -= 1;
			}
			System.out.println("Size after removal: "+GlobalConstants.trainingColSet.size());
		} else if (key == 'd') {
			logStr = "remove last capture pts";
			System.out.println("Size before removal: " + GlobalConstants.trainingClsSet.size());
			//int cnt = GlobalConstants.FFTCLSFOREACHPOS+1;
			int cnt = 20;
			//if (GlobalConstants.trainingClsSet.size() > 0 && GlobalConstants.trainingClsSet.size()%cnt == 0) {
			while(cnt > 0 && fftCnt > 0) {
				GlobalConstants.trainingClsSet.remove(GlobalConstants.trainingClsSet.size()-1);
				cnt -= 1;
				fftCnt -= 1;
			}
			System.out.println("Size after removal: " + GlobalConstants.trainingClsSet.size());
		} else if (key == 's') {
			logStr = "save CSV files";
			if (isTrainingIndex >= 0 && isTrainingIndex < GlobalConstants.TRAININGCLSNUM) {
				generateCsvFile(isTrainingIndex);
				System.out.println("generate CSV file for regression block " + isTrainingIndex);
			} else if (isTrainingIndex == -1) {
				generateClsCsvFile();
				System.out.println("generate CSV file for classification");
			}
			System.out.println("save csv file");
		}
		
		
		if (key == '1') {
			showStdData = true;
			showStdDataIndex = 0;
		} else if (key == '2'){
			showStdData = true;
			showStdDataIndex = 1;
		} else if (key == '3') {
			showStdData = true;
			showStdDataIndex = 2;
		} else if (key == '4') {
			showStdData = true;
			showStdDataIndex = 3;
		} else if (key == '5') {
			showStdData = true;
			showStdDataIndex = 4;
		} else if (key == '6') {
			showStdData = true;
			showStdDataIndex = 5;
		} else if (key == '0') {
			showStdData = false;
		}
		
		if (key == 'u') {
			myPort.write("@-1s#");
			System.out.println("Motor 1 <");
		} else if (key == 'i') {
			myPort.write("@+1s#");
			System.out.println("Motor 1 >");
		} else if (key == 'o') {
			myPort.write("@-2s#");
			System.out.println("Motor 2 <");
		} else if (key == 'p') {
			myPort.write("@+2s#");
			System.out.println("Motor 2 >");
		} else if (key == 'k') {
			myPort.write("@-3s#");
			System.out.println("Motor 3 <");
		} else if (key == 'l') {
			myPort.write("@+3s#");
			System.out.println("Motor 3 >");
		} else if (key == 'n') {
			myPort.write("@-4s#");
			System.out.println("Motor 4 <");
		} else if (key == 'm') {
			myPort.write("@+4s#");
			System.out.println("Motor 4 >");
		} else if (key == 'v') {
			myPort.write("@-5s#");
			System.out.println("Motor 5 <");
		} else if (key == 'b') {
			myPort.write("@+5s#");
			System.out.println("Motor 5 >");
		} else if (key == 'h') {
			myPort.write("@-6s#");
			System.out.println("Motor 6 <");
		} else if (key == 'j') {
			myPort.write("@+6s#");
			System.out.println("Motor 6 >");
		}
		
		if (key == 'z') {
			stopRecv = !stopRecv;
		}
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
		//println("receive a message");
		if (stopRecv) return;
		
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
		    	locClass = SVMClassfier.classifyGesture(testDataRow);
		    	//System.out.println("predict block class: " + locClass);
		    	// predict location using corresponding regression
		    	double predictLocRowTemp = SVMRegressionRow[locClass].classifyGesture(testDataRow);
		    	double predictLocColTemp = SVMRegressionCol[locClass].classifyGesture(testDataRow);
		    	
		    	// normalize location to A4 size
		    	predictLocRow = predictLocRowTemp * GlobalConstants.MAXHEIGHT;
		    	predictLocCol = predictLocColTemp * GlobalConstants.MAXWIDTH;
		    	//System.out.println("predict location: " + predictLocRow + " " + predictLocCol);
		    	
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
	
	private boolean loadStdDataFromFile() {
		String filePath = Paths.get("files", "calibratedData.csv").toAbsolutePath().toString();
		// check if model has been trained in the system
		File f = new File(filePath);
		if(!(f.exists() && !f.isDirectory())) {
			return false;
		}
		
		BufferedReader br = null;  
		String line = "";
		try 
		{
			int lineIdx = 0;
			br = new BufferedReader(new FileReader(filePath));  
			while ((line = br.readLine()) != null) { 
				String[] vals = line.split(",");  
			    for (int i = 0; i < 32; i++) {
			    	calibratStdData[lineIdx][i] = Float.parseFloat(vals[i]);
			    }
			    lineIdx += 1;
			}
		}
		catch (FileNotFoundException e) {  
			e.printStackTrace();  
		} catch (IOException e) {  
			e.printStackTrace();  
		} finally {  
			if (br != null) {  
				try {  
					br.close();  
				} catch (IOException e) {  
					e.printStackTrace();    
			   	}  
			}  
			System.out.println("Done with reading CSV");  
		}
		return true;
	}
}
