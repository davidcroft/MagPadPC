package magPadReference;

import java.util.ArrayList;

import processing.core.*;
import ddf.minim.analysis.*;

public class FFTThreadCls implements Runnable {
	// thread
	private Thread thread;
	private int pauseTime;  // Time to wait between loops
	private PApplet m_parent;
	
	// FFT
	private FFT fftX;
	private FFT fftY;
	private FFT fftZ;
	
	private float[] fft_ptx;
	private float[] fft_pty;
	private float[] fft_ptz;
	private int fft_size;
	private String m_cls;
	//private int m_fftIndex;
	private int m_fftCnt;
	
	// neural netwrok
	private boolean m_isTrained;
	private final int NORMMAXVAL = 12000;
	
	//String TRAINFILEPATH = "files/test.csv";
	//String TRAINFILEPATH = "files/train.csv";

	public FFTThreadCls(PApplet parent, int pt, int fftSize, int sampleRate, float[] ptX, float[] ptY, float[] ptZ, String label, int fftCnt) {
		// pass parameter from parent
		m_parent = parent;
		//m_fftIndex = index;
		m_cls = label;
		m_isTrained = false;
		m_fftCnt = fftCnt;
		
		// thread setting
		//m_parent.registerDispose(this);
		pauseTime = pt;
		  
		// init fft
		fftX = new FFT(fftSize, sampleRate);
		fftY = new FFT(fftSize, sampleRate);
		fftZ = new FFT(fftSize, sampleRate);
		fft_ptx = ptX;
		fft_pty = ptY;
		fft_ptz = ptZ;
		fft_size = fftSize;
	}

	public void start() {
		thread = new Thread(this);
		thread.start();
	}

	public void run() {
		// attach a window to fft data
		for (int i = 0; i < fft_size; i++) {
			// attach a window to sample data in order to avoid frequency leakage
			//float window = 0.5 - 0.5*cos(2*3.141593*i/(fft_size-1));
			//float window = 0.35875 + 0.48829*cos(2*3.141593*i/(fft_size-1)) + 0.14128*cos(4*3.141593*i/(fft_size-1)) + 0.01168*cos(6*3.141593*i/(fft_size-1));
			float window = (float) (0.42 - 0.5*Math.cos(2*3.141593*i/(fft_size-1)) - 0.08*Math.cos(4*3.141593*i/(fft_size-1))); 
			fft_ptx[i] = fft_ptx[i] * window;
			fft_pty[i] = fft_pty[i] * window;
			fft_ptz[i] = fft_ptz[i] * window;
		}
	    
		// FFT
		fftX.forward(fft_ptx);
		fftY.forward(fft_pty);
		fftZ.forward(fft_ptz);
			    
		/////////////////////////////////////
		// normalization
		float[] fft = new float[fftX.specSize()];
	    for (int i = 0; i < fftX.specSize(); i++) {
	      fft[i] = fftX.getBand(i) + fftY.getBand(i) + fftZ.getBand(i);
	    }
	    
	    // find top 2
	    int[] maxValIdx = new int[2];
	    maxValIdx[0] = maxValIdx[1] = 3;
	    for (int i = 3; i < fft.length; i++) {
	    	if (fft[i] > fft[maxValIdx[0]]) {
	    		maxValIdx[1] = maxValIdx[0];
	    		maxValIdx[0] = i;
	    	} else if (fft[i] > fft[maxValIdx[1]]) {
	    		maxValIdx[1] = i;
	    	}
	    }
	    ProcessingSketch.maxValIdx[0] = maxValIdx[0];
	    ProcessingSketch.maxValIdx[1] = maxValIdx[1];
	    
	    for(int i = 0; i < 32; i++) {
	    	ProcessingSketch.guiData[i] = fft[i];
	    }
		//////////////////////////////////////////////////////////

				
		// append data to trainingSet
		if (!m_isTrained) {
			
			// CREATE A FEATURE VECTOR
			
			// add an item in trainingSet for model training
			float[] item = generateFeatureVector();
			String rowItem = new String();
			for (int i = 0; i < item.length; i++) {
				rowItem += String.valueOf(item[i]) + ",";
			}
			rowItem += m_cls + "\n";
			
			GlobalConstants.trainingClsSet.add(rowItem);

			System.out.println(m_fftCnt + "insert a row in classification set, output: label: " + m_cls);

		} else {
			
			// PREDICTING
			
			// add an item in testingSet for predicting
			float[] item = generateFeatureVector();
			GlobalConstants.testingSet.add(item);
			//System.out.println("insert a row in testing set");
		}
	    
		// Wait a little (give other threads time to run)
		try {
			Thread.sleep(pauseTime);
		} catch (InterruptedException e) {}
	}

	public void stop() {
		System.out.println("thread stop");
		thread = null;
	}

	// this will magically be called by the parent once the user hits stop 
	// this functionality hasn't been tested heavily so if it doesn't work, file a bug 
	public void dispose() {
		stop();
	}
	
	
	////////////////////////////////////////////////////////////////////
	// extract FFT spectrum to extract a feature vector	
	float[] generateFeatureVector() {
		
		float[] features = new float[GlobalConstants.NNINPUTNUM];
		// raw bands data, capture one band for each magnet
		// 32*3 features
		int featureIdx = 0;
		for (int i = 0; i < 3; i++) {
			// 3 axis: x, y, z
			for (int j = 0; j < fft_size/4; j++) {
				//System.out.println("featureIdex: " + featureIdx + " bandIdx: " + bandIdx);
				if (i == 0) {
					features[featureIdx++] = fftX.getBand(j)/NORMMAXVAL;
				} else if (i == 1) {
					features[featureIdx++] = fftY.getBand(j)/NORMMAXVAL;
				} else {
					features[featureIdx++] = fftZ.getBand(j)/NORMMAXVAL;
				}
			}
		}
		
		// band ratios
		for (int i = 0; i < 3; i++) {
			// 3 axis: x, y, z
			int start = 5;
			int end = 20;
			for (int j = start; j <= end-1; j++) {
				for (int k = j+1; k <= end; k++) {
					if (i == 0) {
						features[featureIdx++] = (fftX.getBand(j)/NORMMAXVAL) / (fftX.getBand(k)/NORMMAXVAL);
					} else if (i == 1) {
						features[featureIdx++] = (fftY.getBand(j)/NORMMAXVAL) / (fftY.getBand(k)/NORMMAXVAL);
					} else {
						features[featureIdx++] = (fftZ.getBand(j)/NORMMAXVAL) / (fftZ.getBand(k)/NORMMAXVAL);
					}
				}
			}
		}
		
		// Statistical info
		float[] mean = new float[3];
		float[] var  = new float[3];
		float[] maxBand = new float[3];
		float[] maxBandIdx = new float[3];
		float[] kurtosis = new float[3];
		float[][] fftData = new float[3][fft_size/4];
		// init
		mean[0] = mean[1] = mean[2] = 0;
		var[0] = var[1] = var[2] = 0;
		maxBand[0] = maxBand[1] = maxBand[2] = 0;
		maxBandIdx[0] = maxBandIdx[1] = maxBandIdx[2] = 0;
		for (int i = 1; i <= fftData[0].length ; i++) {
			fftData[0][i-1] = fftX.getBand(i)/NORMMAXVAL;		// x
			fftData[1][i-1] = fftY.getBand(i)/NORMMAXVAL;		// y
			fftData[2][i-1] = fftZ.getBand(i)/NORMMAXVAL;		// z
			// mean
			for (int j = 0; j < 3; j++) {
				mean[j] += fftData[j][i-1];
				if(fftData[j][i-1] > maxBand[j]) {
					maxBand[j] = fftData[j][i-1];
					maxBandIdx[j] = i;
				}
			}
		}
		
		// compute mean and add to feature vector
		// 3 features
		for (int i = 0; i < 3; i++) {
			mean[i] /= fftData[0].length;
			features[featureIdx++] = mean[i];
		}
		
		// compute mean difference and add to feature vector
		// 3 features
		features[featureIdx++] = mean[0] - mean[1];
		features[featureIdx++] = mean[1] - mean[2];
		features[featureIdx++] = mean[2] - mean[0];
		
		// add maxBandIdx into feature vector
		// 6 features
		for (int i = 0; i < 3; i++) {
			features[featureIdx++] = maxBandIdx[i]/fftData[0].length;
			features[featureIdx++] = maxBand[i];
		}
		
		// compute var and add to feature vector
		// 3 features
		for (int i = 0; i < fftData[0].length; i++) {
			for (int j = 0; j < 3; j++) {
				var[j] += Math.pow(fftData[j][i] - mean[j], 2);
				kurtosis[j] += Math.pow(fftData[j][i] - mean[j], 4);
			}
		}
		for (int i = 0; i < 3; i++) {
			var[i] /= fftData[0].length;
			features[featureIdx++] = (float) Math.sqrt(var[i]);
		}
		
		// compute Kurtosis
		// 3 features
		for (int i = 0; i < 3; i++) {
			kurtosis[i] = ((kurtosis[i] / fftData[0].length) / (var[i] * var[i]))/100;
			features[featureIdx++] = kurtosis[i];
		}
		
		// debug
		//for (int i = 0; i < 114; i++) System.out.println(features[i]);
		return features;
	}
}