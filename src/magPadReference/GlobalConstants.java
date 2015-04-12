package magPadReference;

import java.util.ArrayList;

public class GlobalConstants {
	// OSC
	public static String SENDHOST = "128.237.178.5";
	//public static String SENDHOST = "169.254.202.224";
	//public static String SENDHOST = "10.0.0.11";
	public static final int SENDPORT = 3001;
	public static final int RECVPORT = 3000;
	
	// FFT
	public static final int BUFFERSIZE = 64;
	public static final int SAMPLERATE = 100;
	public static final int FFTFOREACHPOS = 20;
	//public static final int FFTCLSFOREACHPOS = 400;
	
	// ML Training
	public static final int TRAINTOTALROW = 2;
	public static final int TRAINTOTALCOL = 2;
	public static final int FFTCLSFOREACHPOS = 20;
	
	/*public static final int TRAINTOTALROW = 14;
	public static final int TRAINTOTALCOL = 7;*/
	public static final int TRAININGPOSNUM = TRAINTOTALROW*TRAINTOTALCOL;	// number of positions for training
	public static final int TRAININGCLSNUM = 4;		// number of classification for training
	
	public static final int NNINPUTNUM = 114+360;
	public static final int NNOUTPUTNUM = 1;
	
	public static ArrayList<String> trainingRowSet = new ArrayList<String>();
	public static ArrayList<String> trainingColSet = new ArrayList<String>();
	public static ArrayList<String> trainingClsSet = new ArrayList<String>();
	public static ArrayList<float[]> testingSet = new ArrayList<float[]>();
	
	public static final double MAXWIDTH = 8.5;			// width 7.9 inch
	public static final double MAXHEIGHT = 11;  		// height 9.4 inch
}
