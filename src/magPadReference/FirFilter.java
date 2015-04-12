package magPadReference;

import java.util.ArrayList;
import java.util.List;
import java.io.IOException;
import java.nio.file.*;
import java.nio.charset.*;

public class FirFilter {

	public FirFilter(double coefficients[]) {
		this.h = coefficients;
	    this.N = coefficients.length;            
	    this.x = new double[N];
	    System.out.println("create a filter");
	}

	public FirFilter(String fileName) {
	    this.h = loadCoefficients(Paths.get("files", fileName).toAbsolutePath().toString());
	    this.N = h.length;          
	    this.x = new double[N];
	    System.out.println("create a filter");
	}

	public double filter(double newSample) {       
	    y = 0;
	    // x maintains an array with the same size of length of filter
	    x[iWrite] = newSample;      
	    iRead = iWrite;
	    for (n=0; n<N; n++) {
	    	y += h[n] * x[iRead];
	    	iRead++;
	    	if (iRead == x.length) {
	    		iRead = 0;
	    	}
	    }
	    iWrite--;
	    if (iWrite < 0) {
	    	iWrite = x.length-1;
	    }      
	    return y;
	}

	private double[] loadCoefficients(String fileName) {
		List<String> lines = null;
		try {
			lines = Files.readAllLines(Paths.get(fileName), StandardCharsets.UTF_8);
		} catch (IOException e) {
			e.printStackTrace();
		}
		//String lines[] = loadStrings(fileName);
		ArrayList<Double> list= new ArrayList<Double>();

	    for (int i = 0 ; i < lines.size(); i++) {
	    	list.add (new Double (lines.get(i)));
	    }

	    double[] result= new double[list.size()];
	    for (int i=0; i<list.size(); i++) {
	    	result[i]= list.get(i).doubleValue();
	    }
	    return result;
	}

	private int N;
	private double h[];
	private double y;
	private double x[];
	private int n;
	private int iWrite = 0;
	private int iRead = 0;
}